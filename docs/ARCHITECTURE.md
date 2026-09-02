# Architecture

How diar-native is put together, what it is built on, and why. For running it, start at the
[README](../README.md).

- [Why this exists](#why-this-exists)
- [Why speakrs](#why-speakrs-and-why-not-the-alternatives)
- [How community-1 diarization works](#how-community-1-diarization-works)
- [Repository layout](#repository-layout)
- [Deployment tiers](#deployment-tiers)
- [One image, CPU and CUDA](#one-image-cpu-and-cuda)
- [Production consumes the binary, not the image](#production-consumes-the-binary-not-the-image)
- [Logging](#logging)
- [Apple Silicon](#apple-silicon-native-coreml)

---

## Why this exists

diar-native is the native (Rust/ONNX) speaker diarization engine for OpenTranscribe, built on a
vendored, heavily-patched [`speakrs`](https://github.com/avencera/speakrs) (upstream) — mirrored
at [`attevon-llc/speakrs`](https://github.com/attevon-llc/speakrs) (our fork, Apache-2.0 licence
unchanged) — matching the **pyannote `speaker-diarization-community-1`** pipeline's accuracy at a
fraction of its wall time and deployment weight, with a clean path to **Triton Inference Server**
and **AWS GPU** serving.

OpenTranscribe (`transcribe-app`) diarizes with a **pinned fork** of pyannote.audio
(`davidamacey/pyannote-audio` @ `gpu-optimizations` / `a3f38afb`) that already carries heavy GPU
optimization work (pinned memory, CUDA stream prefetch, VRAM-budgeted batching, −66% VRAM).
It works well, but the Python stack has structural ceilings:

- **GIL + single process**: clustering (VBx, scipy/numpy) is ~41% of wall time on a 4.7 h file
  and serializes with everything else; whisper and diarization fight for the same worker.
- **Deployment weight**: ~9 GB image with full torch — painful for AWS instances and for the
  open-source install matrix (CPU-only boxes; Apple Silicon now has a native CoreML path).
- **No serving story**: models are locked inside a Celery worker process; no dynamic batching, no
  cross-job GPU sharing, no independent scaling of the 53%-of-GPU-time embedding stage.

Goals, in order:

1. **Accuracy parity** with the production fork (hard gates — see
   [`validation/TESTPLAN.md`](../validation/TESTPLAN.md)).
2. **Faster end-to-end** diarization on the same hardware (beefy home servers: 2× RTX A6000).
3. **Lighter, portable deployment**: single native binary + ONNX Runtime; CUDA / CPU-only /
   CoreML backends for the open-source deployment matrix.
4. **Serving-ready**: the same ONNX artifacts servable by Triton (onnxruntime / tensorrt
   backends) for AWS scale-out; orchestration embeddable as a Triton backend later (Triton's
   backend API is C — Rust implements it directly via `diar-ffi`).

## Why speakrs (and why not the alternatives)

[`speakrs`](https://github.com/avencera/speakrs) (Apache-2.0, Rust) implements the **full
community-1 pipeline**: powerset decode, overlap-add aggregation, binarization,
speaker-conditioned embeddings, **PLDA + VBx clustering (AHC init)** — fixture-tested against
scipy/pyannote at 1e-4. Its export script loads the actual
`pyannote/speaker-diarization-community-1` pipeline (we re-export ourselves from the production
HF cache — see [`validation/RESULTS.md`](../validation/RESULTS.md) §4).

| Candidate | Verdict | Reason |
|---|---|---|
| **speakrs** (Rust) | **primary candidate** | full community-1 fidelity incl. VBx; ORT CPU/CUDA/MIGraphX + CoreML; active; Apache-2.0 |
| sherpa-onnx (C++) | rejected | simpler clustering, no VBx → documented accuracy drift vs pyannote (k2-fsa/sherpa-onnx#1708) |
| pyannote-rs (Rust) | rejected | 80.2% DER on VoxConverse subset; no aggregation, cosine-threshold clustering |
| custom C++/Rust from scratch | fallback-of-last-resort | months to re-earn accuracy speakrs already demonstrates; zero inference-speed advantage (same ORT kernels) |

**"Wouldn't C++ be faster?"** No: in every candidate the neural nets execute inside ONNX
Runtime's C++ kernels; orchestration math in Rust compiles to the same machine-code class as
C++. The differentiator is *algorithm fidelity* (VBx), not language. Triton compatibility is
equal too — its custom-backend interface is a **C API**, which a Rust `cdylib` implements without
any C++ shim.

### The vendored patch set

`vendor/speakrs/` is an upstream clone pinned at `b0756b1` with our patches carried as the
**working-tree diff**, exported to `patches/0001-cuda-performance-patch-set.patch`. The patch set
covers folded segmentation, the multimask-batching fix, the fbank pool, VBx vectorization, the
exclusive-overlap fix, shared sessions and the pipelined fbank∥GPU stage.

`scripts/bootstrap_vendor_speakrs.sh` reproduces the vendored tree elsewhere: it clones our fork
(`attevon-llc/speakrs`, Apache-2.0 unchanged) at a pinned commit, verified byte-identical to the
vendored tree with 94/94 speakrs tests passing from a clean clone. `avencera/speakrs` remains the
canonical upstream for PRs; the contribution queue is in
[`UPSTREAM_PRS.md`](UPSTREAM_PRS.md).

## How community-1 diarization works

```
waveform 16 kHz mono
  └─ 1. Segmentation  (PyanNet: SincNet conv → 4-layer BiLSTM → linear → 7-class powerset)
        sliding 10 s window, 1 s step → (num_chunks, 589 frames, 7) log-probs        [NN — GPU]
  └─ 2. Powerset → multilabel  argmax → mapping → (chunks, 589, 3 speakers) binary   [trivial ops]
  └─ 3. Speaker counting  (trim + overlap-add aggregate)                             [numpy]
  └─ 4. Embeddings  (WeSpeaker ResNet34: kaldi fbank 80-mel → ResNet → weighted
        stats-pool → 256-d), one per (chunk × local speaker), masked                 [NN — GPU]
  └─ 5. VBx clustering  (filter → L2-norm → AHC seed (centroid linkage, thr 0.6) →
        PLDA transform → VB-EM (Fa 0.07, Fb 0.8) → centroids → cosine soft-assign →
        per-chunk Hungarian)                                                         [numpy/scipy]
  └─ 6. Reconstruct → aggregate → binarize → Annotation (+ exclusive variant)        [numpy]
```

Only stages **1 and 4 are neural networks** — they export to ONNX cleanly *when done right*.
Stages 2/3/5/6 are ordinary code (Rust in speakrs, numpy in pyannote) and **can never be a single
ONNX graph** — that was root-cause #1 of the previous "export the pipeline to ONNX" failure.

The other historical blockers, all root-caused and fixed (details + evidence in
[`validation/RESULTS.md`](../validation/RESULTS.md) §1/§4):

- *"ORT CUDA has no Sin/Cos kernels"* → SincNet synthesizes conv filters from **frozen** params;
  onnxsim constant-folding removes Sin/Cos/If **bit-exactly** (179 → 40 nodes, 2.0× measured).
- *"TensorRT rebuilds engines per shape"* → the old shape profile declared max 500 fbank frames
  vs the real 998; with correct fixed shapes, batch is the only dynamic axis.
- *"Embedding export fails on torchaudio fbank"* → export the ResNet with fbank outside the
  graph, or fuse a decomposed fbank (framing + DFT/mel matmuls) like speakrs does. (Note: the
  fused variant carries a measured ~6.7× ORT-CUDA cost — RESULTS §4.2 — so fbank-outside +
  batched native fbank is the perf-correct split.)

## Repository layout

```
diar-native/
├── README.md                  ← what it is, how to run it, where to read more
├── install.sh                 ← one-command installer (platform detect → provision → serve)
├── CHANGELOG.md               ← release history (Keep a Changelog)
├── CONTRIBUTING.md            ← how to build/test/PR; reproduces the CI environment locally
├── SECURITY.md                ← vulnerability reporting + the gated-weights policy
├── MODELS_SETS.md             ← fast vs small: what differs and why (matches files.rs)
├── start.sh                   ← dev wrapper: platform detect, token prompt, build/pull, serve
├── docker-compose.yml         ← source-build deployment (provision behind a profile)
├── docker-compose.prod.yml    ← published-image deployment (no build: key, named volumes)
├── docker-compose.gpu.yml     ← the NVIDIA device reservation, as a separate overlay
├── .github/                   ← CI, dependabot, PR template, setup-build-env action
├── docs/                      ← this directory; see the README's link table
├── validation/
│   ├── TESTPLAN.md            ← test matrix, gates, methodology, reproduction commands
│   ├── RESULTS.md             ← every measurement (append-only log; never re-run a logged test)
│   ├── run_fork_baseline.py   ← Engine A runner (backend image, fork bind-mount, RTTM+timing out)
│   ├── run_speakrs.sh         ← Engine B runner (diar-bench image, RTTM per file/run)
│   ├── score_der.py           ← DER scorer (UEM-aware, aggregate + per-file, JSON out)
│   ├── ort_cuda_microbench.py ← ORT CUDA EP folded-vs-unfolded-vs-eager microbench
│   └── triton_bench.py        ← Triton gRPC latency/throughput bench
├── crates/
│   ├── diar-core/             ← speakrs wrapper: shared-session engine handles (clone_shared),
│   │                            centroids, embed_window, exclusive output, gender, media decode,
│   │                            logging policy shared by both binaries (logging.rs),
│   │                            ort_compat.rs (per-platform session workarounds), and
│   │                            provision/ (the models exporter, marker and smoke test)
│   ├── diar-server/           ← the T1 sidecar: /diarize /embed_window /healthz /readyz (axum);
│   │                            per-request engine handles — jobs run concurrently;
│   │                            structured logs to stdout (RUST_LOG, DIAR_LOG_FORMAT);
│   │                            engines.rs = device registry; cli.rs = the four subcommands
│   └── diar-cli/              ← bench/ops runner (RTTM+JSON out, engine traces via RUST_LOG
│                                on stderr — stdout stays parseable JSONL)
├── docker/
│   ├── Dockerfile.server      ← production CUDA image (ORT GPU libs)
│   ├── Dockerfile.server-cpu  ← multi-arch CPU-only image (linux/amd64 + linux/arm64)
│   ├── Dockerfile.provision   ← pinned CPU-only torch env, for provisioning off the plain image
│   ├── Dockerfile.builder     ← the build container (also what CI reproduces)
│   └── Dockerfile.bench       ← self-contained speakrs CUDA build (xtask diarize CLI)
├── patches/0001-…patch        ← THE vendored speakrs diff (regenerate after any vendor edit)
├── scripts/                   ← release.sh, setup_dev_env.sh, bootstrap_vendor_speakrs.sh, …
├── triton/models/             ← Triton spike model repo (config.pbtxt per model)
├── refs/                      ← staged references (AMI test RTTM+UEM, Karpathy fixed RTTM)
├── models*/                   ← [gitignored] self-exported community-1 ONNX + PLDA (gated weights)
│                                models_folded/=fast set (default), models_small/=laptop set
├── vendor/speakrs/            ← [gitignored] upstream clone @ pin b0756b1 + our working-tree
│                                patch set (reproduce: scripts/bootstrap_vendor_speakrs.sh)
├── upstream-work/             ← [gitignored] upstream-tip clone holding the prepared PR branches
└── results/                   ← RTTMs, timing JSONL, DER JSONs per run tag
```

### Crate responsibilities

- **`diar-core`** — the engine wrapper. `DiarEngine::clone_shared()` hands out per-request
  handles over `Arc`-shared ORT sessions, so weights and arenas are loaded once and concurrent
  jobs cost one engine's VRAM. Also centroids, `embed_window`, exclusive segments, gender,
  `audio.rs` (media decode), `logging.rs` (the `RUST_LOG` / `DIAR_LOG_FORMAT` policy both
  binaries share — the sink is a parameter and the library never installs a subscriber itself),
  `ort_compat.rs` (per-platform session workarounds), `shutdown.rs`, and `provision/`.
  **Provisioning lives here, not in diar-server**, because diar-server is a binary crate with no
  `tests/` — nothing in it would be integration-testable.
- **`diar-server`** — the sidecar. Four routes, plus the `provision-models` / `verify-models` /
  `check-token` subcommands in `cli.rs`. No subcommand = serve, which the deployment relies on.
- **`diar-cli`** — the bench/ops runner. Not published as an image; `start.sh --cli` builds it.

### Both binaries must never return

Every exit goes through `diar_core::shutdown::exit` / `exit_main`, which terminate via `_exit`.
Returning a `Result` from `main` — or calling `std::process::exit` — lets libc run ort's
`.fini_array` destructor, which **logs** the drop of its global `Environment` after the main
thread's TLS is gone; with a subscriber installed and `ort::lifetime` at TRACE that aborts the
process *after* the work is written and the output flushed. It cost `diar-cli` exit 134 (139 in a
container, where it is PID 1 — the same SIGABRT) and made `diar-server` answer a port conflict
with an abort instead of "address already in use".

Only a directive enabling `ort::lifetime` at TRACE triggers it: `RUST_LOG=trace` does,
`RUST_LOG=speakrs=trace` does not. It is not fixable downstream — ort's `G_ENV` is private and
strong, and the tracing dispatcher cannot be uninstalled. Guarded by
`crates/diar-core/tests/shutdown_teardown.rs`. Full writeup:
[`ORT_ATEXIT_TEARDOWN.md`](ORT_ATEXIT_TEARDOWN.md).

## Deployment tiers

Product decision, 2026-08-19.

| tier | target | design |
|---|---|---|
| **T1 — embedded/sidecar (DEFAULT, open source — SHIPPED)** | laptops, small computers, single-GPU boxes | speakrs core + our patch set: one model set (Arc-shared ORT sessions, weights + arenas loaded once) + per-request scratch handles — concurrent jobs at one engine's VRAM, verified output-identical to serial. Ingests original media directly (symphonia + FFT resample). **One image serves both `cuda` and `cpu`, selected per request.** `SPEAKRS_ARENA_SHRINK=1` drops the between-job VRAM floor 4.5 GB → 1.1 GB for small cards (~20% per-job cost). No Triton, minimal RAM. |
| **T2 — Triton (opt-in, larger home servers + AWS)** | multi-user / multi-job servers | tritonserver + TRT engines (fp32), dynamic batching across concurrent jobs (measured 2.14× throughput at 8 clients on one weight copy), `diar-ffi` custom backend or sidecar-calls-Triton topology. Higher RAM/system footprint — that is why it is opt-in, not default. |

The Python fork path remains the universal fallback behind a config flag in both tiers.

An in-`ort` TensorRT EP for T1 was implemented, measured (1.33-1.48× at +0.03 pp AMI DER) and
**rolled back** — the compatibility surface was not worth speed that hides behind transcription
(RESULTS §7.26 is the recipe if that changes).

**VRAM budgets per tier, and what actually fits, are measured in
[VRAM_AND_TIERS.md](VRAM_AND_TIERS.md).** The headline for planning: the warm stack holds a
**7 575 MiB floor** on a 12 GB card (sidecar 4 136 + whisper 2 038 + redaction 1 346) and each
concurrent job adds only ~490 MiB, because weights are shared and only activations scale. The
floor is warm-start caching, not a leak — it returns to the same figure after every job. The
sidecar's share is ~547 MiB of weights plus ~3.6 GB of ORT arena acquired on first inference and
never released, so **T1's "laptops" claim holds for 6-8 GB cards but not yet for 4 GB**: that
needs redaction off the GPU, the small model set, and sidecar load/unload — which trades away the
transcribe∥diarize overlap.

## One image, CPU and CUDA

The CUDA image is a **superset** of the CPU image on amd64, and always was — we just had no way
to ask for CPU after startup. The ORT CPU execution provider is *statically linked* into the
binary by `ort-sys` (the `onnxruntime_mlas` kernel library is part of the core static objects),
which is why `ldd diar-server` reports no ONNX Runtime `NEEDED` entry at all. `--features cuda`
only selects a different prebuilt `ort-sys` distribution and adds the dlopen'd provider `.so`
files; it is purely additive. speakrs agrees: `ExecutionMode::Cpu.validate()` returns `Ok(())`
unconditionally and the CPU EP is registered with no feature gate.

**Serving CPU from the GPU image costs zero extra bytes and zero extra libraries** — do not add
the CPU ORT tarball, its `libonnxruntime.so` would collide with the GPU tarball's. The second
engine costs **+620 MB host RSS and 0 MiB VRAM**, and its output is bit-identical to CUDA's for
centroids (boundaries can differ by one frame).

> **Splitting a model half-way is the dangerous failure mode here.** A partially-split model
> **shares a buffer between threads and returns silently wrong numbers**. It does not crash, and
> a smoke test passes. That is why the shared-session work was not attempted as a tail-end
> change: half-done here looks finished.

`docker/Dockerfile.server-cpu` stays. It is not a correctness carve-out; it is the **arm64 and
minimal-footprint** artifact (no NVIDIA runtime or driver). The CUDA image's base and ORT tarball
are x86-64 only, so there is no arm64 superset to fold it into.

All engines load **serially in `run()` before the server binds**, so a misconfigured
`DIAR_DEVICES` fails at startup rather than on the first request. This used to be a *soundness*
requirement — `DiarEngine::load` called `std::env::set_var("SPEAKRS_FBANK_POOL", …)` and speakrs
read it back inside the same call, and glibc `setenv`/`getenv` is not thread-safe. Since
RESULTS §7.50 the pool size is passed to speakrs through `RuntimeConfig` instead,
`DiarEngine::load` mutates no process-global state, and lazy or concurrent loading is safe to add
whenever the ~620 MB RSS of a resident CPU engine is worth reclaiming.

## Production consumes the binary, not the image

This surprises people, so it is stated plainly. `transcribe-app` does **not** run this repo's
image as its sidecar. It runs the **shared backend image**, which has the `diar-server` binary
copied into it:

```yaml
# docker-compose.diar-native.yml
diar-native:
  image: ${DIAR_NATIVE_IMAGE:-davidamacey/opentranscribe-backend:${OT_IMAGE_TAG:-latest}}
  command: ["diar-server"]
```

```dockerfile
# backend/Dockerfile.prod
FROM davidamacey/diar-native@sha256:<pinned digest> AS diar-native-bin
COPY --from=diar-native-bin /usr/local/bin/diar-server         /usr/local/bin/diar-server
COPY --from=diar-native-bin /usr/local/lib/libonnxruntime*.so* /opt/diar-native/lib/
```

Consequences worth internalising:

- **A release ships to production only when that `@sha256:` digest is repointed.** Publishing a
  new tag here changes nothing on its own, and that change is made in `transcribe-app`, not here.
- The sidecar image name in compose is a **backend** image. There is no `diar-server:latest` in
  the consumer's compose file any more; an earlier unqualified `image: diar-server:latest` was
  removed precisely because Docker resolves it to `docker.io/library/diar-server`, a namespace
  nobody here controls.
- `opentr.sh` exports `DIAR_NATIVE_IMAGE` for local dev, which is the supported override point.
- Consumers that copy the binary out want the **amd64 CUDA** digest — that is the build the
  `diar-native-bin` stage extracts from. Digests are listed in
  [DEPLOYMENT.md](DEPLOYMENT.md#published-images).

### Why the binary works inside the backend image

It is not luck. The backend image already ships every CUDA shared object `diar-server` needs, via
torch's pip-installed NVIDIA wheels under
`.../site-packages/nvidia/{cublas,cufft,cuda_runtime,cudnn,curand}/lib/`. **All six sonames match
exactly**: `libcublas.so.12`, `libcublasLt.so.12`, `libcufft.so.11`, `libcurand.so.10`,
`libcudart.so.12`, `libcudnn.so.9`.

One small addition is still needed: **`libopenblas0`** — speakrs' CPU-side BLAS ops, not covered
by any backend library (~48 MB via apt).

> **Centroids drift in the 5th-6th decimal place across the two builds, and that is fine.** A
> different physical cuBLAS build produces ~1e-5 to 1e-6 differences in cosine similarity, which
> is roughly **100 000× smaller** than the gap between speaker tiers. It is not a correctness
> concern and there is nothing to fix — recorded here because someone diffing embeddings will
> otherwise find it and panic.

The flip procedure into OpenTranscribe is [`INSTALL_NATIVE.md`](INSTALL_NATIVE.md).

### Integration contract

Per OpenTranscribe's `SpeakerDiarizer`, all implemented: segments + exclusive segments +
per-speaker un-normalized 256-d centroids (OpenSearch kNN normalizes) + ad-hoc window embedding
(boundary recheck) + optional per-speaker gender. Still open: min/max/num-speaker constraints
(T9b — forced counts currently warn and auto-count).

## Logging

`diar-server` installs a `tracing-subscriber` and logs to **stdout**, so `docker logs` and
compose capture it with no configuration. Fatal startup errors (the provisioning gate's
remediation block) stay on **stderr**, because they are printed on the way to `exit()` and must
survive any log setting. The `provision-models` / `verify-models` subcommands write
machine-readable JSON to stdout and must not be interleaved with log records — only the serve
path installs a subscriber.

Each `/diarize` and `/embed_window` request runs inside a span carrying `request_id`, `endpoint`,
`device`, the audio **basename** and the `gender` flag, and ends with one record giving
`duration_ms`, `outcome`, and either `num_speakers`/`segments` or an `error_class`. The span is
re-entered on the blocking worker thread, so speakrs' pipeline events are attributed to the
request that caused them — 14 of 15 measured. The exception is
`speakrs::inference::segmentation::run`'s "Segmentation thread profile", emitted from a thread
speakrs spawns internally for the fbank∥GPU pipeline; that thread does not inherit the span, so
the event is logged without a `request_id`. Fixing it needs a `vendor/speakrs` change.

`RUST_LOG=speakrs=debug` surfaces the engine's own stage timings (fbank, GPU predict,
clustering). This works in **both** `diar-server` and `diar-cli`; before it landed the server
installed no subscriber at all, so every speakrs event was silently discarded regardless of
`RUST_LOG`.

Full media paths, model weights and the HuggingFace provisioning token are never logged;
`provision-models` scrubs the token out of the exporter's stdout *and* stderr and marks its
`--hf-token` argument `hide_env_values`.

The `RUST_LOG` / `DIAR_LOG_FORMAT` knobs, and why the default is not a bare `info`, are in
[CONFIGURATION.md](CONFIGURATION.md#logging).

## Apple Silicon (native, CoreML)

Brought up end-to-end 2026-08-20 on an Apple M2 Max: a `coreml` feature (mirroring `cuda`) with
real GPU-accelerated inference, speakrs' own `compare_coreml.py` parity checks passed, and real
diarization output verified 99%+ match against CUDA on the same file (93 vs 92 segments).

Two things to be clear about:

- **This is not reachable through Docker.** Docker Desktop's Linux VM on macOS has no
  Metal/CoreML access regardless of image architecture. It needs a native macOS binary, compiled
  on the machine. It is **not published**.
- Speed is *not yet* validated under matched quiet-machine conditions. RESULTS §7.31 has the
  honest caveat: CoreML looked competitive with CUDA in an apples-to-oranges quick check, but the
  CUDA side was on a contended machine.

Under `coreml`, `clone_shared` is compiled out (speakrs cfgs its own equivalent out too — CoreML
is not ORT sessions and is single-thread-at-a-time), so `AppState::with_engine` holds the engine
mutex for the whole request and `DIAR_MAX_INFLIGHT` has no effect in that mode (RESULTS §7.31).

Two gaps in speakrs' own conversion tooling were found and fixed along the way (a missing
`export_fbank_30s.py`, pushed to the fork) — full writeup in RESULTS §7.31.

---

See also: [DEPLOYMENT.md](DEPLOYMENT.md) · [API.md](API.md) · [PERFORMANCE.md](PERFORMANCE.md) ·
[DEVELOPMENT.md](DEVELOPMENT.md) · [README](../README.md)
