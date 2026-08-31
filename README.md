# diar-native — Fast, Deployable Speaker Diarization for OpenTranscribe

Native (Rust/ONNX) speaker diarization engine for OpenTranscribe, built on a vendored,
heavily-patched [`speakrs`](https://github.com/avencera/speakrs) (upstream) —
mirrored at [`attevon-llc/speakrs`](https://github.com/attevon-llc/speakrs) (our fork, Apache-2.0
license unchanged) — matching the **pyannote
`speaker-diarization-community-1`** pipeline's accuracy at a fraction of its wall time and
deployment weight, with a clean path to **Triton Inference Server** and **AWS GPU** serving.

> **Status: SHIPPED — `diar-server:0.2.0` runs live in the OpenTranscribe stack (2026-08-20).**
> Accuracy holds the recorded gates exactly (AMI-16 full **13.101%** / exclusive **17.813%**,
> Karpathy **8.219%**, VoxConverse **4.847%** — beats the production fork). Warm engine speed:
> Karpathy 66.5 min diarized in **21.6 s (184× RT)**; ES2004a 36 min in **6.6 s**. Concurrent
> requests share one engine's VRAM (T9a shared sessions — spans no longer double under load),
> fbank runs pipelined against the GPU, and the sidecar ingests original media (mp3/m4a/flac/
> any-rate wav) directly. Upload→transcript on the reference file: 108.4 s (Python) → **54.4 s**.
>
> Read next: [`PLAN.md`](PLAN.md) (roadmap + locked decisions) ·
> [`validation/RESULTS.md`](validation/RESULTS.md) (every measurement, append-only — §7.25-7.30
> are the latest) · [`docs/TEST_CORPORA_AND_BASELINES.md`](docs/TEST_CORPORA_AND_BASELINES.md)
> (every number to beat) · [`docs/UPSTREAM_PRS.md`](docs/UPSTREAM_PRS.md) +
> [`docs/pr_drafts.md`](docs/pr_drafts.md) (speakrs contribution queue — branches prepared in
> `upstream-work/`, submission pending approval).

---

## 1. Why this exists

OpenTranscribe (`transcribe-app`) diarizes with a **pinned fork** of pyannote.audio
(`davidamacey/pyannote-audio` @ `gpu-optimizations` / `a3f38afb`) that already carries heavy GPU
optimization work (pinned memory, CUDA stream prefetch, VRAM-budgeted batching, −66% VRAM).
It works well, but the Python stack has structural ceilings:

- **GIL + single process**: clustering (VBx, scipy/numpy) is ~41% of wall time on a 4.7 h file and
  serializes with everything else; whisper and diarization fight for the same worker.
- **Deployment weight**: ~9 GB image with full torch — painful for AWS instances and for the
  open-source install matrix (CPU-only boxes; Apple Silicon now has a native CoreML path, §6).
- **No serving story**: models are locked inside a Celery worker process; no dynamic batching, no
  cross-job GPU sharing, no independent scaling of the 53%-of-GPU-time embedding stage.

Goals, in order:
1. **Accuracy parity** with the production fork (hard gates — see TESTPLAN).
2. **Faster end-to-end** diarization on the same hardware (beefy home servers: 2× RTX A6000).
3. **Lighter, portable deployment**: single native binary + ONNX Runtime; CUDA / CPU-only /
   CoreML backends for the open-source deployment matrix.
4. **Serving-ready**: the same ONNX artifacts servable by Triton (onnxruntime / tensorrt backends)
   for AWS scale-out; orchestration embeddable as a Triton backend later (Triton's backend API is C —
   Rust implements it directly via `diar-ffi`).

## 2. Why speakrs (and why not the alternatives)

[`speakrs`](https://github.com/avencera/speakrs) (Apache-2.0, Rust) implements the **full
community-1 pipeline**: powerset decode, overlap-add aggregation, binarization,
speaker-conditioned embeddings, **PLDA + VBx clustering (AHC init)** — fixture-tested against
scipy/pyannote at 1e-4. Its export script loads the actual `pyannote/speaker-diarization-community-1`
pipeline (we re-export ourselves from the production HF cache — see §4 of RESULTS).

| Candidate | Verdict | Reason |
|---|---|---|
| **speakrs** (Rust) | **primary candidate** | full community-1 fidelity incl. VBx; ORT CPU/CUDA/MIGraphX + CoreML; active; Apache-2.0 |
| sherpa-onnx (C++) | rejected | simpler clustering, no VBx → documented accuracy drift vs pyannote (k2-fsa/sherpa-onnx#1708) |
| pyannote-rs (Rust) | rejected | 80.2% DER on VoxConverse subset; no aggregation, cosine-threshold clustering |
| custom C++/Rust from scratch | fallback-of-last-resort | months to re-earn accuracy speakrs already demonstrates; zero inference-speed advantage (same ORT kernels) |

**"Wouldn't C++ be faster?"** No: in every candidate the neural nets execute inside ONNX Runtime's
C++ kernels; orchestration math in Rust compiles to the same machine-code class as C++. The
differentiator is *algorithm fidelity* (VBx), not language. Triton compatibility is equal too — its
custom-backend interface is a **C API**, which a Rust `cdylib` implements without any C++ shim.

## 3. Background: how community-1 diarization works (and what can/can't be ONNX)

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

The other historical blockers, all now root-caused and fixed (details + evidence in RESULTS §1/§4):
- *"ORT CUDA has no Sin/Cos kernels"* → SincNet synthesizes conv filters from **frozen** params;
  onnxsim constant-folding removes Sin/Cos/If **bit-exactly** (179 → 40 nodes, 2.0× measured).
- *"TensorRT rebuilds engines per shape"* → the old shape profile declared max 500 fbank frames vs
  the real 998; with correct fixed shapes, batch is the only dynamic axis.
- *"Embedding export fails on torchaudio fbank"* → export the ResNet with fbank outside the graph,
  or fuse a decomposed fbank (framing + DFT/mel matmuls) like speakrs does. (Note: the fused
  variant carries a measured ~6.7× ORT-CUDA cost — see RESULTS §4.2 — so fbank-outside + batched
  native fbank is the perf-correct split.)

## 4. What we're testing (summary — full matrix in [validation/TESTPLAN.md](validation/TESTPLAN.md))

Two engines, identical corpora, identical scoring (`pyannote.metrics` DER, collar 0.25, overlap
included, AMI with official UEMs):

- **Engine A (baseline):** production fork @ `a3f38afb`, run inside `opentranscribe-backend:latest`
  with the fork bind-mounted — the exact production code path.
- **Engine B (candidate):** speakrs @ `b0756b1` (`cuda` mode = 1.0 s step, pyannote-equivalent;
  never `cuda-fast`), our self-exported community-1 ONNX models.
- **Serving spike:** the same ONNX graphs on **Triton** (onnxruntime backend now; tensorrt次)
  to measure the serving stack directly.

Corpora: AMI test-16 (ground truth + UEMs), Karpathy 66-min hand-labeled acceptance clip,
0.5→4.7 h duration curve (RTF/VRAM scaling), VoxConverse dev-216 (speakrs' published corpus),
CPU-only legs for the lite deployment. Gates G1–G5 (accuracy / speed / memory / determinism)
are defined in TESTPLAN §3; results land in RESULTS.md as they complete.

## 5. Repository layout

```
diar-native/
├── README.md                  ← you are here: what/why/how
├── validation/
│   ├── TESTPLAN.md            ← test matrix, gates, methodology, reproduction commands
│   ├── RESULTS.md             ← every measurement (append-only log; never re-run a logged test)
│   ├── run_fork_baseline.py   ← Engine A runner (backend image, fork bind-mount, RTTM+timing out)
│   ├── run_speakrs.sh         ← Engine B runner (diar-bench image, RTTM per file/run)
│   ├── score_der.py           ← DER scorer (UEM-aware, aggregate + per-file, JSON out)
│   ├── ort_cuda_microbench.py ← ORT CUDA EP folded-vs-unfolded-vs-eager microbench
│   └── triton_bench.py        ← Triton gRPC latency/throughput bench
├── docs/
│   ├── TEST_CORPORA_AND_BASELINES.md   ← where the audio/refs live + every number to beat
│   ├── HANDOFF_T9A_SHARED_SESSIONS.md  ← shared sessions so diarization stops serialising
│   ├── HANDOFF_DIARIZATION_SPEED.md    ← the remaining single-job speed levers, ranked
│   ├── VRAM_AND_TIERS.md      ← what holds GPU memory, why, and what fits on 4/8/12 GB
│   ├── INSTALL_NATIVE.md      ← the flip procedure into OpenTranscribe
│   ├── E2E_PIPELINE_MAP.md    ← app pipeline anchors + ranked levers L1-L10
│   └── UPSTREAM_PRS.md        ← the speakrs contribution queue (incl. PR-7 exclusive fix)
├── crates/
│   ├── diar-core/             ← speakrs wrapper: shared-session engine handles (clone_shared),
│   │                            centroids, embed_window, exclusive output, gender, media decode,
│   │                            logging policy shared by both binaries (logging.rs)
│   ├── diar-server/           ← the T1 sidecar: /diarize /embed_window /healthz (axum);
│   │                            per-request engine handles — jobs run concurrently;
│   │                            structured logs to stdout (RUST_LOG, DIAR_LOG_FORMAT)
│   └── diar-cli/              ← bench/ops runner (RTTM+JSON out, engine traces via RUST_LOG
│                                on stderr — stdout stays parseable JSONL)
├── docker/
│   ├── Dockerfile.server      ← production image (diar-server:0.2.0; ORT 1.24.2 GPU libs)
│   └── Dockerfile.bench       ← self-contained speakrs CUDA build (xtask diarize CLI)
├── patches/0001-…patch        ← THE vendored speakrs diff (regenerate after any vendor edit)
├── upstream-work/             ← [gitignored] upstream-tip clone holding the 7 prepared PR
│                                branches (see docs/pr_drafts.md); pushed to attevon-llc/speakrs,
│                                not yet opened as PRs against avencera/speakrs
├── triton/models/             ← Triton spike model repo (config.pbtxt per model)
├── refs/                      ← staged references (AMI test RTTM+UEM, Karpathy fixed RTTM)
├── models*/                   ← [gitignored] self-exported community-1 ONNX + PLDA (gated weights)
│                                models_folded/=fast set (default), models_small/=laptop set
├── vendor/speakrs/            ← upstream clone @ pin b0756b1 + our working-tree patch set
│                                (reproduce: scripts/bootstrap_vendor_speakrs.sh — clones our
│                                fork attevon-llc/speakrs pinned to the master commit matching
│                                live diar-server:0.2.0, verified byte-identical + 94/94 tests)
├── scripts/bootstrap_vendor_speakrs.sh  ← populates vendor/speakrs from our fork at a pinned
│                                commit (idempotent; re-run to refresh)
└── results/                   ← RTTMs, timing JSONL, DER JSONs per run tag
```

Integration contract with OpenTranscribe (per its `SpeakerDiarizer`), all implemented:
segments + exclusive segments + per-speaker un-normalized 256-d centroids (OpenSearch kNN
normalizes) + ad-hoc window embedding (boundary recheck) + optional per-speaker gender.
Still open: min/max/num-speaker constraints (T9b — forced counts currently warn + auto-count).

## 6. Deployment tiers (product decision, 2026-08-19)

| tier | target | design |
|---|---|---|
| **T1 — embedded/sidecar (DEFAULT, open source — SHIPPED as `diar-server:0.2.0`)** | laptops, small computers, single-GPU boxes | speakrs core + our patch set (folded seg, multimask-batching fix, fbank pool, VBx vectorization, exclusive-overlap fix, **shared sessions**, **pipelined fbank∥GPU**): one model set (Arc-shared ORT sessions, weights + arenas loaded once) + per-request scratch handles — concurrent jobs at one engine's VRAM, verified output-identical to serial. Ingests original media directly (symphonia + FFT resample). **CPU and CUDA are served from one image, selected per request** (ORT CPU EP, ≈ eager-torch parity measured) — see §6b. `SPEAKRS_ARENA_SHRINK=1` drops the between-job VRAM floor 4.5 GB → 1.1 GB for small cards (~20% per-job cost). No Triton, minimal RAM. |
| **T2 — Triton (opt-in, larger home servers + AWS)** | multi-user / multi-job servers | tritonserver + TRT engines (fp32), dynamic batching across concurrent jobs (measured 2.14× throughput at 8 clients on one weight copy), `diar-ffi` custom backend or sidecar-calls-Triton topology. Higher RAM/system footprint — that's why it's opt-in, not default. Note: an in-`ort` TensorRT EP for T1 was implemented, measured (1.33-1.48× at +0.03 pp AMI DER) and **rolled back** — the compatibility surface wasn't worth speed that hides behind transcription (RESULTS §7.26 is the recipe if this changes). |

The Python fork path remains the universal fallback behind a config flag in both tiers.

**VRAM budgets per tier, and what actually fits, are measured in
[docs/VRAM_AND_TIERS.md](docs/VRAM_AND_TIERS.md).** The headline for planning: the warm stack
holds a **7 575 MiB floor** on a 12 GB card (sidecar 4 136 + whisper 2 038 + redaction 1 346) and
each concurrent job adds only ~490 MiB, because weights are shared and only activations scale.
The floor is warm-start caching, not a leak — it returns to the same figure after every job.
Note the sidecar's share is ~547 MiB of weights plus ~3.6 GB of ORT arena acquired on first
inference and never released, so **T1's "laptops" claim above holds for 6-8 GB cards but not yet
for 4 GB**: that needs redaction off the GPU, the small model set, and sidecar load/unload —
which trades away the transcribe∥diarize overlap.

**Apple Silicon (native, CoreML):** brought up end-to-end 2026-08-20 on an Apple M2 Max —
`coreml` feature (mirrors `cuda`), real GPU-accelerated inference (not Docker: Docker Desktop's
Linux VM on macOS has no Metal/CoreML access regardless of image arch — this needs a native
macOS binary), speakrs' own `compare_coreml.py` parity checks passed, real diarization output
verified 99%+ match against CUDA on the same file. Speed is *not yet* validated under matched
quiet-machine conditions (RESULTS §7.31 has the honest caveat — CoreML looked competitive with
CUDA in an apples-to-oranges quick check, but the CUDA side was on a contended machine). Two
gaps in speakrs' own conversion tooling found and fixed along the way (missing
`export_fbank_30s.py`, pushed to the fork) — full writeup in RESULTS §7.31.

## 6a. Publishing & consuming the image

`diar-native` stays its own repo — different toolchain (Rust/CUDA) than the rest of
OpenTranscribe (Python/Node), independent release cadence. The connection to `transcribe-app`
is at the image layer only (`docker-compose.diar-native.yml`'s `diar-native` service), not git.

- **Release images** publish to Docker Hub as `davidamacey/diar-native:<version>` (+ `:latest`),
  after a `trivy image --severity HIGH,CRITICAL` scan comes back clean (or documented as
  accepted). Built via `docker build -f docker/Dockerfile.server -t diar-server:<ver> .`, then
  `docker tag`/`docker push` to the registry name.
- **Non-developers / prod:** pull the published tag — no Rust toolchain, no clone of this repo,
  no local build. `transcribe-app`'s compose just needs its `diar-native` service's `image:` to
  point at `davidamacey/diar-native:<version>` instead of the local-only `diar-server:latest`
  (a one-line compose change, made there, not here — see Ground Rules below).
- **Developers modifying diar-native:** clone this repo, build locally
  (`docker build -f docker/Dockerfile.server -t diar-server:latest .` — matches the compose
  file's current default image name/tag exactly, so it's picked up automatically), point a
  local `transcribe-app` deployment at it (already the default — override
  `DIAR_NATIVE_MODELS_DIR` / the image tag via `.env` if diverging from the registry version).
  No push required for local iteration.

## 6b. Execution devices (one image, CPU and CUDA)

The CUDA image is a **superset** of the CPU image on amd64, and always was — we just had no way
to ask for CPU after startup. The ORT CPU execution provider is *statically linked* into the
binary by `ort-sys` (the `onnxruntime_mlas` kernel library is part of the core static objects),
which is why `ldd diar-server` reports no ONNX Runtime `NEEDED` entry at all. `--features cuda`
only selects a different prebuilt `ort-sys` distribution and adds the dlopen'd provider `.so`
files; it is purely additive. speakrs agrees: `ExecutionMode::Cpu.validate()` returns `Ok(())`
unconditionally and the CPU EP is registered with no feature gate. **Serving CPU from the GPU
image costs zero extra bytes and zero extra libraries** — do not add the CPU ORT tarball, its
`libonnxruntime.so` would collide with the GPU tarball's.

`docker/Dockerfile.server-cpu` stays. It is not a correctness carve-out; it is the **arm64 and
minimal-footprint** artifact (189 MB vs 3.46 GB, no NVIDIA runtime or driver). The CUDA image's
base and ORT tarball are x86-64 only, so there is no arm64 superset to fold it into.

### Selecting devices

| knob | scope | meaning |
|---|---|---|
| `DIAR_MODE` | startup | Unchanged. Single device: `cpu`, `coreml`, `coreml_fast`; **unset or unrecognized ⇒ `cuda`** (a long-standing quirk, deliberately preserved). |
| `DIAR_DEVICES` | startup | Comma list, e.g. `cuda,cpu`. **First entry is the default device.** Wins over `DIAR_MODE`; blank is treated as unset. Duplicates collapse, order is preserved. |
| `DIAR_MAX_INFLIGHT` | startup | Unchanged, default 2. The **global** admission gate — bounds TOTAL inflight across all devices, so adding an engine cannot silently double concurrency. |
| `DIAR_MAX_INFLIGHT_CPU` | startup | Optional inner sub-gate for CPU work only. Unset (default) = no inner gate and no behaviour change. CPU requests take the global permit first, this one second — always that order. |
| `"device"` | per request | New optional field on `/diarize` and `/embed_window`. Omitted/null = the default device. |

All engines load **serially in `run()` before the server binds** — never lazily on a request
thread. This is not a missed optimization: `DiarEngine::load` calls `std::env::set_var`
(`SPEAKRS_FBANK_POOL`, read back inside the same call), and glibc `setenv`/`getenv` is not
thread-safe, so a lazy load would race live tokio workers.

Defaults are unchanged end to end: with neither new variable set, the server loads exactly one
engine from `DIAR_MODE`, exactly as before.

```bash
# GPU deployment that can also answer CPU requests
DIAR_DEVICES=cuda,cpu diar-server
curl -sX POST localhost:8701/diarize -d '{"wav_path":"/audio/x.wav","device":"cpu"}'
```

Responses carry `x-diar-device: cuda|cpu` naming the device that actually ran the job. It is a
header, not a body field, because `DiarizeOutput` is the consumer's parsed schema.

### `/healthz` now returns JSON

```json
{"status":"ok","default_device":"cuda","devices":["cuda","cpu"],"supported_devices":["cuda","cpu"]}
```

`devices` = loaded and serving in this process (first is the default). `supported_devices` =
what this **build** can serve — a superset; something listed there but missing from `devices`
needs a `DIAR_DEVICES` change, not a rebuild. Non-breaking for the compose healthcheck, which
is `curl -sf .../healthz` and only inspects the HTTP status.

> **Silent-ignore trap — read this before sending `device`.** Neither request struct uses
> `deny_unknown_fields`, so an **old** diar-server does not reject `{"device":"cpu"}` — it
> *ignores* it and runs the job on CUDA anyway, returning 200. Serde cannot help you here.
> Consumers MUST negotiate on `/healthz` `supported_devices` (or on the presence of the
> `x-diar-device` response header) before relying on the field. An unknown device name on a
> *new* server is a 400 that names the devices the build serves; on an old one it is a silent
> success on the wrong device.

## 6c. Logging

`diar-server` installs a `tracing-subscriber` and logs to **stdout**, so `docker logs` and
compose capture it with no configuration. Fatal startup errors (the provisioning gate's
remediation block) stay on **stderr**, because they are printed on the way to `exit()` and must
survive any log setting.

| knob | scope | meaning |
|---|---|---|
| `RUST_LOG` | startup | Standard `tracing` filter, e.g. `info`, `debug`, `speakrs=debug`, `diar_server=debug,speakrs=trace`. **Unset or empty ⇒ `info,ort::logging=warn`** — the container is useful out of the box. A malformed value logs a warning and falls back to that same default rather than starting the process silent. |
| `DIAR_LOG_FORMAT` | startup | `text` (default) — human-readable lines, ANSI only when stdout is a terminal. `json` — one flattened JSON object per line for log aggregation. An unrecognized value warns and uses `text`. |
| `x-request-id` | per request | Request header, optional. Honoured if present so a job keeps one id end to end through a larger stack; otherwise one is generated. Echoed back on the response, including on errors. Sanitized before it is logged (control characters stripped, 64 chars max) — a caller cannot forge a log record with it. |

`RUST_LOG=speakrs=debug` is what surfaces the engine's own stage timings (fbank, GPU predict,
clustering). This works in **both** `diar-server` and `diar-cli`; before this landed the server
installed no subscriber at all, so every speakrs event was silently discarded regardless of
`RUST_LOG`.

> **Why the default is not a bare `info`.** ONNX Runtime's native log bridge (`ort::logging`)
> emits **5797 INFO lines** on a CUDA startup — "Removing NodeArg …", "GraphTransformer …
> modified: 0" — against 3 lines from diar-server. Measured, not estimated (RESULTS §7.37). A
> blanket `info` buries the startup record 2000:1, so the default holds that one target at
> `warn`. Its warnings are real perf diagnostics (Memcpy nodes, unassigned nodes) and are kept,
> as is `ort::ep`, which reports which execution provider actually registered. An explicit
> `RUST_LOG=ort=info` or `RUST_LOG=debug` still gets the firehose.

Each `/diarize` and `/embed_window` request runs inside a span carrying `request_id`,
`endpoint`, `device`, the audio **basename** and the `gender` flag, and ends with one record
giving `duration_ms`, `outcome`, and either `num_speakers`/`segments` or an `error_class`
(`bad_device`, `admission`, `invalid_input`, `audio_decode`, `inference`, `panic`). The span is
re-entered on the blocking worker thread, so speakrs' pipeline events are attributed to the
request that caused them.

Full media paths, model weights and the HuggingFace provisioning token are never logged;
`provision-models` scrubs the token out of the exporter's stdout *and* stderr and marks its
`--hf-token` argument `hide_env_values`.

```bash
# human-readable, default level
docker run --rm -p 8701:8701 -v /srv/models:/models:ro diar-server:latest

# engine stage timings, JSON for an aggregator
docker run --rm -p 8701:8701 -v /srv/models:/models:ro \
  -e RUST_LOG=speakrs=debug -e DIAR_LOG_FORMAT=json diar-server:latest
```

## 7. Ground rules

- `transcribe-app` and `pyannote-audio-fork` are **read-only** (other agents work there; the fork
  is production-pinned). Their harness scripts/images are *used*, never modified.
- Exported ONNX/PLDA artifacts derive from **gated** `pyannote/speaker-diarization-community-1`
  (CC-BY-4.0 + terms): regenerate locally, never commit, never redistribute.
- Benchmark GPUs: A6000s (GPU 0/2) for engine A/B timing; 3080 Ti (GPU 1) for serving spikes.
- Publishable numbers only from Tier-A corpora (AMI, VoxConverse, Earnings-21).
