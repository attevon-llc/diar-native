# diar-native — Fast, Deployable Speaker Diarization

Speaker diarization — **"who spoke when"** — as a small self-hosted HTTP service. Rust and ONNX
Runtime, no Python at serving time, **195 MB** on a CPU host. It matches the pyannote
`speaker-diarization-community-1` pipeline's accuracy at a fraction of its wall time and
deployment weight.

## Run it

**You do not need this repository to run diar-native.** The deployment is two files — a compose
file and a `.env` — plus images published on Docker Hub. Nothing is cloned, built or compiled.

One thing cannot be shipped for you: the models are derivatives of the **gated**
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
weights, which nobody may redistribute. So you need a free **HuggingFace read token**, once, and
the first startup exports your own copy of the models locally (~484 MB, ~3 minutes). See
[The token](#the-token) — it is two clicks and there is no waiting list.

### Linux/amd64, CPU only

The default. Works on any amd64 machine, GPU or not, and needs no NVIDIA runtime.

```bash
mkdir -p diar-native/audio && cd diar-native
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/docker-compose.prod.yml -o docker-compose.yml
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/.env.example -o .env
$EDITOR .env      # set HUGGINGFACE_TOKEN=hf_...   (the only thing you must supply)
docker compose up
```

That single `up` exports the models and then serves on port 8701. **Every later `docker compose
up` skips the export** — it finds the provenance marker and starts serving in seconds, with no
token and no network access at all.

### Linux/amd64 with an NVIDIA GPU

The CUDA image (3.04 GB) serves **both `cuda` and `cpu`**, chosen per request. It needs the
NVIDIA container toolkit installed and registered with Docker — a working `nvidia-smi` on the
host is not sufficient by itself.

```bash
mkdir -p diar-native/audio && cd diar-native
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/docker-compose.prod.yml -o docker-compose.yml
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/docker-compose.gpu.yml -o docker-compose.gpu.yml
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/.env.example -o .env
printf 'DIAR_IMAGE=davidamacey/diar-native:0.3.0\n' >> .env
$EDITOR .env      # set HUGGINGFACE_TOKEN=hf_...
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

The GPU is opt-in through that overlay because a compose device reservation cannot be made
conditional: present and unsatisfiable, it is a hard startup failure on every GPU-less host.

### macOS / Apple Silicon, and arm64 Linux

Docker runs the **arm64 CPU image**, and it is worth being blunt about what that is: it uses
your **CPU cores**. It does **not** use the GPU and does **not** use the Neural Engine. Docker
on macOS has no Metal access at any image architecture, so an `arm64` tag buys you the right
instruction set and nothing else. It works correctly; it is simply not accelerated.

```bash
mkdir -p diar-native/audio && cd diar-native
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/docker-compose.prod.yml -o docker-compose.yml
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/.env.example -o .env
printf 'DIAR_IMAGE=davidamacey/diar-native:0.3.0-cpu-arm64\nDIAR_PROVISION_IMAGE=davidamacey/diar-native:0.3.0-provision-arm64\n' >> .env
$EDITOR .env      # set HUGGINGFACE_TOKEN=hf_...
docker compose up
```

**Do not use `:latest` or `:0.3.0` on an arm64 host.** Those two are the CUDA image and are
published for **linux/amd64 only** — permanently, because no aarch64 ONNX Runtime GPU build
exists. Docker Desktop will emulate them rather than refuse them, so the symptom is
"inexplicably slow" rather than an error you can act on. The `-arm64` tags above name the
architecture, which is why this quickstart uses them; see [Published
images](#published-images) for the full tag model.

A native **CoreML** build does exist that uses the Apple GPU, and it works — verified on an
M2 Max at 93 vs 92 segments against the CUDA reference
([`validation/RESULTS.md`](validation/RESULTS.md) §7.31). But it is **not published**, and
Docker cannot reach Metal, so it has to be compiled from source on the machine itself. See §6.

### The token

Two free, instant, auto-approved steps — no waiting list, no human review, and the pipeline
itself is CC-BY-4.0:

1. Create a **read** token at <https://huggingface.co/settings/tokens>.
2. **Signed in as that same account**, accept the terms at
   <https://huggingface.co/pyannote/speaker-diarization-community-1>.

Step 2 catches people out: a perfectly valid token whose account never accepted the gate fails
with HTTP 403, and accepting while signed in as a *different* account fails identically. Only
~32 MB is downloaded; the rest is converted on your machine. Serving never touches the network.

### Diarize something

```bash
curl -s localhost:8701/readyz                     # 200 only when models are loaded + verified

cp ~/meeting.wav ./audio/
curl -s -X POST localhost:8701/diarize \
  -H 'content-type: application/json' \
  -d '{"wav_path": "/audio/meeting.wav", "gender": true}'
```

`/diarize` takes a **path inside the container**, not an upload — which is why the audio
directory is bind-mounted at `/audio`. Despite the field name, wav/flac/mp3/m4a/ogg all decode.

## Use it

Four routes, documented in full in §6b-§6e:

| route | what it is |
|---|---|
| `POST /diarize` | the whole job: `segments[]` (may overlap), `exclusive_segments[]` (overlaps resolved — **use these for transcripts**), `num_speakers`, `rttm`, and `speaker_gender` when you send `"gender": true` |
| `POST /embed_window` | a speaker embedding for one time window |
| `GET /healthz` | **always 200 while the process is serving**, in every model state. Carries `models_state`/`models_reason`, plus `devices` and `supported_devices`. Liveness — gate container health on this |
| `GET /readyz` | 200 **only** once the models are verified. Readiness — gate your rollout on this |

Speaker labels (`SPEAKER_00`, …) are arbitrary and stable only within one response; the same
person in two files will not get the same label. Every environment variable the binary reads is
tabulated in [§6e](#6e-environment-variables--the-authoritative-list).

## Develop it

Only needed if you are **changing** diar-native. Everything above runs without a checkout.

```bash
git clone https://github.com/attevon-llc/diar-native && cd diar-native
./start.sh --build          # compile from this checkout, provision, serve, wait on /readyz
./start.sh --cli meeting.wav   # one-shot diarization, no server (always builds — diar-cli is
                               # the one binary that is not published)
```

Without `--build`, `./start.sh` pulls the published image matching your architecture and GPU —
the same thing the compose file does, with platform detection and a token prompt on top. Build
rules, traps and the benchmark protocol are in [`CONTRIBUTING.md`](CONTRIBUTING.md),
[`CLAUDE.md`](CLAUDE.md) and [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md). Longer
walkthrough of both paths: [`QUICKSTART.md`](QUICKSTART.md).

---

[![CI](https://github.com/attevon-llc/diar-native/actions/workflows/ci.yml/badge.svg)](https://github.com/attevon-llc/diar-native/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) · Vulnerabilities:
[`SECURITY.md`](SECURITY.md) · Release history: [`CHANGELOG.md`](CHANGELOG.md)

> **Everything below this line is the reference manual** — architecture, benchmarks, the
> deployment tiers and the authoritative environment table. The four sections above are the
> whole of what running diar-native requires.

Native (Rust/ONNX) speaker diarization engine for OpenTranscribe, built on a vendored,
heavily-patched [`speakrs`](https://github.com/avencera/speakrs) (upstream) —
mirrored at [`attevon-llc/speakrs`](https://github.com/attevon-llc/speakrs) (our fork, Apache-2.0
license unchanged) — matching the **pyannote
`speaker-diarization-community-1`** pipeline's accuracy at a fraction of its wall time and
deployment weight, with a clean path to **Triton Inference Server** and **AWS GPU** serving.

> **Status: 0.3.0 (2026-09-01).** `0.2.0` is what runs live in the OpenTranscribe stack today —
> the live sidecar picks up a release only when `transcribe-app` repoints its pinned
> `davidamacey/diar-native@sha256:…` digest (§6a), because production consumes the **binary**,
> not this image.
>
> Accuracy holds the recorded gates exactly (AMI-16 full **13.101%** / exclusive **17.813%**,
> Karpathy **8.219%**, VoxConverse **4.847%** — beats the production fork). Warm engine speed:
> Karpathy 66.5 min diarized in **21.6 s (184× RT)**; ES2004a 36 min in **6.6 s**. Concurrent
> requests share one engine's VRAM (T9a shared sessions — spans no longer double under load),
> fbank runs pipelined against the GPU, and the sidecar ingests original media (mp3/m4a/flac/
> any-rate wav) directly. Upload→transcript on the reference file: 108.4 s (Python) → **54.4 s**.
>
> **New in 0.3.0:** you can now get the models — `diar-server provision-models` exports them from
> your own Hugging Face token (§6d). One image serves **both `cuda` and `cpu`**, chosen per
> request (§6b). The sidecar has structured logging and a `/readyz` endpoint. Full list:
> [`CHANGELOG.md`](CHANGELOG.md).
>
> Read next: [`PLAN.md`](PLAN.md) (roadmap + locked decisions) ·
> [`validation/RESULTS.md`](validation/RESULTS.md) (every measurement, append-only — §7.34-7.40
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
├── CHANGELOG.md               ← release history (Keep a Changelog)
├── CONTRIBUTING.md            ← how to build/test/PR; reproduces the CI environment locally
├── SECURITY.md                ← vulnerability reporting + the gated-weights policy
├── MODELS_SETS.md             ← fast vs small: what differs and why (matches files.rs)
├── .github/                   ← CI, release, dependabot, PR template, setup-build-env action
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
│   ├── ORT_FUSION_FP16_AARCH64.md ← why fp16 gender won't load on linux/arm64 (issue #14)
│   ├── E2E_PIPELINE_MAP.md    ← app pipeline anchors + ranked levers L1-L10
│   └── UPSTREAM_PRS.md        ← the speakrs contribution queue (incl. PR-7 exclusive fix)
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
│   ├── Dockerfile.server      ← production CUDA image (3.46 GB; ORT 1.24.2 GPU libs)
│   ├── Dockerfile.server-cpu  ← multi-arch CPU-only image (linux/amd64 + linux/arm64, 194 MB)
│   ├── Dockerfile.provision   ← pinned CPU-only torch env, for provisioning off the plain image
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
| **T1 — embedded/sidecar (DEFAULT, open source — SHIPPED)** | laptops, small computers, single-GPU boxes | speakrs core + our patch set (folded seg, multimask-batching fix, fbank pool, VBx vectorization, exclusive-overlap fix, **shared sessions**, **pipelined fbank∥GPU**): one model set (Arc-shared ORT sessions, weights + arenas loaded once) + per-request scratch handles — concurrent jobs at one engine's VRAM, verified output-identical to serial. Ingests original media directly (symphonia + FFT resample). **ONE image serves both `cuda` and `cpu`, selected per request** — the ORT CPU EP is statically linked into every build, so the GPU image is a strict superset of the CPU image at **no extra bytes** (3.46 GB either way); the second engine costs +620 MB host RSS and **0 MiB VRAM**, and its output is bit-identical to CUDA's. See §6b. `SPEAKRS_ARENA_SHRINK=1` drops the between-job VRAM floor 4.5 GB → 1.1 GB for small cards (~20% per-job cost). No Triton, minimal RAM. |
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

### Production consumes the BINARY, not this image

This is the part that surprises people, so it is stated first. `transcribe-app` does **not** run
this repo's image as the sidecar. It runs the **shared backend image**, which has the
`diar-server` binary copied into it:

```
docker-compose.diar-native.yml:
  diar-native:
    image: ${DIAR_NATIVE_IMAGE:-davidamacey/opentranscribe-backend:${OT_IMAGE_TAG:-latest}}
    command: ["diar-server"]

backend/Dockerfile.prod:
  FROM davidamacey/diar-native@sha256:<pinned digest> AS diar-native-bin
  COPY --from=diar-native-bin /usr/local/bin/diar-server            /usr/local/bin/diar-server
  COPY --from=diar-native-bin /usr/local/lib/libonnxruntime*.so*    /opt/diar-native/lib/
```

Consequences worth internalising:

- **A release ships to production only when that `@sha256:` digest is repointed.** Publishing a
  new tag here changes nothing on its own. That change is made in `transcribe-app`, not here.
- The sidecar image name in compose is a **backend** image. There is no `diar-server:latest` in
  the consumer's compose file any more; an earlier unqualified `image: diar-server:latest` was
  removed precisely because Docker resolves it to `docker.io/library/diar-server`, a namespace
  nobody here controls.
- `opentr.sh` exports `DIAR_NATIVE_IMAGE` for local dev, which is the supported override point.

### Published images

Tag shape follows **capability**, not convenience — a manifest list is only honest when every
architecture under it is the same thing (issue #20):

- **`:latest` and `:<ver>` are `linux/amd64` only, and always will be.** They are the CUDA +
  CPU superset, and there is no aarch64 ONNX Runtime GPU build in existence (issue #4, closed
  as a documented impossibility). The only arm64 entry those tags could carry is the CPU
  image — one tag whose capabilities differ by architecture, so that `docker pull` hands an
  arm64 user a diarizer with no GPU support and an amd64 user one with it. Being explicit is
  better, so they stay amd64 and this section says so.
- **`:<ver>-cpu` and `:<ver>-provision` are multi-arch manifest lists** (amd64 + arm64). Both
  architectures are the same build with the same capabilities there, so `docker pull`
  resolving them for you is the truth rather than a convenient lie.
- **`:<ver>-cpu-arm64` and `:<ver>-provision-arm64` remain published** as single-platform
  arm64 aliases. `start.sh` and `docker-compose.prod.yml` name an exact tag per host and then
  assert the architecture of what they pulled; that path wants a tag that is arm64 *by name*,
  so a mismatch is a clean error rather than a slow emulated one. The manifest list is for
  `docker pull` doing the right thing on its own; the alias is for naming it deliberately.

**0.3.0 predates this.** Its images were all built and pushed by hand, so every 0.3.0 tag is
single-platform: `:0.3.0-cpu` is amd64 and arm64 lives only at `:0.3.0-cpu-arm64`. The manifest
lists start at the first release published by `scripts/release.sh`. The table below
is 0.3.0 as actually published, verified by fresh pull.

| tag | platform | size | contents |
|---|---|---|---|
| `davidamacey/diar-native:0.3.0`<br>`davidamacey/diar-native:latest` | linux/amd64 | 3.04 GB | CUDA **and** CPU, selected per request (§6b) |
| `davidamacey/diar-native:0.3.0-cpu` | linux/amd64 | 195 MB | CPU only — no CUDA libraries, no NVIDIA runtime needed |
| `davidamacey/diar-native:0.3.0-cpu-arm64` | linux/arm64 | 223 MB | CPU only. Runs on ARM Linux and on Apple Silicon under Docker — **on CPU cores**, not the GPU or Neural Engine (Docker on macOS has no Metal access; that needs a native `coreml` build, which is not published) |
| `davidamacey/diar-native:0.3.0-provision`<br>`davidamacey/diar-native:0.3.0-provision-arm64` | linux/amd64<br>linux/arm64 | ~1.9 GB | **Provisioning only.** The serving image plus a pinned CPU-only torch + pyannote.audio environment. Referenced by `docker-compose.prod.yml`; delete it once the models exist |

None of the serving images contain Python — that is why the CPU one is 195 MB rather than
~2 GB — so `provision-models` run against one exits **6** with `No python interpreter at
'python3'`. That is what the separate provisioning tag is for, and why it is CPU-based even for
GPU deployments: the export does `pipeline.to(torch.device("cpu"))` and never touches an
accelerator. It builds with no compiler at all, as a pip layer on the image you already have:

```bash
docker build -f docker/Dockerfile.provision \
  --build-arg BASE=davidamacey/diar-native:0.3.0-cpu \
  -t davidamacey/diar-native:0.3.0-provision .
```

Digests, for pinning:

```
0.3.0 / latest    sha256:f1f9773edc34a6116cb761166ed0811665b3b82b9b475578ac21dc0dbc8a1584
0.3.0-cpu         sha256:1ffd2700b7b9d4c0c3980058f7767caf629129134d42e77e0ddfcf7c5caee1a9
0.3.0-cpu-arm64   sha256:41b328a58fe982ae65e1642fc9b461e766058d3b1e8b02607aff053867697268
```

All three: **trivy 0.67.1, 0 HIGH / 0 CRITICAL**, running as **uid 10001**. Consumers that copy
the binary out of the image (see §6a) want the **amd64 CUDA** digest — that is the build the
`diar-native-bin` stage extracts from.

### Publishing

Releases are built and pushed **locally**, by `scripts/release.sh`, not by CI.

```
./scripts/release.sh 0.3.1                 # build + scan, push nothing
./scripts/release.sh 0.3.1 --push          # ... and publish
./scripts/release.sh 0.3.1 --push --latest # ... and move :latest to this version
```

**Why not GitHub Actions.** Hosted runners give ~14 GB of disk; the CUDA image alone is ~3 GB
before build artifacts, and a release is five images across two architectures. A publish
workflow existed and was removed — the one time it ran, on the `v0.3.0` tag push, it failed at
its credentials gate before it could discover the disk problem, and every 0.3.0 image was built
and pushed by hand anyway.

The script does what that workflow tried to, plus the checks a workflow could not:

- Refuses a **dirty tree**, so the images match a commit someone can check out.
- Refuses if **`vendor/speakrs` has drifted from `patches/0001-*.patch`**. This is the check
  that matters most and the one a developer cannot perform by eye: the patch set lives as an
  uncommitted diff in the working tree, so the machine that makes a vendored change is the one
  machine that cannot see a broken bootstrap. `main` shipped unbuildable exactly this way.
- Asserts each image's **architecture matches its tag name**, that the **binary reports the
  version being released** (catching an un-bumped crate version), and that the image **does not
  run as root**.
- Fails the release on any **HIGH/CRITICAL** trivy finding, across all five images.
- Verifies the published digests **against the registry**, not the local daemon — a local tag
  proves nothing about what a stranger will pull.

Publishing takes an explicit `--push`: a tag someone has already pulled cannot be un-pulled.

- **Non-developers / prod:** pull the published tag — no Rust toolchain, no clone of this repo,
  no local build. The whole deployment is `docker-compose.prod.yml` + `.env`; see "Run it" at
  the top of this file.
- **Developers modifying diar-native:** clone this repo and build locally
  (`docker build -f docker/Dockerfile.server -t diar-server:dev .`). Point a local
  `transcribe-app` deployment at it by setting `DIAR_NATIVE_IMAGE=diar-server:dev` in that
  repo's `.env` — but note that this only exercises the sidecar. To test the binary the way
  production actually consumes it you must also rebuild the backend image against your local
  diar-native image. No push required for local iteration.

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
minimal-footprint** artifact (194 MB vs 3.46 GB, no NVIDIA runtime or driver). The CUDA image's
base and ORT tarball are x86-64 only, so there is no arm64 superset to fold it into.

### Selecting devices

| knob | scope | meaning |
|---|---|---|
| `DIAR_MODE` | startup | Unchanged. Single device: `cpu`, `coreml`, `coreml_fast`; **unset or unrecognized ⇒ `cuda`** (a long-standing quirk, deliberately preserved). |
| `DIAR_DEVICES` | startup | Comma list, e.g. `cuda,cpu`. **First entry is the default device.** Wins over `DIAR_MODE`; blank is treated as unset. Duplicates collapse, order is preserved. |
| `DIAR_MAX_INFLIGHT` | startup | Unchanged, default 2. The **global** admission gate — bounds TOTAL inflight across all devices, so adding an engine cannot silently double concurrency. |
| `DIAR_MAX_INFLIGHT_CPU` | startup | Optional inner sub-gate for CPU work only. Unset (default) = no inner gate and no behaviour change. CPU requests take the global permit first, this one second — always that order. |
| `"device"` | per request | New optional field on `/diarize` and `/embed_window`. Omitted/null = the default device. |

All engines load **serially in `run()` before the server binds**, so a misconfigured
`DIAR_DEVICES` fails startup rather than the first request. This used to be a *soundness*
requirement — `DiarEngine::load` called `std::env::set_var("SPEAKRS_FBANK_POOL", ..)` and speakrs
read it back inside the same call, and glibc `setenv`/`getenv` is not thread-safe. Since §7.50
the pool size is passed to speakrs through `RuntimeConfig` instead, `DiarEngine::load` mutates no
process-global state, and lazy or concurrent loading is safe to add whenever the ~620 MB RSS of a
resident CPU engine is worth reclaiming.

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

Was the bare string `ok`. Now:

```json
{
  "status": "ok",
  "default_device": "cuda",
  "devices": ["cuda", "cpu"],
  "supported_devices": ["cuda", "cpu"],

  "models_verified": true,
  "models_state": "verified",
  "models_dir": "/models",
  "models_set": "fast",
  "models_exporter_version": 2,
  "models_pipeline_revision": "a1b2c3d…",
  "models_smoke_at": "2026-09-01T04:15:22Z",
  "models_gender": true,
  "models_reason": null
}
```

`devices` = loaded and serving in this process (first is the default). `supported_devices` =
what this **build** can serve — a superset; something listed there but missing from `devices`
needs a `DIAR_DEVICES` change, not a rebuild.

`models_state` is one of `verified | stale | unverified | failed`, and `models_reason` carries a
human sentence plus the remediation command for every non-verified state. `models_gender`
reports whether the gender classifier **file is present** — gender is enabled by file presence,
so a `--skip-gender` deployment answers `diarize(gender=true)` with 200 and no genders, and this
field is the difference between that being a decision and a mystery. (It does not report the
model's *precision*; that is `toolchain.gender_precision` in the marker.) The fields are flat
rather than nested so that appending more stays additive.

> **`/healthz` returns 200 in every state — this is a guarantee, not an accident.** The compose
> healthcheck is `curl -sf .../healthz` and `diarizer_native.py` checks `resp.status == 200`.
> Every models directory deployed today has no marker, so a 503 for "unverified" would fail
> every existing healthcheck on the day it shipped, fail `up --wait` for the whole stack, and
> silently fall OpenTranscribe back to in-process PyAnnote — the exact quality regression this
> work exists to prevent. Changing the **body** is safe; changing the **code** is not. Use
> `/readyz` when you want a readiness signal that is allowed to fail.

### `/readyz`

Same body, but **200 only when `models_state == "verified"`** and 503 otherwise — so `stale` and
`unverified` both return 503 while the server serves requests normally. This is where "still
provisioning" is distinguished from "broken", with zero blast radius on existing callers. After
provisioning once, move the compose healthcheck here.

### Response headers

| header | on | meaning |
|---|---|---|
| `x-diar-device` | `/diarize`, `/embed_window` success | The device that actually ran the job (`cuda`\|`cpu`\|…). A header rather than a body field because `DiarizeOutput` is the consumer's parsed schema. Its **presence** is also the cheapest capability probe for the multi-device feature. |
| `x-request-id` | `/diarize`, `/embed_window`, success **and** error | The id this request was logged under. Echoed from the inbound `x-request-id` if the caller sent one (sanitized), otherwise generated. Present on 4xx/5xx too, so a caller looking at a failure can find the matching server-side record without guessing. |

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
request that caused them — 14 of 15 measured. The exception is
`speakrs::inference::segmentation::run`'s "Segmentation thread profile", emitted from a thread
speakrs spawns internally for the fbank∥GPU pipeline; that thread does not inherit the span, so
the event is logged without a `request_id`. Fixing it needs a `vendor/speakrs` change.

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

## 6d. Getting the models (`provision-models`)

The models are **not distributed and cannot be**: they are derivatives of the gated
`pyannote/speaker-diarization-community-1` weights. There is no `.onnx` on Hugging Face for any
of them — upstream ships `pytorch_model.bin` plus `plda/*.npz` — so conversion is mandatory, not
an optimisation. Each operator exports locally with their own token; nothing is redistributed.

```bash
diar-server check-token                                    # ~200 ms, no download
export HF_TOKEN=<your huggingface read token>
diar-server provision-models --models-dir /models --set fast
diar-server verify-models   --models-dir /models           # deep: full sha256 + smoke test
```

Expect roughly **484 MB** written for the `fast` set with gender, in a couple of minutes (the
acceptance run measured 119.5 s). Only ~32 MB is downloaded; the rest is produced locally by the
export. The gender classifier is 189.5 MB of the output (~40%) — `--skip-gender` omits it, at the
cost of speaker gender detection.

Provisioning needs a python interpreter with torch and pyannote.audio, which `diar-server`
deliberately does not bundle; it shells out to `DIAR_EXPORT_PYTHON` (default `python3`). Use
`docker/Dockerfile.provision` if the host has no such environment. Full procedure, prerequisites,
the CPython 3.13 caveat and what the smoke test checks:
[`docs/INSTALL_NATIVE.md`](docs/INSTALL_NATIVE.md).

**What the provenance marker claims.** `provision-models` writes `diar-provision.json` recording
the export-recipe version, upstream pipeline revision, toolchain versions and every file's size
and sha256. Startup checks it `stat`-only — the marker parses, the recipe version is current, the
smoke test passed, every recorded file is present at its recorded length. There is deliberately
**no hashing at startup**: re-reading ~484 MB on every boot is unacceptable. So startup answers
*"is this the directory that passed?"*, not *"is this directory still byte-perfect?"* — the
latter is `verify-models`. Claiming more would itself be a fail-open.

### Exit codes

Authoritative source: `crates/diar-core/src/provision/mod.rs::exit`. Stable — scripts and
supervisors branch on these.

| code | name | meaning | emitted by |
|---|---|---|---|
| 0 | `OK` | Success, including a no-op on an already-valid directory. | all subcommands |
| 1 | *(none)* | Serve path only: any other startup failure (bind failed, engine load failed) surfacing as a non-zero `main`. | serve |
| 2 | `USAGE` | Bad arguments — unknown `--set`/`--mode`, unresolvable `--smoke-clip`. | all |
| 3 | `SMOKE_FAILED` | Files exist but the smoke test rejected them; in `verify-models` also means recorded-hash **drift**. | provision-models, verify-models |
| 4 | `EXPORT_FAILED` | The export subprocess failed. | provision-models |
| 5 | `TOKEN_DENIED` | Token missing/invalid, or the repo terms have not been accepted. | provision-models, check-token |
| 6 | `NO_EXPORTER_ENV` | No usable python export environment — interpreter missing, or it cannot import torch / pyannote.audio / onnx. The fix is `pip install`. | provision-models |
| 7 | `NOT_WRITABLE` | The models directory is not writable. Checked up front, before a multi-hundred-MB export. | provision-models |
| 8 | `MODELS_UNUSABLE` | **Serve only:** the models directory is too broken to start against. | serve |
| 9 | `DEVICE_UNAVAILABLE` | The requested execution device is not usable here. Says nothing about the models, and — unlike a smoke failure — never marks them known-bad. | provision-models, verify-models |
| 10 | `UNVERIFIABLE` | **`verify-models` only:** the files work, but there is no marker to verify them *against*, so nothing was compared to a recorded hash. Not `OK`, not `SMOKE_FAILED`. | verify-models |

> **6 and 8 were one code before 0.3.0.** A supervisor could not tell "install torch into the
> exporter" from "provision the models", which have nothing to do with each other — serving needs
> no python at all.

## 6e. Environment variables — the authoritative list

Every variable below has a read site in the code; anything not listed here is not read by
anything. `DIAR_NATIVE_*` names that appear in OpenTranscribe's compose file are **not** in this
table on purpose — they are compose-level indirection that expands into these
(`DIAR_MODE=${DIAR_NATIVE_MODE:-cuda}`), and no Rust code reads them.

### Serving (`diar-server` with no subcommand)

| var | what it does | default | notes |
|---|---|---|---|
| `DIAR_MODELS_DIR` | Directory holding the model set. | `/models` | Also read by `provision-models`/`verify-models` as the `--models-dir` default. |
| `DIAR_BIND` | Listen address. | `0.0.0.0:8701` | Not validated at parse time; a bad value fails at `bind()` and the process exits non-zero. |
| `DIAR_MAX_INFLIGHT` | Global admission gate — bounds **total** inflight across all devices, so adding an engine cannot silently double concurrency. | `2` | Unparseable → falls back to 2. **`0` is accepted and deadlocks every request** — see Known sharp edges. |
| `DIAR_MAX_INFLIGHT_CPU` | Optional inner sub-gate for CPU work only. CPU requests take the global permit first, this one second — always that order. | unset (no inner gate) | Unparseable **or `0`** → treated as unset. |
| `DIAR_DEVICES` | Comma list of devices to load, e.g. `cuda,cpu`. **First entry is the default device.** Duplicates collapse, order preserved. Wins over `DIAR_MODE`. | unset → `DIAR_MODE` | Blank/whitespace-only is treated as unset (`${FOO:-}` in compose must not be fatal). An unknown or not-compiled-in name is **fatal at startup**. |
| `DIAR_MODE` | Legacy single-device knob, used when `DIAR_DEVICES` is absent. | `cuda` | Matches `cpu`, `coreml`, `coreml_fast` exactly; **unset *or unrecognized* falls through to `cuda`** — a long-standing quirk, deliberately preserved. The result is still capability-checked. |
| `DIAR_MODEL_SET` | Assert which tier the startup gate should require (`fast`\|`small`). | unset → read from the directory's own marker, else `fast` | An unparseable value is silently treated as unset. |
| `DIAR_ALLOW_UNVERIFIED_MODELS` | Downgrade the startup gate's fatal cases to warnings. | off | Accepts exactly `1`, `true`, `TRUE`, `yes`. Note `True` and `YES` do **not** work. |
| `DIAR_GENDER_MAX_SECONDS` | Cap (seconds, taken from the middle of the window) on the clip fed to the wav2vec2 gender classifier. Unbounded turns cost ~6.3 GB VRAM. | `5` | Unparseable or `0` → falls back to 5. |
| `RUST_LOG` | `tracing` filter. | `info,ort::logging=warn` | **Unset does not mean silent.** Empty is treated as unset; a malformed value warns and falls back to the default rather than starting the process blind. See §6c. |
| `DIAR_LOG_FORMAT` | `text` (human) or `json` (one flattened object per line). | `text` | Unrecognized values warn and use `text`. |
| `RUST_MIN_STACK` | Only inspected for presence; the binary sets it to `16777216` when unset, because speakrs pipeline and ORT worker threads overflow the 2 MiB default. | effectively 16 MiB | An operator-supplied value is left untouched. The tokio runtime separately hardcodes a 16 MiB stack, so this affects non-tokio threads. |

### Provisioning (`provision-models`, `verify-models`, `check-token`)

| var | what it does | default | notes |
|---|---|---|---|
| `HF_TOKEN` | Hugging Face read token. | none | Also `HUGGINGFACE_TOKEN` and `HUGGING_FACE_HUB_TOKEN`, tried in that order. `--hf-token` wins over all three. Empty values are skipped, not treated as a token. |
| `HF_ENDPOINT` | Base URL for the Hugging Face API. | `https://huggingface.co` | **The only knob that makes provisioning work against a mirror or an air-gapped proxy.** Trailing `/` is stripped. Empty is treated as unset **by the Rust side only** — see below. |
| `HF_HOME` | Hugging Face cache directory. Forwarded to the python export child. | none (child uses its own default) | `--hf-cache` overrides. |
| `HF_HUB_OFFLINE` | Read by `huggingface_hub` **inside the python exporter**, not by Rust — set it to re-export from a warm cache with no network (the §7.36 acceptance run did exactly this, and needed no token). | unset | **Not forwarded to the child.** `diar-core` explicitly sets only `PYTHONUNBUFFERED`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`, `HF_TOKEN` and `HF_HOME` on the export subprocess, so this only works if it is already in `diar-server`'s own environment and inherited. |
| `DIAR_EXPORT_PYTHON` | Interpreter (with torch + pyannote.audio) used to run the export scripts. | `python3` | `--python` overrides. A non-working interpreter exits 6. |
| `DIAR_MODE` / `DIAR_DEVICES` | Device for the end-to-end smoke stage. | **`cpu`** | Deliberately *not* the serving default. Provisioning defaulting to a GPU is what used to brick GPU-less hosts. An unrecognized name is exit 2 here, never a silent fall-through to `cuda`. |
| `DIAR_ORT_OPT_LEVEL` | Override the ORT graph optimization level for sessions built through `ort_compat` — the gender session and the smoke test, **not** speakrs' 15 diarization graphs. | unset | `disable`\|`none`\|`0`, `basic`\|`1`, `extended`\|`2`, `all`\|`3`. Escape hatch for a platform hitting the aarch64-class problem in §6f; setting it **bypasses** the automatic aarch64 cap. |
| `DIAR_ORT_DISABLED_OPTIMIZERS` | Pass a disable-list straight to ORT. Same scope as above, and it **bypasses** the automatic aarch64 cap. | unset | Three traps, all measured (§7.40): the pass you probably want is `GeluFusionL2` (`GeluFusion` and `GeluFusionL1` do nothing); the separator is **`;`**, not `,`, despite the `ort` crate's doc comment; and **a misspelled name is silently ignored** — no error, no warning, no effect. |

### Engine tuning (read by speakrs)

| var | what it does | default |
|---|---|---|
| `SPEAKRS_LAZY_SESSIONS` | Skip building the heavy batch-64 primary and batched split-tail sessions the CUDA multimask pipeline never runs; each idle session pins its own ORT arena. Live compose sets `1`. | off (all sessions built) |
| `SPEAKRS_ARENA_SHRINK` | Shrink the device arena back to its initial chunk after each big batched run — a VRAM floor for 4 GB-tier cards, at roughly a 20 % per-job cost. | off |
| `SPEAKRS_INTRA_THREADS` | Intra-op threads for the embedding sessions. | `min(cores, 6)` |
| `SPEAKRS_FBANK_THREADS` | Intra-op threads for the fbank session specifically. | `min(cores, 4)` |
| `SPEAKRS_AHC_THREADS` | Workers for the blocked pairwise-distance computation in AHC clustering. Higher oversubscribes, since each worker also drives a multi-threaded BLAS `dot`. | `min(cores, 8)` |
| `SPEAKRS_FBANK_POOL` | Size of the CPU fbank session pool fanned out per chunk. Read **once**, in `EngineConfig::new`, and passed to speakrs as a `RuntimeConfig` field — `diar-server` no longer overwrites it (§7.50, issue #3). `0` disables the pool and keeps the single fbank session; a malformed value warns and falls back to the default. | `1` on CPU/CoreML (the pool contends with inference for cores — §4.12), `min(cores/4, 8)` on CUDA |

### Not settable, and dead

| var | status |
|---|---|
| `SPEAKRS_TRT`, `SPEAKRS_TRT_CACHE` | **Dead.** No read sites; they do nothing. Left documented only so nobody rediscovers them in the TensorRT-era notes and assumes they work (RESULTS §7.26). |
| `ORT_DYLIB_PATH` | Not applicable to these builds — it is only read under `ort`'s `load-dynamic` feature, which is not enabled. ORT is statically linked. |

### Known sharp edges

- **`DIAR_MAX_INFLIGHT=0` deadlocks every request.** The global gate has no `> 0` guard, unlike
  `DIAR_MAX_INFLIGHT_CPU`, which explicitly treats `0` as unset for exactly this reason.
- **`DIAR_ORT_OPT_LEVEL` typos are silent.** An unrecognized level is ignored with no diagnostic
  and control falls through, whereas a bad `DIAR_ORT_DISABLED_OPTIMIZERS` is fatal.
- **`DIAR_ALLOW_UNVERIFIED_MODELS` is case-sensitive** in a way its neighbours are not: `true`
  and `TRUE` work, `True` does not.
- **Setting `HF_ENDPOINT` to the empty string breaks `provision-models`,** even though this
  table says empty is treated as unset. That is true of the Rust side — but the variable is not
  *stripped* from the environment, and the Python export child inherits it. `huggingface_hub`
  does `os.environ.get("HF_ENDPOINT", "https://huggingface.co")`, which returns the **empty
  string** when the key exists and is blank, so the download URL loses its scheme and the export
  dies with `httpx.UnsupportedProtocol: Request URL is missing an 'http://' or 'https://'
  protocol`. It reads like a network fault and is not one. This is easy to hit from compose,
  where `HF_ENDPOINT: ${HF_ENDPOINT:-}` is the natural way to make a variable optional; the
  bundled `docker-compose.yml` defaults it to the literal URL instead. Either leave it unset
  entirely or give it a real value (RESULTS §7.43).

## 6f. Platform note: the fp16 gender model on linux/arm64

The fp16 gender classifier does not load at all on **linux/arm64** without a workaround — which
silently disables speaker gender there while the server still answers 200. It is not a bug in the
model. The graph is plain opset-17 `ai.onnx` with no contrib domain, but it has 20 `Erf` nodes,
and one of ORT's *extended* (level-2) optimizations rewrites that GELU pattern into
`com.microsoft.Gelu`, for which no fp16 kernel exists. The optimizer synthesizes a node the very
same runtime then refuses to execute. The node named in the error **is not in the file on disk**,
which is what makes the message confusing on first read.

The obvious explanation — "x86_64 has the fp16 kernel, aarch64 does not" — is only half right,
and the wrong half is the half that matters. *Every* aarch64 ORT build checked lacks that kernel,
**including macOS arm64, where the model loads fine**. What differs is whether the fusion fires:

| platform | fp16 kernel | fuses fp16? | result |
|---|---|---|---|
| linux/amd64 | yes | yes | loads |
| **linux/arm64** | **no** | **yes** | **fails** |
| macOS arm64 | no | **no** | loads |

So it is a build-configuration divergence between two targets of the same ORT release, not an
architecture property. `crates/diar-core/src/ort_compat.rs` caps optimization at `Level1` for
that one model on aarch64 — a no-op on the platform that declines the fusion anyway. The 15
diarization graphs keep full optimization.

Full analysis, the measured alternatives, and three traps around the escape hatches (the
optimizer is named `GeluFusionL2`, a wrong name is **silently ignored**, and the separator is
`;` not `,`): [`docs/ORT_FUSION_FP16_AARCH64.md`](docs/ORT_FUSION_FP16_AARCH64.md) and
RESULTS §7.40.

## 7. Ground rules

- `transcribe-app` and `pyannote-audio-fork` are **read-only** (other agents work there; the fork
  is production-pinned). Their harness scripts/images are *used*, never modified.
- Exported ONNX/PLDA artifacts derive from **gated** `pyannote/speaker-diarization-community-1`
  (CC-BY-4.0 + terms): regenerate locally, never commit, never redistribute.
- Benchmark GPUs: A6000s (GPU 0/2) for engine A/B timing; 3080 Ti (GPU 1) for serving spikes.
- Publishable numbers only from Tier-A corpora (AMI, VoxConverse, Earnings-21).
