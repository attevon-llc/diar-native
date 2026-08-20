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
  open-source install matrix (CPU-only boxes, future Apple Silicon).
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
│   │                            centroids, embed_window, exclusive output, gender, media decode
│   ├── diar-server/           ← the T1 sidecar: /diarize /embed_window /healthz (axum);
│   │                            per-request engine handles — jobs run concurrently
│   └── diar-cli/              ← bench/ops runner (RTTM+JSON out, engine traces via RUST_LOG)
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
│                                (reproduce: clone attevon-llc/speakrs, checkout
│                                attevon/production-0.2.0 for the patch pre-applied as commits,
│                                or checkout b0756b1 + `git apply patches/0001-…patch`)
└── results/                   ← RTTMs, timing JSONL, DER JSONs per run tag
```

Integration contract with OpenTranscribe (per its `SpeakerDiarizer`), all implemented:
segments + exclusive segments + per-speaker un-normalized 256-d centroids (OpenSearch kNN
normalizes) + ad-hoc window embedding (boundary recheck) + optional per-speaker gender.
Still open: min/max/num-speaker constraints (T9b — forced counts currently warn + auto-count).

## 6. Deployment tiers (product decision, 2026-08-19)

| tier | target | design |
|---|---|---|
| **T1 — embedded/sidecar (DEFAULT, open source — SHIPPED as `diar-server:0.2.0`)** | laptops, small computers, single-GPU boxes | speakrs core + our patch set (folded seg, multimask-batching fix, fbank pool, VBx vectorization, exclusive-overlap fix, **shared sessions**, **pipelined fbank∥GPU**): one model set (Arc-shared ORT sessions, weights + arenas loaded once) + per-request scratch handles — concurrent jobs at one engine's VRAM, verified output-identical to serial. Ingests original media directly (symphonia + FFT resample). CPU-only works (ORT CPU EP, ≈ eager-torch parity measured). `SPEAKRS_ARENA_SHRINK=1` drops the between-job VRAM floor 4.5 GB → 1.1 GB for small cards (~20% per-job cost). No Triton, minimal RAM. |
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
which trades away the transcribe∥diarize overlap. Apple Silicon is future work, after Linux is
stable.

## 7. Ground rules

- `transcribe-app` and `pyannote-audio-fork` are **read-only** (other agents work there; the fork
  is production-pinned). Their harness scripts/images are *used*, never modified.
- Exported ONNX/PLDA artifacts derive from **gated** `pyannote/speaker-diarization-community-1`
  (CC-BY-4.0 + terms): regenerate locally, never commit, never redistribute.
- Benchmark GPUs: A6000s (GPU 0/2) for engine A/B timing; 3080 Ti (GPU 1) for serving spikes.
- Publishable numbers only from Tier-A corpora (AMI, VoxConverse, Earnings-21).
