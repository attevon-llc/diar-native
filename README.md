# diar-native — Fast, Deployable Speaker Diarization for OpenTranscribe

Native (Rust/ONNX) speaker diarization engine evaluation and (pending validation) implementation,
targeting the **pyannote `speaker-diarization-community-1`** pipeline that OpenTranscribe runs in
production — at equal accuracy, higher speed, lower deployment weight, and with a clean path to
**Triton Inference Server** and **AWS GPU** serving.

> **Status: Phase A/B — validation.** Every measurement is logged in
> [`validation/RESULTS.md`](validation/RESULTS.md). The test matrix and accept/reject gates are in
> [`validation/TESTPLAN.md`](validation/TESTPLAN.md). Nothing here modifies the production repos.

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
├── docker/Dockerfile.bench    ← self-contained speakrs CUDA build (xtask diarize CLI)
├── triton/models/             ← Triton spike model repo (config.pbtxt per model)
├── refs/                      ← staged references (AMI test RTTM+UEM, Karpathy fixed RTTM)
├── models/                    ← [gitignored] self-exported community-1 ONNX + PLDA (gated weights)
├── vendor/speakrs/            ← [gitignored] upstream clone @ pin (re-clone to reproduce)
└── results/                   ← RTTMs, timing JSONL, DER JSONs per run tag
```

Planned (post-validation, C-pass path): `crates/diar-core` (speakrs wrapper: centroids,
`embed_window`, speaker-count constraints, dual outputs), `crates/diar-server` (axum sidecar:
`/diarize`, `/embed_window`, `/healthz`), `crates/diar-ffi` (C ABI for Triton custom backend),
`crates/diar-cli`. Integration contract with OpenTranscribe (per its `SpeakerDiarizer`):
segments + exclusive segments + per-speaker L2-normalized 256-d centroids (OpenSearch kNN) +
ad-hoc window embedding (boundary recheck) + min/max/num-speaker constraints.

## 6. Deployment tiers (product decision, 2026-08-19)

| tier | target | design |
|---|---|---|
| **T1 — embedded/sidecar (DEFAULT, open source)** | laptops, small computers, single-GPU boxes | speakrs core + our patch set (folded seg, multimask-batching fix, fbank pool), **shared-weights concurrency**: one model set (Arc-shared ORT sessions — thread-safe `run()`, weights loaded once) + per-request scratch buffers + job queue, mirroring OpenTranscribe's Celery shared-weights pattern. CPU-only works (ORT CPU EP, ≈ eager-torch parity measured). No Triton, minimal RAM. |
| **T2 — Triton (opt-in, larger home servers + AWS)** | multi-user / multi-job servers | tritonserver + TRT engines (fp32), dynamic batching across concurrent jobs (measured 2.14× throughput at 8 clients on one weight copy), `diar-ffi` custom backend or sidecar-calls-Triton topology. Higher RAM/system footprint — that's why it's opt-in, not default. |

The Python fork path remains the universal fallback behind a config flag in both tiers.

## 7. Ground rules

- `transcribe-app` and `pyannote-audio-fork` are **read-only** (other agents work there; the fork
  is production-pinned). Their harness scripts/images are *used*, never modified.
- Exported ONNX/PLDA artifacts derive from **gated** `pyannote/speaker-diarization-community-1`
  (CC-BY-4.0 + terms): regenerate locally, never commit, never redistribute.
- Benchmark GPUs: A6000s (GPU 0/2) for engine A/B timing; 3080 Ti (GPU 1) for serving spikes.
- Publishable numbers only from Tier-A corpora (AMI, VoxConverse, Earnings-21).
