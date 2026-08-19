# Diarization Engine Validation — Results Log

**Purpose:** permanent record of every measurement in the speakrs-vs-pyannote-fork validation
(Phases A/B of the plan) so no test ever needs re-running. Update this file with every new result.

- **Date started:** 2026-08-19
- **Host:** 2× RTX A6000 (GPU 0, 2), RTX 3080 Ti (GPU 1), CUDA 13.x driver, Linux 6.8
- **Fork under test:** `davidamacey/pyannote-audio` branch `gpu-optimizations` @ `a3f38afb` (production pin)
- **speakrs under test:** `avencera/speakrs` @ `b0756b1` (2026-07-20, v0.5.x), vendored at `vendor/speakrs`
- **Pipeline:** `pyannote/speaker-diarization-community-1` (primary model of OpenTranscribe)
- **Runner images:** `opentranscribe-backend:latest` (fork; torch 2.11.0+cu128), `diar-bench:latest`
  (speakrs; ORT 1.24.2 GPU, CUDA 12.8 base, built by `docker/Dockerfile.bench`)
- **Scoring:** `validation/score_der.py` — pyannote.metrics DER, collar 0.25, overlap INCLUDED,
  AMI scored with official UEMs. Raw JSONs in `results/`.

## 1. Model exports (Phase B1) — DONE

Exported with speakrs `scripts/export_models.py` **inside the backend image, offline, from the
app's own HF cache** → guarantees the exact production community-1 checkpoints. Output: `models/`
(segmentation-3.0{,-b32,-b64}.onnx, wespeaker-voxceleb-resnet34{,-b32,-b64,-tail*}.onnx,
wespeaker-fbank{,-b32}.onnx, wespeaker-multimask-tail{,-b32}.onnx, plda_{lda,tr,mu,psi,mean1,mean2}.npy).

Verified facts:
- Fresh seg export op census: `Sin`×2 `Cos`×2 `If`×1 `LSTM`×4 (community-1 = 4-layer) `InstanceNorm`×4.
- speakrs' fused embedding graph: inputs `waveform (B,1,160000)` + `weights (B,589)` — kaldi fbank
  is INSIDE the graph (framing + DFT/mel matmuls; no Sin/Cos/STFT ops).
- **Checkpoint identity:** sha256 of cached checkpoints proves community-1's `segmentation/` and
  `embedding/` weights DIFFER from `pyannote/segmentation-3.0` and standalone
  `wespeaker-voxceleb-resnet34-LM`. The pre-existing ONNX artifacts in
  `transcribe-app/models/onnx/` were exported from the WRONG (fallback 3.1) models —
  any old E2E numbers using them measured a different model.

## 2. Stock-fork baselines (Phase A) — Gate 0 PASSED

### 2.1 Frozen-baseline reproduction (A6000, GPU 0, 5 runs, canonical harness)
`benchmark-pyannote-direct.py --variant optimized` in the `diarization-probe` compose service.
Scored vs frozen `baseline_a6000_20260421_213621_short` with `diarization-der-compare.py`:

| file | runs | DER vs frozen | spk | segs | wall mean (cv) |
|---|---|---|---|---|---|
| 0.5h_1899s | 5 | **0.0000% ×5** | 4=4 | 703=703 | 22.41 s (2.9%) |
| 2.2h_7998s | 5 | **0.0000% ×5** | 3=3 | 2855=2855 | 104.09 s (1.6%) |

→ Fork reproduces **bit-identically** on current drivers; fully deterministic. Peak VRAM 845 MB.
Stage split (2.2h): seg 5.5 s, embeddings ~78 s, clustering ~13 s, reconstruction+discrete ~7 s.

### 2.2 AMI test-16 (A6000 GPU 2, 1 run, `results/rttm/fork_ami_test16`)
**Aggregate DER 13.093%** (collar 0.25, overlap incl., UEM-cropped), per-file 6.34–19.98%,
speakers within ±1 of reference everywhere (4-5 vs 4, 3 vs 3). RTF ≈ **80×** RT. Peak VRAM 886 MB.
Per-file table in `results/fork_ami_test16_der.json`.

### 2.3 Karpathy 66.5-min acceptance clip (A6000 GPU 0, 3 runs, `results/rttm/fork_karpathy`)
**DER 8.194%**, identical across all 3 runs (deterministic); 2 speakers exact; **83× RT** (48.0 s).
NOTE: staged reference at `refs/karpathy/karpathy.rttm` fixes speaker names containing spaces
("Sarah Guo" → "Sarah_Guo") which break RTTM parsing — original reference untouched.

### 2.4 Duration curve (1.0h/3.2h/4.7h ×3, canonical harness, GPU 0) — RUNNING
Results → `transcribe-app/benchmark/results/rttm/curve_20260819` + this file when done.

## 3. speakrs runs (Phase B3) — IN PROGRESS

Runner: `validation/run_speakrs.sh` (one process per file per run; RTTM from stdout;
exit codes untrusted — see §5 bug list). Mode `cuda` (1.0 s seg step = pyannote-equivalent;
`cuda-fast` never used). Models = our §1 exports.

- Smoke: clip30.wav → CPU 15.8 s / CUDA 1.4 s, identical segments across modes.
- AMI test-16 on A6000 (GPU 2): IN PROGRESS. Early per-file wall: IS1009a (26 min) 31.8 s ≈ 49×;
  ES2004a-d (~36 min) 68–75 s ≈ **31× RT** → **~2.5× slower than the fork (80×) on the same GPU**.
  Cause identified in §4: fused-fbank ORT-CUDA tax.

## 4. ORT/Triton serving spike (Phase B4 + user-approved Triton deploy) — KEY FINDINGS

### 4.1 Constant folding (onnxsim) of segmentation graph — bit-exact, kills all problem ops
- 179 nodes → **40 nodes**; `Sin`/`Cos`/`If`/`Slice`/`MatMul` ALL eliminated;
  remaining: Conv×3, InstanceNormalization×4, LSTM×4 + elementwise.
- Parity vs unfolded: **max_abs_diff = 0.000e+00** (bit-exact), argmax mismatch 0.
- → The phase-6 "ORT CUDA EP has no Sin/Cos kernels" blocker is fully removed by folding.
  Artifacts: `models/segmentation-3.0{,-b32}-sim.onnx`.

### 4.2 Triton Inference Server spike (tritonserver:26.06-py3, ONNX Runtime backend, RTX 3080 Ti GPU 1)
Model repo `triton/models/` (fixed batch-32 graphs, max_batch_size 0, KIND_GPU).
Server-side `compute_infer` per batch-32 request:

| model | compute_infer | note |
|---|---|---|
| seg_unfolded (b32) | 190.0 ms | Memcpy nodes inserted; CPU-fallback tax |
| seg_folded (b32-sim) | **96.6 ms** | **2.0× faster than unfolded** — folding win confirmed |
| torch eager seg (same GPU, ref) | 15.6 ms | cuDNN fused LSTM |
| embedding FUSED fbank (speakrs graph) | **811.8 ms** | fbank subgraph ~690 ms of it |
| embedding fbank-OUTSIDE (ResNet only) | **120.4 ms** | 6.7× faster than fused |

Conclusions (3080 Ti; re-validate headline numbers on A6000 later):
1. **Folding works and is mandatory** for any ORT-CUDA use of segmentation (2× + removes Memcpys).
2. **ORT-CUDA LSTM remains ~6× slower than torch/cuDNN** even folded → for serving segmentation
   at eager speed, use TensorRT plan or a torch/python backend; or accept it (seg = 3% of GPU time).
3. **The fused-fbank embedding graph is the speakrs CUDA bottleneck** (~690 ms/batch tax on
   ORT-CUDA). This explains speakrs' ~31× vs fork's ~80× RT. Fix path: compute fbank outside
   (batched torch/kaldi-native-fbank) + serve ResNet-only graph, or optimize the fused subgraph
   (framing-as-Conv instead of unfold ops). HIGH-VALUE, actionable for both speakrs and Triton.
4. ORT CPU (backend image, b32): seg ~290 ms/batch folded or not (host CPU) — folding is a
   GPU-placement fix, not a CPU speedup.

### 4.3 Production-image defect found (document for transcribe-app)
`opentranscribe-backend:latest` ships `onnxruntime-gpu==1.28` built for **CUDA 13**
(`libcublasLt.so.13`) while the image's pip CUDA libs are cu12 (torch 2.11+cu128 stack,
cuDNN 9.19). → **CUDA EP cannot load in the production image at all** (app code never uses it,
so dormant). Any future in-app ORT-GPU work must pin an ORT cu12 build or add cu13 libs.

### 4.4 TensorRT engines (trtexec in 26.06 image, TRT 11.0, RTX 3080 Ti = sm_86, same as A6000)

Both engines built **successfully on the first attempt** from the dynamic-axis ONNX exports —
including segmentation's 4-layer BiLSTM **with unfolded Sin/Cos/If** (TRT parses them natively).
**The historical "TRT can't run pyannote" claim is dead**; the phase-6 failure was the ORT-TRT-EP
shape-profile bug, not TensorRT. Profiles: seg min 1×1×160000 / opt+max 32×1×160000;
emb (fbank-outside) min 1×998×80 / opt 32 / max 64. Engines: `triton/engines/*_fp32.plan`
(seg 8.5 MB, emb 27 MB). NOTE: speed-representative exports (3.1-era weights, identical
architecture); accuracy-correct community-1 engines to be rebuilt from fresh exports before adoption.

Server-side `compute_infer`, batch-32, `tensorrt_plan` backend with dynamic batching enabled:

| model | TRT fp32 | ORT-CUDA best | speedup vs ORT | torch eager ref |
|---|---|---|---|---|
| segmentation | **45.3 ms** | 96.6 ms (folded) | 2.1× (4.2× vs unfolded) | 15.6 ms (cuDNN LSTM) |
| embedding (fbank-outside) | **49.5 ms** | 120.4 ms | 2.4× (**16.4×** vs fused-fbank 811.8 ms) | — |

Reading: TRT is the right backend for BOTH models on Triton. Segmentation stays slower than
eager torch (TRT LSTM ≠ cuDNN fused) but seg is ~3% of pipeline GPU time. Embedding at 49.5 ms/b32
on a 3080 Ti is the serving-competitive number; A6000 re-run = M10. fp16 engines building (speed
test only — fp16 adoption REQUIRES StatsPool-pinned mixed precision + DER re-validation, per the
fork's 26-33% DER collapse finding, treated as a re-test hypothesis).

### 4.5 Gate G1 — PASSED (speakrs AMI test-16 accuracy parity)

| engine | aggregate DER | note |
|---|---|---|
| fork (A) | 13.093% | §2.2 |
| speakrs (B) | **13.100%** | **+0.007 pp — 14× inside the +0.1 pp gate** |

Per-file DERs track within ~0.5% everywhere; **per-file speaker counts are IDENTICAL between
engines on all 16 files** — including the same 5-vs-4 overcounts on ES2004a/b/c + IS1009b.
speakrs is algorithmically faithful to community-1. `results/speakrs_ami_test16_der.json`.
speakrs AMI RTF ≈ 31-49× vs fork 80× (fused-fbank ORT-CUDA tax, §4.2) — speed verdict at G4.

### 4.6 M10 — A6000 serving numbers (tritonserver 26.06, engines built ON the A6000, fp32)

Server-side `compute_infer`, batch-32: **emb_trt 37.4 ms** | embedding_fbankout(ORT-CUDA) 102.2 ms |
seg_trt 59.2 ms | seg_folded(ORT-CUDA) 100.4 ms. Cross-check vs fork eager E2E (embedding stage
78 s / 2.2 h file ≈ 104 ms per batch-32-equiv incl. fbank): **TRT embedding ≈ 1.9× the eager
stage** — independently reproduces the phase-6 "TRT EP 1.96×" microbench. Ops lesson: TRT wins
conv/matmul (fusion); torch/cuDNN wins LSTM (fused sequence kernel; TRT/ORT run generic loops).
NOTE: cp-while-writing race produced a 0-byte plan on first deploy (fixed); cross-SKU plan reuse
(3080Ti→A6000) triggers a TRT warning — build per-device (both are sm_86 so it *loads*, but
rebuild is correct practice).

### 4.7 M11 (model-level) — concurrency + dynamic batching on the A6000

emb_trt (max_batch 64, dynamic batching), clients sending batch-8 requests:

| clients | throughput | p50 | p95 |
|---|---|---|---|
| 1 | 364 items/s | 20.6 ms | 27.7 ms |
| 2 | 643 items/s | 23.1 ms | 34.6 ms |
| 4 | 742 items/s | 41.1 ms | 53.3 ms |
| 8 | **780 items/s (2.14×)** | 81.8 ms | 97.6 ms |

Saturation ≈ 4 concurrent clients. Multi-request serving on one A6000 does >2× the embedding
work of serial — the AWS concurrency thesis holds at the model level. Full-pipeline M11 pending.

### 4.8 speakrs fused-vs-split embedding — the fix already exists upstream (PR opportunity)

Source reading (`src/inference/embedding/{load,batch}.rs`, `load/sessions.rs`):
- CUDA batch path uses the **fused** session (`primary_batched_session`, waveform input) — the
  one measured at 812 ms/b32 on ORT-CUDA (§4.2).
- A **split fbank+tail path fully exists** (sessions, buffers, exported models
  `wespeaker-fbank.onnx` + `-tail*.onnx`) but is wired only for **CoreML** (fbank session built
  on CPU, native tail on ANE/GPU).
- **Proposed upstream PR: enable the split path for `ExecutionMode::Cuda`** (fbank CPU session +
  tail on CUDA). Measured ceiling: ResNet-only = 6.7× faster on ORT-CUDA, 16× under TRT.
  Projected: speakrs ~31× RT → fork-class ~80× or better (concurrency overlaps CPU fbank with
  GPU tail). Evidence package for the PR: §4.2/§4.7 tables + §4.5 DER parity.

### 4.9 fp16 engines — deferred (TRT 11 strongly-typed)

TRT 11 (26.06 image) removed `--fp16`: networks are **strongly typed by default** — engine
precision follows the ONNX graph's tensor types. fp16 therefore requires converting the ONNX
graph itself (onnxconverter-common float16 pass) with the StatsPool variance/sqrt subgraph kept
fp32 (the fork measured 26-33% DER collapse from fp16 there). That graph surgery + mandatory DER
re-validation = Milestone-1 work if speakrs is adopted; NOT a spike. fp32(+TF32 default) numbers
in §4.4 are the honest serving baseline.

## 5. Bugs/quirks found (upstream-reportable)

1. **speakrs teardown crash:** `corrupted double-linked list` (glibc) at process exit in `cuda`
   mode AFTER results are flushed — ORT CUDA EP unload vs mimalloc interplay. Workaround:
   don't trust exit codes; validate RTTM output content (run_speakrs.sh does).
2. **speakrs xtask build without BLAS:** `cargo build -p xtask --features cuda` fails
   (`speakrs requires a BLAS backend`) because xtask pins `default-features = false`.
   Fix: `--features cuda,speakrs/openblas-system` (Dockerfile.bench does this).
3. **ORT provider lib lookup:** ORT resolves `libonnxruntime_providers_shared.so` relative to the
   BINARY's directory, not ldconfig → symlink provider libs into `/usr/local/bin/` (Dockerfile.bench).
4. **Karpathy reference RTTM has spaces in speaker names** → breaks any standard RTTM parser
   (loads as empty → silent DER 100%). Staged fixed copy in `refs/karpathy/`.
5. **VoxConverse zip on NAS was truncated/corrupt** (no EOCD). Re-downloading from official
   Oxford VGG URL (CC-BY-4.0). Corrupt file kept as `.corrupt`.
6. **FORK BUG — CPU-only Linux crash:** `speaker_diarization.py:559` `_gpu_empty_cache()`:
   when CUDA is unavailable it calls `torch.mps.empty_cache()` (guarded only by `hasattr`,
   which is True on Linux torch builds) → `RuntimeError: Cannot execute emptyCache() without
   MPS backend`. **Every pure-CPU Linux deployment (docker-compose.lite class) crashes.**
   Production never sees it (GPU always present). Fix (for a future fork PR, NOT applied —
   repo read-only): guard with `torch.backends.mps.is_available()`. Benchmark workaround:
   run container with a GPU visible but `--device cpu`.
7. **TRT 11 removed `--fp16`** (strongly-typed networks by default) — see §4.6.

## 6. Pending

- [ ] Fork duration curve 1.0h/3.2h/4.7h ×3 (running, GPU 0)
- [ ] speakrs AMI test-16 finish + DER scoring vs §2.2
- [ ] speakrs Karpathy ×3, duration curve ×3 (G2/G3/G4/G5 gates)
- [ ] CPU legs: fork vs speakrs on 0.5h file (lite-deployment comparison)
- [ ] VoxConverse dev (216 files) both engines — official comparison corpus for speakrs' claims
- [ ] A6000 re-run of §4.2 headline numbers when a GPU frees
- [ ] Gate evaluation G1–G5 + go/no-go report
