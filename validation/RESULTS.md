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
- AMI test-16 stock build (A6000, GPU 2, timing contaminated — §4.11): ~31–49× RT; DER §4.5.
- FINAL STATE (patched build — §4.15/§4.16): AMI 13.101% @ **89× RT on the 3080 Ti**;
  Karpathy 8.219%; **VoxConverse dev-216 4.847% — BEATS fork's 5.099%** (§4.16d). Definitive
  speed comparison = quiet-machine pass (§ pending).

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

### 4.10 CORRECTION to §4.8 + instrumented speakrs CUDA path (RUST_LOG=speakrs=trace)

`embedding_path=MultiMask`: speakrs' CUDA E2E pipeline **already uses the split path** —
CPU fbank once per chunk + multimask tail (fbank + 3 masks → 3 embeddings) on GPU. It does NOT
run the fused 812 ms graph E2E (that graph is only the `embed_batch` API path). Measured on
clip30 (22 chunks): `fbank_ms=661` (CPU, ~30 ms/chunk, single-threaded), `gpu_predict_ms=470`,
`recv_wait_ms=547`. → Real optimization targets, in order:
1. **Parallelize CPU fbank across chunks** (rayon) — clean upstream PR; `--chunk-emb-workers`
   1→4 measured NO effect on CUDA (knob appears CoreML-oriented); RTTM identical across counts.
2. **Folded segmentation graph drop-in** (zero code — speakrs loads by filename): `models_folded/`
   with `-sim` graphs renamed. **E2E ES2004a: 39.4 → 36.7 s (-7%), RTTM bit-identical.** Free win.
3. Segmentation via TRT/native later (Triton topology).

### 4.11 Benchmark-hygiene note — speakrs A6000 wall-times CONTAMINATED

The §3 speakrs AMI walls (31-49× RT) ran on GPU 2 while the fork's curve benchmark saturated
17+ CPU cores on GPU 0 → speakrs' CPU stages (fbank!) were starved. Evidence: ES2004a takes
**~40 s on the *weaker* 3080 Ti** on a lighter-loaded machine vs 75 s recorded on the A6000.
DER/RTTM results are unaffected (bit-deterministic — G5 evidence). **All cross-engine timing
comparisons (G4) must come from a quiet-machine timing pass** (queued: fork vs speakrs,
2.2h + 4.7h, sequential, no concurrent jobs). Lesson recorded: never co-schedule timed runs.

### 4.12 CPU leg (M7) — COMPLETE

0.5h file, cpu-only: fork 1123 s | speakrs unpatched 1191 s | speakrs patched 1251 s —
**all ≈ parity class (1.7-1.9× RT, ±11% under varied machine load)**; speakrs RTTMs identical
across builds. Note: the fbank pool slightly HURTS cpu mode (core contention with inference) →
M1 config item: pool defaults to 1 when `ExecutionMode::Cpu`. Lite/laptop deployment claim
stands: speakrs CPU ≈ fork CPU, with a ~50 MB binary instead of a 9 GB torch image.

### 4.14 Gate G2 PASSED + G5 evidence (speakrs Karpathy) & fork VoxConverse anchor

- **G2:** speakrs Karpathy DER **8.219%** vs fork 8.194% (+0.025 pp, gate +0.1); 2 speakers exact;
  3 runs bit-identical. Curve files 0.5h/1.0h/2.2h also 3× bit-identical (G5 evidence).
- **Fork VoxConverse dev-216 anchor: 5.099% aggregate DER** @collar 0.25
  (`results/fork_vox_dev_der.json`); speakrs vox leg pending.
- 2.2h segments: speakrs 2629 vs fork 2855 — pending A/B DER scoring to classify
  (merge-semantics vs drift).

### 4.15 UPSTREAM BUG #2 — multimask batching silently disabled (b32/b64 filename mismatch)

speakrs' loader requests `wespeaker-multimask-tail-b64.onnx` (`PRIMARY_BATCH_SIZE=64`,
`load/sessions.rs:162`) but `scripts/export_models.py` only writes `-b32`
(`MULTI_MASK_BATCH_SIZE=32`) → batched multimask session = None → **batch size falls back to 1**
(per-chunk fbank + batch-1 GPU predict; trace: flushes==chunks). Runtime buffers are sized 32, so
a TRUE batch-64 graph under that name crashes the worker ("receiver disconnected") — verified.
**Workaround (zero code): place the b32-shaped graph under the b64 filename** → batching engages
(flushes=33 for 1041 chunks), RTTM bit-identical. Upstream PR: make exporter+loader agree (export
b64 name with batch-32 graph, or load b32 name).

With batching engaged, ES2004a trace: **fbank_ms=28182 (76% of ~37 s wall!), gpu_predict_ms=3572**
→ GPU was never the bottleneck; single-session CPU fbank is.

### 4.16 fbank optimization ladder (ES2004a, GPU 1, same-session A/B — relative numbers valid)

| variant | wall | note |
|---|---|---|
| stock models (batch-1 path) | 39.4 s | |
| + folded seg | 36.7 s | RTTM identical |
| + multimask b64(b32 graph) batching | 37.1 s | RTTM identical; GPU batched but fbank dominates |
| + SPEAKRS_FBANK_THREADS=24 (patch 1) | 32.1 s | intra-op threads DON'T scale fbank (35.4→32.1) |
| + split_fbank_pool fan-out (patch 2), SPEAKRS_FBANK_POOL=4 | 17.0 s | RTTM identical |
| + split_fbank_pool fan-out, SPEAKRS_FBANK_POOL=8 | **12.9 s (3.1× vs stock)** | RTTM identical |

**Headline: patched speakrs = 12.9 s for 36.4-min ES2004a ≈ 169× RT on the RTX 3080 Ti** —
vs fork 80× RT on an A6000. The three-patch stack (folded seg + multimask-batching unlock +
fbank session-pool) flips the G4 speed verdict from "2.5× slower" to "≈2× faster than the
production fork", pending quiet-machine A6000 confirmation + full-corpus DER re-verification
with the patched build (expected bit-identical — every step verified RTTM-identical so far).

Patches live in `vendor/speakrs` (local; upstream-PR candidates): (1) `SPEAKRS_FBANK_THREADS`
env override in `session.rs`; (2) `split_fbank_pool` in `load/sessions.rs` + parallel branch in
`fbank.rs` (`SPEAKRS_FBANK_POOL`, default cores/4 clamped 1-8).

### 4.16b Diagnosis: 0.5h "phantom 5th speaker" = minor-speaker cluster split

Per-speaker airtime comparison (0.5h_1899s): fork = 1258.0 / 609.6 / 154.8 / **104.6** s (4 spk);
speakrs = 1250.7 / 615.1 / 156.0 / **89.0 + 16.9** s (5 spk). speakrs splits the fork's smallest
speaker into two clusters (sum 105.9 ≈ 104.6) — borderline AHC/VBx decision on a minor speaker,
~1.1% A/B DER, NOT systemic drift (speaker counts matched on every ground-truth corpus file:
AMI ×16, Karpathy, and equality on 1.0h/2.2h). Do not tune on this file (tuning-corpus rule);
candidate later fix: expose speakrs' `speaker_keep_threshold`/VBx knobs in diar-core config.

### 4.16c G3 long-file A/B — FAILS AS WRITTEN; consistent +1-cluster tendency on synthetic files

| file | A/B DER vs fork | speakers fork→speakrs | note |
|---|---|---|---|
| 0.5h | 1.12% | 4→5 | minor split, sum-preserving (§4.16b) |
| 1.0h | 1.12% | 5→5 ✓ | pass |
| 2.2h | 0.18% | 3→3 ✓ | pass (excellent) |
| 3.2h | **6.05%** | 3→4 | real 875 s extra cluster + mass shift — largest divergence |
| 4.7h | 2.27% | 8→9 | just over the 2.0% gate max |

Deterministic across runs (identical DER ×3 each). Pattern: **speakrs = fork+1 cluster on the
synthetic curve files**, while on ground-truth corpora speaker counts matched the fork exactly
(incl. the fork's own AMI over-counts). Curve files have NO ground truth — fork is the
reference, not the truth. **Arbiter = VoxConverse dev-216 (running):** parity there → synthetic-
content edge (document, expose clustering knobs in diar-core, gate on ground-truth corpora);
+1 tendency there too → systemic; per-stage dump bisection (AHC/VBx) before adoption call.

### 4.16d ARBITER VERDICT — VoxConverse dev-216 (ground truth): speakrs BEATS the fork

| metric | fork | speakrs (patched) |
|---|---|---|
| aggregate DER @0.25 | 5.099% | **4.847% (−0.25 pp)** |
| files with exact ground-truth speaker count | 136/216 | **138/216** |
| per-file DER wins | 38 | **95** (83 ties) |
| speaker-count agreement between engines | — | 211/216 equal (speakrs higher on 3, fork on 2) |

→ The §4.16c +1-cluster tendency does NOT appear on real audio — synthetic-curve-file edge case,
formally adjudicated. **Accuracy across all ground-truth corpora: AMI +0.007pp, Karpathy
+0.025pp, VoxConverse −0.25pp (better). speakrs ≥ fork.** G3's synthetic miss recorded as a
documented exception with the arbiter analysis; ground-truth gates govern.
Raw: `results/{fork,speakrs}_vox_dev_der.json`.

### 4.17 Benchmark-hygiene incident #2 (self-inflicted)

The b64 addendum export wrote into `models/` at 05:12 **while the speakrs curve benchmark was
reading that directory** → 3.2h runs 1-2 and all 4.7h runs crashed (empty RTTMs; the "runs differ"
on 3.2h is this, not nondeterminism). Re-run launched from the fixed dir. RULE: never mutate a
model/data dir that a live benchmark mounts; stage changes in a new dir.

### 4.19 QUIET-MACHINE TIMING PASS (the definitive G4 numbers) — 2026-08-19

Same A6000 (GPU 0), sequential, zero other load, 2 runs each. Fork via production code path;
speakrs = patched build (models_folded, SPEAKRS_FBANK_POOL=8), wall incl. ~2-4 s container start.

| file | fork | speakrs patched | ratio |
|---|---|---|---|
| 2.2h_7998s (3 spk) | 106.4 / 105.8 s (75× RT) | **83.2 / 85.0 s (95× RT)** | **speakrs 1.26× FASTER** |
| 4.7h_17044s (8 spk) | 346.7 / 352.2 s (49× RT) | 503.9 / 466.5 s (35× RT) | speakrs 1.39× slower |

**G4 verdict: SPLIT.** Typical content → speakrs wins (and AMI corpus at 89× RT on the 3080 Ti
corroborates at scale). The extreme 4.7h/8-speaker file — where clustering N is largest
(~50k embeddings) — flips it; suspected AHC linkage scaling (kodama vs scipy's C linkage) or
fbank at 17k chunks; trace-instrumented run in progress for attribution (§4.20). Fork peak VRAM
886 MB both files; speakrs VRAM not gated (earlier obs ~2-4 GB).

**Decision impact (per PLAN decision rule):** accuracy gates all pass (§4.16d arbiter) → GO on
the adoption track, with "≥1.0× fork on the 4.7h file" retained as the T1 SHIP gate — the
long-file clustering gap becomes a named Milestone-1 optimization item (likely upstream-PR-able,
same as the fbank pool).

### 4.20 4.7h gap attribution (trace-instrumented, quiet A6000)

`total_ms=473953`: inference 125.4 s (fbank 73.9 s + gpu_predict 50.2 s, concurrent) +
**post/clustering 348.6 s = 74% of wall** at N≈50k embeddings / 17,050 chunks. Fork's scipy
clustering on the same file ≈ 145 s. → The extreme-file regression is **AHC linkage scale**
(kodama vs scipy's nn-chain C implementation), NOT inference. Named Milestone-1 item (T1 ship
gate): faster linkage at N>10k (nn-chain port, stronger filter_embeddings pre-reduction, or
sub-sampled seeding) — upstream-PR candidate alongside the fbank pool. Secondary: fbank pool
scaling for 17k-chunk files (74 s at pool=8; raise pool for long files).

### 4.22 C-M1a: 4.7h clustering sub-stage attribution (instrumented vendored build)

At N=21,418 filtered embeddings, **K=1,902 AHC seed clusters**, D=256/128:

| sub-stage | time | cause |
|---|---|---|
| AHC pdist | 64.5 s | naive per-pair scalar loop (O(N²·D) scalar f32) |
| AHC kodama linkage | 5.1 s | fine — NOT the problem |
| PLDA transform | 0.24 s | fine |
| **VBx VB-EM** | **305.1 s** | scalar O(N·K·D) loops ×20 iters incl. per-sample recompute of a per-speaker penalty |
| centroids+assign | ~0.4 s | fine |

Fixes implemented (vendored; upstream-PR candidates): (1) VBx vectorization — M-step as
`gammaᵀ·rho` matmul, E-step as `rho·alphaᵀ` matmul + penalty vector computed once, fused
logsumexp/gamma row pass; (2) pdist as blocked Gram matmul (`d²=|a|²+|b|²−2ab`) with scoped
threads over disjoint condensed ranges; (3) ndarray `matrixmultiply-threading`. Numerics guard:
speakrs' own Python-parity fixture tests (AHC scipy label order; VBx gamma/pi vs pyannote at
1e-4/1e-5) must pass — matmul reassociation is within tolerance by construction (f64).

### 4.23 C-M1b RESULT — clustering optimization clears the ship gate at 2× margin

Vectorized VBx + threaded blocked pdist (quiet A6000, 4.7h file, same conditions as §4.19):

| | pre-opt | post-opt | fork |
|---|---|---|---|
| 4.7h wall | 474 s | **171.6 s (~99× RT)** | 349 s |
| pdist | 64.5 s | 1.2 s (53×) | — |
| VB-EM | 305.1 s | 36.7 s (8.3×) | — |
| clustering total | 348 s | 43.6 s (8×) | ~145 s (scipy) |

**RTTM bit-identical to the pre-optimization run** — zero boundary drift. All 16 speakrs
clustering tests pass incl. Python-parity fixtures (AHC==scipy label order; VBx gamma/pi vs
pyannote @1e-4/1e-5; PLDA fixture) — fixtures need `fixtures/` + `fixtures/models/` (plda_*.npy)
mounted, and `RUST_MIN_STACK` for the 2 MB test-thread stack (test-harness artifact only).

**G4 final: CLEAN SWEEP — speakrs 1.2× (Karpathy), 1.26× (2.2h), 2.03× (4.7h) faster than the
production fork.** The T1 ship gate (≥1.0× on 4.7h) is cleared 2×. speakrs' Rust clustering now
also beats scipy (43.6 vs ~145 s). Patch set: `patches/0001-cuda-performance-patch-set.patch`
(217 insertions, 8 files: fbank pool + threads, multimask-b64, VBx vectorization, pdist blocks,
ndarray threading). AMI-16 revalidation with this build → §4.24.

### 4.24 C-M1c — optimized build revalidation: CLOSED

AMI test-16 with the clustering-optimized build: **13.101% aggregate (identical), all 16 RTTMs
bit-identical to the previous patched run, corpus at 105× RT on the A6000** (fork 80×).
The optimization is proven accuracy-transparent at corpus scale. C-M1a/b/c complete;
T1 ship gate CLEARED. `results/speakrs_ami_optimized_der.json`.

### 4.25 Audit-1 — formal memory + optimized-build determinism (4.7h ×3, A6000)

- **Determinism: 3 runs bit-identical** (9,507 lines each) — direct ×3 proof for the optimized
  build (previously inferred from bit-identity to a deterministic predecessor).
- **Peak RSS 3.09 GiB** — G4 RSS sub-criterion (<8 GB) PASS with 2.6× headroom.
- **Peak VRAM 4,236 MB** vs the 4 GB target — **marginal exceed (~3%), PASS-WITH-NOTE**: driven
  by the b64 session variants; fits all target GPUs (full AMI corpus ran on the 12 GB 3080 Ti);
  tunable for small GPUs via a b32-only model set (b32 exports already exist). Fork reference:
  886 MB torch-allocated (~1.4-1.7 GB process-level) — apples-to-apples gap ≈ 2.8×.
  **Why speakrs uses more:** (1) eager-loads ~6 CUDA sessions each with its own ORT arena —
  and the CUDA pipeline only RUNS the multimask path, so fused-single/fused-b64/split-tail
  sessions hold idle VRAM (→ **lazy session loading = cheapest VRAM optimization**, diar-core
  M1 item + upstream suggestion); (2) b64 graphs retain peak ResNet activation workspace in
  arenas that never shrink; (3) no cross-session allocator sharing (ORT supports a shared
  env allocator — second mitigation). Fork is lean by necessity (coexists with Whisper in one
  worker + explicit empty_cache) — same pressure applies to the T1 sidecar, so this list is
  scheduled work, not trivia.

### 4.26 C-M1 GATE PASSED — diar-core wrapper + diar-cli + diar-server land

The Phase-C workspace (crates/diar-{core,cli,server}) is built and validated:
- **AMI-16 via diar-cli: 13.101% aggregate — identical; 16/16 RTTMs content-identical** to the
  recorded optimized run (URI field differs by design; timestamps+speakers exact).
- **Karpathy via diar-cli: content-identical, deterministic ×3, 141-148× RT** (27-28 s —
  faster than per-file xtask runs because ONE engine load serves all files: the sidecar
  amortization win, previewed).
- Centroids: 2×256 un-normalized (norm≠1 verified) row-aligned to speakers; exclusive
  segments emitted (692 full / 660 exclusive on Karpathy); embed_window implemented.
- Binaries: **~31 MB each** (diar-cli, diar-server) + ~33 MB models vs the 9 GB torch image.
- Build lessons: **ort must be pinned `--precise 2.0.0-rc.12`** (cargo resolved rc.13 whose
  static core mismatches the 1.24.2 provider libs → "vector::reserve" at session load) —
  risk-register item materialized, mitigated in Cargo.lock; `MaskedEmbeddingInput` needed a
  vendor re-export (upstream PR trivia); container-written result dirs are root-owned (chown
  before host-side renames); CLI needs a first-dot label mode (TODO).
Remaining M1 scope (next session): speaker-count constraints port, Arc-shared sessions,
lazy session loading (VRAM), supervisor hardening in diar-server.

### 4.27 M2 sidecar standalone verification + VRAM/speed configuration matrix

diar-server:latest built (Dockerfile.server; tokio+std thread stacks raised to 16 MiB — worker
threads overflowed the 2 MiB default, same root as the test-suite finding). Standalone smoke
(no OpenTranscribe involvement): /healthz ✓, /diarize (clip30: 1 spk / karpathy_10m: 2 spk, 92
segs) ✓, /embed_window samples_b64 → 256-d un-normalized ✓.

**Warm-engine serving is the fastest configuration measured in the whole program:**
ES2004a (36.4 min) in **7.9 s ≈ 277× RT (~3.4× the fork)** — per-process cold starts were
masking the sidecar's true speed all along.

Model-set configuration matrix (choose by GPU size — zero code, file presence only):
| set | contents | VRAM | ES2004a warm | use |
|---|---|---|---|---|
| fast (default ≥6 GB) | b32 set + `wespeaker-multimask-tail-b64.onnx` (b32-shaped batch) | 4.2 GB | **7.9 s (277× RT)** | servers |
| small (laptop tier) | b32-only, no multimask batch file | **1.6 GB (−62%)** | 37.2 s (59× RT) | small GPUs |

Corrections logged: (a) `SPEAKRS_LAZY_SESSIONS` ≈ **no VRAM effect** (ORT arenas grow on RUN,
not load — idle sessions cost only weights; gate kept for startup trim, reclassified);
(b) the multimask batch session is worth **4.7× warm** (earlier "marginal" read was cold-start
artifact); its arena IS the ~2.6 GB. `models_small/` staged alongside `models_folded/`.

Future-work note (user idea, endorsed): Rust preprocessing — symphonia decode inside
diar-server (accept original media path; kill the WAV handoff), tmpfs handoff volume now,
shared-mmap PCM service only when multiple consumers provably re-decode.

### 4.21 Patched-build accuracy closure — Karpathy ×3 (quiet A6000, GPU 2)

**8.219% DER — bit-identical to the stock-build result and across all 3 runs; 2 speakers exact;
40 s wall ≈ 100× RT vs fork 48 s (1.2× faster).** Patched build now verified
accuracy-preserving on EVERY corpus: AMI (13.101%), VoxConverse (4.847%, beats fork), Karpathy
(8.219%). `results/speakrs_karpathy_patched_der.json`.

### 4.18 fp16 engines — deferred (TRT 11 strongly-typed)

TRT 11 (26.06 image) removed `--fp16`: networks are **strongly typed by default** — engine
precision follows the ONNX graph's tensor types. fp16 therefore requires converting the ONNX
graph itself (onnxconverter-common float16 pass) with the StatsPool variance/sqrt subgraph kept
fp32 (the fork measured 26-33% DER collapse from fp16 there). That graph surgery + mandatory DER
re-validation = Milestone-1 work if speakrs is adopted; NOT a spike. fp32(+TF32 default) numbers
in §4.4 are the honest serving baseline.

## 5. Bugs/quirks found (upstream-reportable)

1. **speakrs teardown crash:** `corrupted double-linked list` (glibc) at process EXIT in `cuda`
   mode, after results flush — reproduced with both mimalloc (xtask) and glibc malloc (diar-cli),
   so it's the ORT-CUDA teardown path, not the allocator. (A suspected mid-run occurrence during
   C-M1f was traced to a shell-quoting bug in the benchmark harness invocation, NOT the crash —
   de-escalated after verification; multi-file single-process runs are stable.) diar-server
   still gets a supervisor/auto-restart as standard hardening. Workaround: validate output
   content, not exit codes.
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

## 7. Phase C-T1 — the flip into OpenTranscribe (app-level)

Deployment under test: live OpenTranscribe stack (`transcribe-app`, compose files
`docker-compose.yml + override + gpu + nas`), worker and sidecar both on **GPU 1 (RTX 3080 Ti,
12 GB)** — deliberately the same device the PyAnnote path used, so python-vs-native is an
engine-only A/B. `ENABLE_BENCHMARK_TIMING=true`, `ENABLE_VRAM_PROFILING=true` were already live.
App branch: `feat/native-diarization-flip` (3 commits, never pushed).

Controls held constant for every comparison below: ASR `large-v3-turbo`, `compute_type=int8_float16`,
`batch_size=8`, `beam_size=5`, VAD default, one file at a time, same audio, same reference.

### 7.1 Historical production baseline (passive, python engine, pre-flip)

`file_pipeline_timing` had 85 rows (71 with `user_perceived_duration_ms`) accumulated 2026-07-03
→ 2026-08-19. Two regimes, and they must not be pooled:

| date | n | avg audio | concurrency | avg upload→presented | avg × RT |
|---|---|---|---|---|---|
| 2026-07-03 (bulk load) | 25 | 9601 s | 9.7 files | 1473.0 s | 10.7× |
| 2026-08-05 → 08-19 (single-file) | 46 | 358-3698 s | 1.0 | 14.7-99.1 s | 18.3-37.5× |

Single-file medians by duration: <10 min → **15.3 s** presented (12.7 s GPU stage, 1.1 s
preprocess, 1.3 s postprocess); 1-2 h → **100.6 s** (93.3 s GPU); >2 h (the loaded bulk regime)
→ 1369.1 s. This is baseline source (a) — real media mix, no controlled repetition. Source (b)
is the controlled corpus in §7.5.

### 7.2 Flip verified end to end — PASSED

`docker-compose.diar-native.yml` sidecar up on GPU 1 (fast set, `models_folded/`), `/healthz` ok
from the worker, `DIARIZER_ENGINE=native` on `celery-worker`. Worker preload logs
`diar-native sidecar ready` and loads **no PyAnnote pipeline at all**.

A real upload through the API (`POST /api/files`, 4 min video, id 177598) produced: 22 segments,
**4 speakers**, 256-d **L2-normalized** embeddings in OpenSearch `speakers_v4` (norm = 1.000000
verified), and one speaker **auto-matched to an existing profile** — i.e. the native centroids
flow correctly through the whole speaker-matching chain. Native diarization: **2.1 s** for
238.8 s of audio (≈114× RT) with **Δ0 MB** of worker VRAM (it is the sidecar's memory now).
Warm PyAnnote on the same file and GPU: 3.57 s.

### 7.3 Fallback + recovery drill — PASSED (after fixing a one-way fallback)

Three-phase drill on the same file, engine confirmed from worker logs each time:

| phase | sidecar | engine that served | diarization |
|---|---|---|---|
| A | up | native | 3.0 s |
| B | `docker stop` | PyAnnote fallback | 10.3 s |
| C | restarted | native again | 2.9 s |

Phase C only passes because of a bug found here: the fallback was **one-way**. A cached
in-process engine is always "reachable", so a worker that ever fell back stayed on PyAnnote
until it restarted (verified: post-restart job served by cached PyAnnote in 3.57 s). Fixed by
probing the sidecar per task and rebuilding whenever the cached engine disagrees with what can
actually serve, in both directions, unloading the released engine first so returning to the
sidecar also frees the PyAnnote VRAM.

### 7.4 ACCURACY GATE — FAILED AS WRITTEN; cause isolated to exclusive-segment extraction

Harness: `backend/scripts/benchmark_boundary.py` on the full 66.5-min Karpathy acceptance clip
(`benchmark/diarization-boundary/karpathy/`), reference = the maintainer's hand labels
(`reference.rttm`, midpoint-mapped — ASR-model independent), both engines scored over an
**identical 14 978-word inventory** (14 781 scored, 197 excluded), 2 speakers found by both.

| engine | WSER smoothing OFF | WSER smoothing ON | word errors ON | islands ON |
|---|---|---|---|---|
| python (PyAnnote fork) | 1.231% | **0.859%** | 127 | 21 |
| native (speakrs) | 1.942% | **1.312%** | 194 | 7 |

Neither engine reaches the written `≤ 0.27% smoothed` bar **on this clip**, because the bar was
never about this clip — see §7.9, which traces it to the first **10 minutes** and reproduces it
exactly on both engines. On the full 66.5-min clip the gate that binds is native-vs-fork parity,
and native fails it here by **+0.45 pp WSER** (fixed in §7.7).

Attribution — four scorings against the same reference, same audio:

| representation | segments | DER c0.25 | DER c0 |
|---|---|---|---|
| fork **full** | 697 | **8.194%** | 10.510% |
| speakrs **full** | 692 | **8.219%** | 10.532% |
| fork **exclusive** | 774 | **6.161%** | 8.159% |
| speakrs **exclusive** | 660 | **6.545%** | 8.658% |

Both full-diarization numbers **reproduce their recorded validation values exactly** (§2.3
fork 8.194%, §4.21 speakrs 8.219%) — the engines remain at +0.025 pp parity, and the sidecar
reproduces diar-cli's segment counts exactly (692 full / 660 exclusive, §4.26). The app
integration is also faithful: the float32→int16 WAV round-trip in `diarizer_native.diarize`
costs **0.001 pp** (app-path 6.546% vs sidecar-direct 6.545%).

**The entire app-level accuracy gap is created by exclusive-segment extraction**, which is what
the app consumes for word attribution. Root cause and fix: §7.7.

An early hypothesis here — "speakrs discards overlapped speech" — was **wrong, and is retracted**:
the union of speech time is identical between speakrs' full and exclusive outputs (0.00% across
all 16 AMI meetings, §7.7). A naive trim-the-overlaps reimplementation was also tested and is
**worse** (7.863% c0.25). The defect is *which* speaker wins a contested frame, not whether the
frame survives.

Scope: ~0.45% of words on a 2-speaker clip with little overlap; it grows with overlap density.
Note the corpus-level picture stayed favorable to speakrs throughout (VoxConverse 216-file
arbiter: 4.847% vs fork 5.099%, §4.16d) — this was a representation defect, not a clustering one.

### 7.6 Integration defects found by T1 (both fixed)

1. **Handoff volume was root-owned** — the sidecar image had no `/tmp/diar-native`, so the named
   volume was created `root:root 0755` while every worker in this deployment runs as `appuser`
   (uid 1000). The first real job died with `PermissionError: /tmp/diar-native/diar_*.wav`.
   Fixed durably in `docker/Dockerfile.server` (`mkdir -p /tmp/diar-native && chmod 1777`), so a
   fresh named volume inherits the mode — verified on a throwaway volume.
2. **One-way fallback** — see §7.3.

Pre-existing deployment wart, noted not fixed: `/tmp/transcription` and `/scratch/opentranscribe`
are root-owned 0755 in this stack, so `write_wav_to_shared_volume` silently warns and skips.
**This matters for T2**: S-T2 assumes preprocess already wrote the 16 kHz WAV to the shared
volume so the sidecar can read it with zero handoff cost — in this deployment that write is
currently failing.

### 7.7 ROOT CAUSE + FIX — exclusive overlap resolution was decided by cluster index

**What exclusive means, and why its DER is higher than full for BOTH engines.** The app must
give every word exactly one speaker, so it consumes a non-overlapping timeline. AMI is meeting
audio and its hand labels mark both speakers during overlap, so any exclusive hypothesis is
scored as missing the speaker it did not name. That is arithmetic, not a defect: `missed` rises
7.78% → 14.39% for the fork and 7.77% → 14.38% for speakrs — **identically**. False alarm and
confusion both fall, because an exclusive timeline stops asserting a second speaker.

**Two measurements killed the "speakrs drops overlapped speech" hypothesis.** (1) The union of
speech time is bit-identical between speakrs' full and exclusive outputs — 26073.1 s vs
26073.1 s, 0.00% on every one of the 16 meetings. (2) The DER components show the fork/speakrs
exclusive gap is *entirely* confusion (2.655% vs 1.808%), with missed detection identical to
within 0.012 pp. Nothing is lost; the wrong speaker is named.

**Root cause.** `vendor/speakrs/src/reconstruct.rs:170` `make_exclusive` documents itself as
"zero out all but the highest-scoring speaker in each frame". But `Reconstructor::reconstruct*`
writes **1.0** for every active speaker (reconstruct.rs:142/161), and `post_inference` may
additionally `binarize()` — so by the time `make_exclusive` runs, every active speaker in a
frame holds exactly 1.0. Every overlapped frame is an N-way tie, and Rust's `Iterator::max_by`
returns the **last** maximum on ties, so the winner is the highest cluster index. Empirically
confirmed: over **22 297 sampled overlap frames across AMI-16, the surviving speaker was the
highest-indexed one 100.0% of the time.** Overlap ownership was decided by cluster numbering,
never by acoustics. The continuous evidence exists — `Reconstructor::frame_activations` — and
was being discarded one step before it was needed.

**Fix** (vendored speakrs + diar-core wiring):
1. `reconstruct.rs`: `reconstruct`/`reconstruct_smoothed` split into `*_with(activations, ...)`
   variants so one activation pass feeds both reconstructions; new `exclusive_from(full,
   activations)` keeps, per frame, the active speaker with the highest **activation score**.
   Invariants by construction: a frame with ≥1 active speaker keeps exactly one, a frame with
   none stays empty — speech is never invented or lost.
2. `post_inference.rs`: builds `exclusive_diarization` alongside the full one and puts it
   through the **same** `binarize` duration filter and `merge_segments(merge_gap)` the full path
   uses (previously the exclusive path in diar-core skipped gap merging entirely).
3. `DiarizationResult` gains `exclusive_segments`; `diar-core` consumes it instead of calling
   `make_exclusive` on the binarized array.

**Results — AMI test-16** (collar 0.25, UEM, overlap included):

| variant | DER | missed | false alarm | confusion |
|---|---|---|---|---|
| fork full | **13.093%** | 7.780 | 2.351 | 2.962 |
| speakrs full | 13.102% | 7.766 | 2.344 | 2.991 |
| fork exclusive (target) | **17.828%** | 14.387 | 1.632 | **1.808** |
| speakrs exclusive (before) | 18.654% | 14.375 | 1.625 | **2.655** |
| **speakrs exclusive (fixed)** | **17.813%** | 14.375 | 1.624 | **1.814** |

`fork full` reproduces the recorded §2.2 baseline (13.093%) exactly, and `speakrs full` the
recorded §4.26 value (13.101%) — both engines are measured against their own known-good
numbers. **The fixed exclusive beats the fork's (17.813% vs 17.828%) and collapses confusion
2.655% → 1.814%.** The full diarization is **bit-identical on 16/16 files** after the change —
the fix touches only the exclusive path.

**Results — Karpathy 66.5 min, WSER through the app pipeline** (large-v3-turbo, identical
14 978-word inventory):

| engine | WSER off | WSER on | word errors on | DER c0 on |
|---|---|---|---|---|
| fork | 1.231% | **0.859%** | 127 | 0.047628 |
| speakrs (before) | 1.942% | 1.312% | 194 | 0.050906 |
| **speakrs (fixed)** | 1.270% | **0.890%** | 132 | 0.047986 |

App-level parity: native was +0.45 pp worse than the fork, now **+0.031 pp** — 5 words out of
14 781. The written `≤ 0.27%` bar is still not reachable by either engine under this reference
method and remains a documentation defect to re-derive, not an engine one.

Upstream value: this is a correctness bug in speakrs' `exclusive_speaker_diarization`
equivalent, independent of our deployment — a PR candidate alongside the perf patches.

### 7.5 E2E SPEED BASELINE — before/after upload→presented, per file (COMPLETE)

Protocol: `validation/run_e2e_baseline.sh` per engine configuration, 5 files × 3 runs each,
driven by `transcribe-app/scripts/benchmark_e2e.py` (reprocess → Redis markers → CSV). **Timed
legs strictly sequential** (§4.11) — one configuration at a time, quiet machine, nothing else on
the GPUs. Raw CSVs: `results/e2e_baseline/{python,native_fast,native_small}/`.

Controls held constant across all three legs (the only variable is the diarization engine):
ASR `large-v3-turbo`, `compute_type=int8_float16`, `batch_size=8`, `beam_size=5`, GPU 1 (RTX
3080 Ti) for worker **and** sidecar, one file at a time, same 5 files, same app build. The
python leg ran with the sidecar stopped (true pre-flip shape); native legs with it up.
Configurations: `python` = PyAnnote community-1 in-process; `native_fast` = sidecar with
`models_folded/`; `native_small` = sidecar with `models_small/`.

**Headline — upload → presented (median of 3, `total_dispatch_to_postprocess`):**

| file | audio | python (before) | native fast (after) | speedup | RT before → after |
|---|---|---|---|---|---|
| test_ai_video | 24 s | 5.7 s | **5.2 s** | 1.10× | 4× → 5× |
| pyramids | 239 s | 11.9 s | **9.1 s** | 1.31× | 20× → 26× |
| warp drive | 358 s | 15.0 s | **11.8 s** | 1.27× | 24× → 30× |
| Karpathy | 3989 s | 108.4 s | **80.5 s** | 1.35× | 37× → 50× |
| seed file | 7558 s | 206.7 s | **147.2 s** | **1.40×** | 37× → **51×** |

`fully_indexed_duration` tracks it (2.1 h: 210.3 s → 152.9 s). The speedup grows with duration
because the fixed per-job overhead (~4.5 s: dispatch, ffmpeg, model warm, postprocess) dominates
short files — on the 24 s clip almost nothing is left to win.

**Where the time went — GPU-stage split (last run of each leg):**

| file | transcribe (py / fast) | diarize python | diarize native fast | diarize speedup |
|---|---|---|---|---|
| pyramids 239 s | 2.9 / 2.7 s | 3.6 s | **2.0 s** | 1.80× |
| warp drive 358 s | 3.7 / 3.6 s | 5.2 s | **2.9 s** | 1.79× |
| Karpathy 3989 s | 41.9 / 39.6 s | 59.1 s | **32.5 s** | 1.82× |
| seed 7558 s | 70.3 / 71.7 s | 116.6 s | **58.0 s** | **2.01×** |

Transcription is unchanged leg-to-leg (±2%), which is the control working: the entire E2E delta
is the diarization stage. **The bottleneck ordering has flipped as PLAN predicted** — on the
2.1 h file diarization was 116.6 s vs transcription 70.3 s (62% of GPU stage); it is now 58.0 s
vs 71.7 s (45%). Transcription is the #1 stage from here on.

**T2 headroom, quantified:** the two stages still run *sequentially* (130 s combined on the 2.1 h
file). L2 overlap makes the GPU stage ≈ max(71.7, 58.0) ≈ 72 s — a further **~45% off the GPU
stage** with no model change. That is now the single largest remaining lever.

**Small set is not a speed/VRAM trade on this hardware — it is a VRAM floor.**

| config | GPU 1 used (worker+sidecar) | 2.1 h diarize | 2.1 h upload→presented |
|---|---|---|---|
| python | 3809 MB | 116.6 s | 206.7 s |
| native fast | 7847 MB | **58.0 s** | **147.2 s** |
| native small | 5235 MB | 207.0 s | 295.5 s |

The small set is **3.6× slower than the fast set and 1.8× slower than PyAnnote**, for 2.6 GB
saved. It earns its place only where the fast set does not fit (laptop-class GPUs, MODELS_SETS.md
tier); on any card with ~5 GB free for the sidecar, `models_folded/` is the correct choice.

Baseline source (a) cross-check: the passive production rows in §7.1 put single-file 1-2 h media
at ~100 s presented on the python engine, consistent with the 108.4 s measured here for a
66.5-min file under controlled conditions.

### 7.9 WSER GATE RE-DERIVED — the 0.27% bar is real, and native meets it exactly

The bar was traced to `transcribe-app/docs/diarization-boundary-results/cloud-comparison.md`:
it is the **first 10 minutes** of the Karpathy clip (2 257 words), not the 66.5-min acceptance
clip §7.4 measured (14 978 words). EXECUTION_TASKS/INSTALL_NATIVE inherited the number without
those qualifiers, which is what made it look unreachable. Re-measured on the documented clip
(`karpathy_10m.wav`, hand-labelled `reference.rttm` midpoint-mapped, large-v3-turbo, smoothing
ON, `benchmark_boundary.py`):

| engine | WSER off | WSER on | gate ≤ 0.27% |
|---|---|---|---|
| python (PyAnnote fork) | 1.15% | **0.27%** | PASS |
| native (speakrs, §7.7 fix) | 1.15% | **0.27%** | **PASS** |

Both reproduce the published 1.15% → 0.27% pair to the digit. The two engines are *not* running
the same diarization — their `diarize_records` differ (verified: `native != python`, while two
independent PyAnnote runs are byte-identical, confirming the fork is deterministic and the env
override really switches engines) — they simply land on the same word-level labels once the
boundary smoother has run on this clean 2-speaker clip.

**Correction to §7.4:** the claim that the bar "is not reproducible as stated" was wrong. It is
reproducible; it was under-specified. The gate now reads, in full: *Karpathy **10-minute** clip,
hand-labelled reference, midpoint-mapped, smoothing ON → WSER ≤ 0.27%.* The 66.5-min clip is a
harder, longer-context test and belongs in the plan as a **parity** gate (native within noise of
the fork: 0.890% vs 0.859% after the §7.7 fix), not an absolute-threshold one.

### 7.10 Deployment hardening — shared-volume ownership (permanent fix)

`/scratch/opentranscribe` and `/tmp/transcription` are docker **named volumes** (not host binds;
they surface under the NAS only because this host's docker data-root is `/mnt/nas/docker`, a
local ext4 RAID). A named volume inherits its ownership from the image directory at creation —
and neither backend image reserved those paths, so every volume was created `root:root 0755`
while the workers run as `appuser` (uid 1000). Consequence: `write_wav_to_shared_volume` failed
with EACCES on every job and silently degraded to a re-decode in stage 2 — **and S-T2's premise
(the sidecar reads the WAV preprocess already wrote) was broken before T2 even started.**

Fix, in three parts so it holds on any machine:
1. `backend/Dockerfile.prod` + `Dockerfile.lite` reserve both paths owned by appuser, placed in a
   **late layer** so the change does not invalidate the expensive build stages. Verified on the
   rebuilt image: a freshly created named volume now reports `1000:999 755`.
2. `scripts/fix-shared-volume-perms.sh` — idempotent repair for deployments whose volumes predate
   the image change (an image fix cannot retro-repair an existing volume). Applied here:
   `0:0 755 → 1000:1000 755` on both volumes; worker write probe passes on both.
3. `audio_loader.write_wav_to_shared_volume` now catches `PermissionError` separately and names
   the repair script instead of logging a bare EACCES.

### 7.11 PRODUCTION BUG FOUND — gender detection can wedge the whole CPU worker pool

Surfaced while running the T2 E2E leg. The pipeline stalled indefinitely: the GPU task for
`warpdrive_358s` succeeded, but the file never left `processing` and the next iteration's
`transcription.preprocess` was never scheduled.

`celery inspect active` on `celery-cpu-worker` (`--concurrency=8`): **all 8 slots held by
`detect_speaker_attributes`**, oldest ~2.5 h old, `priority: 2`, `acknowledged: True`. With
every slot taken, `preprocess` — which shares the `cpu` queue — can never run. This is a
**liveness bug**, not the priority competition L5a describes: gender detection can deadlock the
entire ingest pipeline.

Established: 2 `ForkPoolWorker` children exited with `signal 9 (SIGKILL)`; the container is not
`OOMKilled` and the host had 445 GB available, so host memory pressure is ruled out. **Not**
established: whether the remaining slots are dead-but-leaked bookkeeping or live-and-stuck
children — the `/proc` probe was inconclusive and is not claimed either way.

Prime suspect (`speaker_attribute_task.py:125-134`): `fut.result(timeout=30)` bounds only the
*wait*, not the work, and `with ThreadPoolExecutor(...)` calls `shutdown(wait=True)` on exit —
so a segment fetch that never returns makes the task hang forever regardless of the timeout.
Every reprocess enqueues another of these, which is why a benchmark loop reproduced it quickly
where normal usage would take much longer to accumulate 8.

Bearing on the plan: this raises T5a from a latency tweak to a correctness fix, and strengthens
T5b (moving gender into the sidecar deletes the presigned-URL fetch pool that is the suspected
hang). Any fix must also cap how many CPU slots this task can occupy, so it can never take the
whole pool again.

Benchmark hygiene: `speaker_attribute.detection_enabled=false` was set in `system_settings` to
keep the wedge from recurring mid-measurement. **It must be set back to true** once the T2 legs
are recorded — the setting is a benchmark accommodation, not a product decision.

### 7.12 VRAM BASELINE for the post-flip workflow (and what it means for parallelism)

Steady-state, all models warm, GPU 1 (RTX 3080 Ti, 12 288 MiB), measured by
`nvidia-smi --query-compute-apps` with PIDs mapped back to containers:

| consumer | VRAM | share | per-job or fixed? |
|---|---|---|---|
| **diar-native sidecar** (fast set) | **4 136 MiB** | 34% | **fixed** — one copy serves every job |
| celery-worker (whisper large-v3-turbo, int8_float16, batch 8) | 2 038 MiB | 17% | **per worker process** |
| celery-redaction (toxic-bert / xlm-roberta) | 1 346 MiB | 11% | fixed, resident even when idle |
| free | 4 328 MiB | 35% | |

Gender detection (wav2vec2) runs **on CPU** — 0 VRAM, ~380 MB RAM, 87-90 s per file. The two
A6000s (49 GB each) are **completely idle**: 15 MiB apiece.

Three things follow, and they change how parallel throughput should be approached:

1. **Diarization is now the largest GPU consumer — 2× the ASR model** — but it is a *fixed*
   cost, not a per-job one. **Marginal cost per concurrent job is ~490 MiB** (measured §7.14),
   not the ~2 GB first estimated here: `celery-worker` runs `--pool=threads`, so all 8 threads
   share one ModelManager and one copy of the whisper weights — only activations scale.
2. **The advertised concurrency does not match the memory.** `celery-worker` runs
   `--pool=threads --concurrency=8` with `GPU_CONCURRENT_REQUESTS=8`, but 4.3 GB of headroom
   only funds ~2 more concurrent whispers (`GPU_DEFAULT_CONCURRENCY=2` is the honest number).
   `DIAR_MAX_INFLIGHT=2` is likewise notional while diar-server serializes on
   `Mutex<DiarEngine>` (T9a).
3. **The cheapest capacity wins are placement, not shrinking models.** Moving the redaction
   worker off GPU 1 (`REDACTION_GPU_DEVICE_ID`, config-only) returns 1.3 GB; moving the sidecar
   to an idle A6000 (`DIAR_NATIVE_GPU`, config-only) returns 4.1 GB. Doing both leaves GPU 1
   almost entirely to whisper — roughly 5 concurrent jobs instead of 2 — with **zero** model
   or code changes. The small model set is the wrong lever here: it saves 2.6 GB but costs
   3.6× diarization speed (§7.5).

What still gates real parallel throughput is T9a: even with VRAM freed, one diar-server
serializes requests, so N concurrent app jobs queue at the sidecar. Arc-shared sessions turn
that fixed 4.1 GB into a genuinely shared engine (`1 engine + N × scratch`).

Caveat: these are steady-state resident figures. The VRAM profiler recorded no meaningful
per-step growth during the T2 legs (device_used held at ~7.9 GB across pipeline_start →
after_diarization), so resident ≈ peak for this workload — but that was measured at
concurrency 1, and peaks under genuine multi-job load are unmeasured.

### 7.13 T2 GATE PASSED — transcribe ∥ diarize overlap

Change: `_AsyncDiarization` starts diarization as soon as the audio is in memory and collects it
after transcription, in both `_GpuRawStage` (the production path) and `_GpuStage`. Gated on
`DIARIZER_ENGINE=native`; `DIAR_OVERLAP=0` restores the sequential order; a failed overlapped
attempt falls back to a plain inline run.

**Output identity — the gate.** Both stages read the same numpy buffer, so identity was proved,
not assumed. Same clip, overlap ON vs OFF: `diarize_records` (766), `overlap_info`, speaker
embeddings and whisper `raw_segments` all **byte-identical**. Nothing mutates the shared audio.

**Max-not-sum.** 66.5-min clip: transcription 50.3 s with 37.5 s of diarization running inside
it; the join cost **0.005 s**. Sequential the same work is 87.8 s.

**E2E, 5 files × 3 runs, sequential legs, quiet machine** (median upload→presented):

| file | audio | python | native | native+overlap | total speedup | RT before → after |
|---|---|---|---|---|---|---|
| test_ai_video | 24 s | 5.7 s | 5.2 s | **4.2 s** | 1.36× | 4× → 6× |
| pyramids | 239 s | 11.9 s | 9.1 s | **7.2 s** | 1.65× | 20× → 33× |
| warp drive | 358 s | 15.0 s | 11.8 s | **8.6 s** | 1.74× | 24× → 42× |
| Karpathy | 3989 s | 108.4 s | 80.5 s | **54.4 s** | **1.99×** | 37× → **73×** |
| seed file | 7558 s | 206.7 s | 147.2 s | **120.3 s** | 1.72× | 37× → 63× |

Cumulative effect of the flip plus overlap: **upload→presented is 1.7-2.0× faster** on media
over a minute, and the short-file floor (~4 s of fixed per-job overhead) is now what limits the
small end.

**Measurement incident (logged per §4.11).** A first attempt at this leg was **discarded**: the
run that had stalled behind §7.11's wedge resumed when the CPU worker was restarted and ran
concurrently with its own replacement, so two timed legs overlapped. The numbers above are from
a clean re-run. `run_e2e_baseline.sh` now takes an `flock`, so a leg refuses to start while
another holds the lock.

### 7.14 Concurrency + VRAM behaviour under load (answering "will we run out?")

Measured on GPU 1 by sampling `nvidia-smi` **during** three concurrent reprocess jobs (a first
attempt sampled after the jobs had already finished and was discarded as meaningless):

| state | GPU 1 used |
|---|---|
| idle floor | 7 575 MiB |
| peak, 3 concurrent jobs | 9 047 MiB |
| **marginal per concurrent job** | **~490 MiB** |
| settled afterwards | 7 575 MiB — **exactly** the idle floor |

**Not a leak.** It returns to the same figure after every run, so VRAM does not creep upward
over time. What is held is a floor, and headroom above it is genuinely reusable: 4 328 MiB free
÷ ~490 MiB ≈ **8 concurrent jobs**, which is consistent with `GPU_CONCURRENT_REQUESTS=8`.

**Correction to §7.12:** the marginal cost is ~490 MiB, not the ~2 GB stated there. The GPU
worker runs `--pool=threads --concurrency=8`, so all eight threads share one ModelManager and
one copy of the whisper weights — only per-request activations scale with concurrency.

**Why the floor is 7.5 GB in the first place** — three deliberate warm-start decisions, and one
of them is earned rather than allocated:

| moment | GPU 1 total | change |
|---|---|---|
| app only, no sidecar | 2 838 MiB | whisper preloaded and pinned at worker startup |
| sidecar started | 3 385 MiB | +547 MiB — ONNX **weights only** |
| after a 30 s clip | 4 069 MiB | arena begins growing |
| after a 10 min clip | 6 979 MiB | arena sized to batch-32 activations |
| steady since | 7 575 MiB | high-water mark, never returned |

So the sidecar's 4 136 MiB is ~547 MiB of weights plus ~3.6 GB of ORT BFC arena and cuDNN conv
workspace, acquired on first real inference and kept. `arena_extend_strategy` is **already**
`SameAsRequested` (the lean setting), so the remaining levers are `with_conv_max_workspace(false)`
with a cheaper `ConvAlgorithmSearch`, ORT's per-run `memory.enable_memory_arena_shrinkage` (the
closest analogue to `torch.cuda.empty_cache()`), and a shared cross-session allocator (§4.25).
Each trades against speed and must be benchmarked, not assumed.

### 7.15 FULL TASK CENSUS — every task on one file, and where the remaining work is

`validation/task_census.sh` reprocesses one file, waits for user-visible completion, lets the
enrichment tail drain, then reads every Celery task's duration out of the worker logs plus the
authoritative stage split from `file_pipeline_timing`. Reference file: Karpathy 66.5 min.

**Harness note (a real trap):** the first version polled `celery inspect active` across six
workers in a loop. Each call costs seconds per worker, so the harness dominated the number it
was reporting — minutes of "measuring" a 54 s job, with CPU and GPU sitting idle waiting on the
script. Rewritten to poll only the cheap file-status endpoint and read durations after the fact;
census wall time went from ~7 min to 1 min 19 s for the same job.

| task | time | device | share of task time |
|---|---|---|---|
| **transcription.gpu_transcribe** | **52.91 s** | GPU | **80.0%** |
| redaction.detect | 5.75 s | **GPU** | 8.7% |
| index_transcript_search | 2.46 s | CPU | 3.7% |
| speaker.cluster_for_file | 1.60 s | CPU | 2.4% |
| transcription.preprocess | 1.30 s | CPU (ffmpeg) | 2.0% |
| transcription.postprocess | 0.53 s | CPU | 0.8% |
| 10 further tasks | < 0.5 s each | mixed | ~2% |
| **sum of task time** | **66.15 s** | 16 tasks | |
| user-visible (stage table) | **54.8 s** | prep 1.3 + gpu 52.9 + post 0.5 | |

Sum exceeds wall time because workers run in parallel. **`detect_speaker_attributes` does not
appear**: its idempotency guard skips files that already carry attributes. On a *new* file it
costs **87-90 s** (§7.11), which would make gender detection the **second-largest task in the
entire pipeline** — larger than everything except transcription, and it runs on one CPU core.

**Where the remaining throughput lives.** Transcription is 80% of task time and is already at
int8_float16 on a batched pipeline, so the addressable remainder is the other 20% plus the
gender tail:

| target | cost today | opportunity | tracked as |
|---|---|---|---|
| gender detection | 87-90 s CPU, new files only | wav2vec2 → ONNX in the sidecar, reusing the PCM already held for diarization; deletes the presigned-URL fetch pool | T5b (#18) |
| redaction.detect | 5.75 s GPU | toxic-bert/xlm-roberta → optimum-ORT; also holds 1 346 MiB resident for a task that runs seconds | T13 (#21) |
| index_transcript_search | 2.46 s | embedding model → ONNX | T13 (#21) |
| speaker.cluster_for_file | 1.60 s | ANN index in Rust | T12 (#21) |
| preprocess ffmpeg decode | 1.30 s | symphonia decode inside diar-server; sidecar takes the original media path and no WAV is written at all | SPEEDUP_ROADMAP §Preprocessing |

The shared-memory theme the user raised applies twice over: gender detection re-fetches audio
over presigned URLs that diarization already had in memory, and preprocess writes a WAV that
exists only so a second process can read it back. Both disappear if the sidecar owns decode and
holds the PCM for the tasks that need windows of it.

### 7.16 Gender detection moved into the sidecar (T5b) — and two things it taught

Motivation from §7.15: `detect_speaker_attributes` was the second-largest task in the pipeline
at **87-90 s** on one CPU core, re-fetching clips over presigned URLs that diarization had
already decoded. It now rides the `/diarize` call behind a `gender` flag, classifying windows of
the PCM already in hand.

| | before (app, CPU) | after (sidecar, GPU) |
|---|---|---|
| wall time | 87-90 s | **~1.5 s** |
| audio fetch | presigned URL + ffmpeg per clip | slice of the existing buffer |
| when it runs | after user-visible completion | inside transcription's window (free) |

**Parity gate: PASSED.** ONNX vs PyTorch across 20 clips of varying length: max |logit diff|
**5.96e-06** (bar 1e-4), **zero** label mismatches.

**Window length was a self-inflicted VRAM bug.** Speaker turns run to a minute, and wav2vec2
activations scale with input length, so passing whole turns meant 60 s clips and **6 340 MiB**
of VRAM for no accuracy gain. Windows are now centre-cropped (the middle of a turn is the
cleanest voice). Cap swept on the reference clip:

| cap | VRAM (container) | wall, 10-min clip | verdicts |
|---|---|---|---|
| 3 s | 4 804 MiB | 5.88 s | female 0.796 / male 0.999 |
| **5 s (default)** | **4 804 MiB** | 5.96 s | female 0.797 / male 0.999 |
| 10 s | 5 316 MiB | 6.16 s | female 0.789 / male 0.999 |
| (uncapped, 60 s) | 6 340 MiB | — | same decisions |

**Identical decisions at every cap**, so the choice is pure cost: 5 s costs the same memory as
3 s and sees more voice. Marginal VRAM over diarization alone is **~670 MiB**, tunable with
`DIAR_GENDER_MAX_SECONDS` for the laptop tier. The app never hit this because CPU inference on
an oversized clip is merely slow, not fatal.

### 7.17 Whisper batch size — no win available, and why (INVALID SWEEP RETRACTED)

A sweep of `BATCH_SIZE` ∈ {8,16,24,32} produced 55-61 s across all four settings. **It was
invalid**: the compose variable is `GPU_DEFAULT_BATCH_SIZE` (mapped to `BATCH_SIZE` inside the
container), so setting `BATCH_SIZE` was overwritten with `auto` and every run was batch=8. The
spread was run-to-run noise. Retracted rather than reported.

The answer was already in the codebase, and it matches the user's instinct that the largest
batch is not the most efficient — `hardware_detection.py:118`, from the Phase B VRAM sweep
(raw data `transcribe-app/docs/whisper-vram-profile/`):

> Plateau points (RTF stops improving above these): **large-v3-turbo: batch=8** (RTF 0.009 from
> bs=8 onward)

So batch=8 is correct on speed grounds and raising it only spends VRAM. Two notes for later:
the 3080 Ti reports 11 902 MB usable, landing *just* under the rule's `>= 12 GB → 24` threshold;
and that rule's budget still assumes "CUDA context + PyAnnote diarization models pre-loaded in
the celery worker", which the flip made obsolete — roughly 2 GB more headroom than the rule
believes. Neither changes the conclusion, because the plateau binds before VRAM does.

### 7.18 fp16 gender model — ADOPTED (67/67 verdicts identical on AMI-16)

Converted with `onnxconverter_common.float16(keep_io_types=True)`, so the graph runs in fp16
while still accepting and returning fp32 — the Rust caller is unchanged. Compared against fp32
across **all 16 AMI meetings, 67 speaker verdicts**:

| | fp32 | fp16 |
|---|---|---|
| labels | — | **67/67 identical** |
| confidence | — | max Δ 0.0118, **mean Δ 0.00019** |
| container VRAM (AMI run) | 5 396 MiB | **4 890 MiB** (−506) |
| model on disk | 361 MB | **181 MB** (−50%) |
| wall, 10-min clip | 6.02 s | 5.90 s (within noise) |

Adopted as the shipped model. Speed is unchanged because gender is ~1.5 s of a 6 s call — the
win is memory and footprint, which is what the laptop tier needs. Note this is the *opposite*
outcome to RESULTS §4.18, where fp16 was rejected for the diarization graph: there the
StatsPool variance/sqrt subgraph collapsed DER. Different graph, different verdict — which is
why it was measured rather than assumed either way.

### 7.19 VAD: already running, and already ONNX — the gap is sharing, not speed

Checked because "precompute_vad is dead" reads as "no VAD is running". Both are true and they
are different features:

| | status | evidence |
|---|---|---|
| faster-whisper's internal Silero | **running on every job** | `vad_filter=True` hardcoded, unconditional, `transcriber.py:243`; `silero_vad_v6.onnx` ships in the faster-whisper assets |
| the app's `precompute_vad` | **dead** | `vad_regions=None` hardcoded, `stages.py:429`; config defaults false |

So transcription *is* skipping silence, via an ONNX Silero model, tuned by
`vad_{threshold,min_silence_ms,min_speech_ms,speech_pad_ms}` — which is what T8's sweep tunes.

**This reframes T7.** The opportunity is not to enable VAD, nor to port it to Rust: it is
already an ONNX model, so a rewrite would have to beat it. The value is running it **once,
upstream, and sharing it** — today whisper computes VAD for itself while diarization runs its
own segmentation over the same audio, and neither sees the other's work. That is a
shared-computation win, not a kernel-speed one, and it should be scoped that way.

### 7.20 Gender window size — swept across AMI-16, 5 s confirmed

Asked because the sidecar reported higher confidence than the CPU path on the same file
(0.989 vs 0.593 for one speaker) — which is window selection, not fp16 (§7.18 showed fp16 and
fp32 agree 67/67 with a mean confidence delta of 0.0002). Swept across all 16 AMI meetings,
67 speaker verdicts each:

| cap | VRAM | mean confidence | labels vs 5 s |
|---|---|---|---|
| 3 s | 4 634 MiB | 0.9425 | **67/67 identical** |
| **5 s (default)** | 4 890 MiB | 0.9481 | — |
| 10 s | 4 892 MiB | 0.9497 | 66/67 (1 differs) |
| 20 s | **13 633 MiB** | 0.9527 | 66/67 (1 differs) |

**Larger is not better here.** 20 s costs 13.6 GB — it would not fit on the 12 GB target at all
(this ran on a 49 GB A6000), and it changes one verdict in 67. 10 s costs the same memory as
5 s for +0.0016 mean confidence and the same single flip. Decisions are effectively flat from
3 s upward, so the cap is a pure cost choice and 5 s stays.

Honest caveat: confidence rising with window length says the model is more *certain*, not more
*correct*, and AMI carries no speaker-gender ground truth — so the one verdict that differs
between 5 s and 10/20 s cannot be adjudicated here. If gender accuracy ever needs a real gate it
needs a labelled set, not a confidence comparison.

### 7.21 T6 telemetry — the tail markers were never missing, the flush was too early

E2E_PIPELINE_MAP recorded `summary_*`, `clustering_*`, `search_index_*` as "never written".
Measured: **six columns null on every row** (search_index, clustering, summary, redaction,
speaker_upsert, waveform — 0 of 23 rows) while `gpu_end_ms` was 23/23, and
`fully_indexed_duration_ms` was *exactly* equal to `user_perceived_duration_ms` on every row,
making the work after the user sees the transcript look free.

The diagnosis differs from the plan's. Reading the Redis hash for a completed task showed
`redaction_start`, `redaction_end`, `search_index_chunks_start/end` **present**, alongside the
new `transcript_ready`, `diarize_request_sent` and `diarize_joined` — while the DB row held
NULL for all of them. The markers were being emitted all along; `_persist_timing_row` runs at
the end of `postprocess`, *before* the enrichment fan-out, so every tail marker arrived after
the only flush.

Fix: a delayed re-flush (`pipeline_timing.flush_tail`, 180 s). `record_pipeline_timing` already
upserts and merges, so the second write fills the late columns without disturbing the first,
and an early flush records less rather than corrupting anything. Plus columns for the three
markers T2/T3 introduced (migration `v393`) — without them the overlap claim is unfalsifiable
from telemetry.

Verified on a real job (358 s file):

| metric | before | after |
|---|---|---|
| user-visible | 11.7 s | 11.7 s |
| transcript_ready | not recorded | **8.9 s — 2.8 s before completion** |
| diarize span | not recorded | **5.0 s, inside transcription** |
| redaction | NULL | 1.1 s |
| search index | NULL | 0.7 s |
| fully_indexed | = user-visible | **12.6 s** |

### 7.22 T8 VAD silence sweep — no effect on the acceptance clip

`vad_min_silence_ms` ∈ {500, 1000, 2000} on the Karpathy 10-min clip, WSER harness:

| setting | WSER off | WSER on | words | word errors |
|---|---|---|---|---|
| 500 ms | 1.15% | 0.27% | 2257 | 6 |
| 1000 ms | 1.15% | 0.27% | 2257 | 6 |
| 2000 ms (default) | 1.15% | 0.27% | 2257 | 6 |

Identical to the digit — and confirmed as a real result, not a dead knob: the engine reports
`vad_min_silence_ms: 500` when the env is set, so the setting applied and the output did not
move. A dense two-speaker conversation has few silences in the 500-2000 ms band, so the
detected speech regions are the same at every threshold.

Conclusion: **accuracy-neutral here, and therefore no reason to change it on this evidence.**
The latency argument for a lower threshold only pays on silence-heavy media, which this
corpus does not contain — judging the knob needs such a file, not this one.

Gap found while testing: `VAD_MIN_SILENCE_MS` was read by the engine but passed by **no**
compose file, so the documented knob could not be set on any deployment. Now plumbed through
the three GPU worker services.

### 7.23 VRAM floor — conv workspace is not the lever, and 12 GB does not need it

Tested the two cuDNN knobs suggested in §7.12 by adding a `SPEAKRS_CONV_LEAN` switch
(`ConvAlgorithmSearch::Heuristic` + `with_conv_max_workspace(false)` instead of Exhaustive +
max) and running both against the reference clip:

| conv setting | VRAM | wall | RTTM |
|---|---|---|---|
| default (exhaustive, max workspace) | 4 526 MiB | 6.02 s | md5 99964a720ed5 |
| lean (heuristic, no max workspace) | **4 526 MiB** | 5.92 s | **md5 99964a720ed5** (identical) |

**No effect.** Same memory to the megabyte, same output, wall-clock within noise. So the ~3.6 GB
gap between 251 MB of weights and 4.1 GB resident is the ORT BFC arena sized to peak
activations, not cuDNN conv workspace. The change was reverted rather than kept: a vendored
diff that buys nothing only complicates the upstream patch set (T10).

**And the floor does not need shrinking for the shipping target.** On 12 GB: 7 575 MiB floor,
4 328 MiB free, ~490 MiB marginal per concurrent job (§7.14) — roughly eight concurrent jobs
fit. VRAM is not a constraint on this hardware; it constrains only the 4 GB laptop tier, where
the remaining levers are arena shrinkage on RunOptions and a shared cross-session allocator,
both untested. Relocating models to a second GPU was considered and rejected: real deployments
have one GPU, so moving a model elsewhere hides the problem rather than solving it.

### 7.24 T9a is justified — measured, not assumed (the sidecar mutex binds)

`diar-server` holds `engine: Mutex<DiarEngine>`, so `DIAR_MAX_INFLIGHT` bounds queueing but not
execution. Whether that matters was measured with the new `diarize_request_sent`/`diarize_joined`
columns (§7.21), running the two largest files concurrently:

| file | diarize span, solo | diarize span, 2 concurrent | GPU stage |
|---|---|---|---|
| Karpathy 66.5 min | 37.5 s | **76.1 s (2.0×)** | 52.9 s → 80.0 s |
| seed 2.1 h | ~58 s | **113.8 s (2.0×)** | — → 119.4 s |

Both spans **exactly doubled** — the signature of two jobs serialising on one lock, each waiting
out the other's diarization. The pair still finished in 131 s against 174.7 s sequential, so
concurrency helps overall, but roughly 11 s of that window is pure mutex wait, and the ceiling
gets worse with more concurrent files: diarization is 37-58 s of work per large file, so at
three or more the sidecar becomes the binding constraint rather than transcription.

**Scope of the fix, measured before starting.** The spec's option 1 (mutex per session) does
*not* deliver on this code shape: `DiarizationPipeline` borrows `&'a mut SegmentationModel` and
`&'a mut EmbeddingModel` for its whole lifetime, so a per-model lock is still held for the whole
job. Real concurrency needs the split decision #4 describes — immutable weights shared, mutable
scratch per request — and the structures are:

| | sessions (shareable) | mutable buffers (per-request) |
|---|---|---|
| `SegmentationModel` | 3 | 2 ndarray + 2 cached shapes |
| `EmbeddingModel` (`OrtEmbeddingState` + `EmbeddingBuffers`) | 10 | 12+ ndarray |

So ~13 ORT sessions to Arc-share and ~14 buffers to move into a per-request `Scratch`, threaded
through every inference call site in speakrs, behind DER-parity and determinism gates. That is
the "vendored-crate surgery" the plan reserved for a dedicated session, and it is the honest
reason not to start it as a tail-end change.

### 7.25 T9a LANDED — shared sessions; the sidecar no longer serialises (engine-level gates)

**What changed.** All 13 ORT sessions (3 segmentation + 10 embedding) became
`Arc<Mutex<Session>>` (`SharedSession`, `vendor/speakrs/src/inference.rs`), locked for exactly
one `run()` per inference call. The model structs themselves are now the per-request scratch:
`SegmentationModel::clone_shared()` / `EmbeddingModel::clone_shared()` Arc-clone the sessions
and re-allocate the ~14 staging buffers (~130 MB host RAM) plus a fresh
`primary_batch_run_options` (its preallocated output tensor is per-handle). diar-core gains
`DiarEngine::clone_shared()` (seg + emb + gender handles); diar-server keeps one prototype
engine and clones a handle per request — the engine mutex from §7.24 is gone. No speakrs
method signatures changed; the pipeline code is untouched. The spec's option 1 as written
could not work (per-model locks are held for the whole pipeline lifetime) — S-T9a corrected.

**Gates (engine level):**

| gate | bar | result |
|---|---|---|
| speakrs test suite | 94 tests | **94 pass** (74+5+8+7, container, openblas-system+online) |
| determinism | Karpathy ×3 byte-identical | **PASS** (one md5 across 3 runs) |
| AMI-16 full DER | 13.10 ± 0.01 | **13.101%**, 16/16 RTTMs content-identical to `results/rttm/diarcli_ami` |
| AMI-16 exclusive DER | 17.81 | **17.813%** exactly; exclusive RTTMs content-identical to `results/exclusive_study/exclusive_fixed` (8/8 sampled) |
| Karpathy full DER | 8.219 | **8.219%**, content-identical to `results/rttm/diarcli_karpathy` |
| Karpathy exclusive DER | (see below) | **6.188%** |
| concurrency correctness | N=4 concurrent ≡ serial | **PASS 4/4, three separate runs** (rttm, segments, exclusive, centroids all equal) |
| VRAM | 1 engine + N×scratch | **PASS**: 4 510 MiB peak DURING 4 concurrent jobs ≈ one warm engine; zero per-job VRAM growth |
| throughput | ≥ 2× serial, 4 short files, quiet machine | **1.51× — machine NOT quiet** (load 10-13, dsva-postprocess at 8+ cores, sibling session active); GPU util 46%→66% serial→concurrent. `SPEAKRS_FBANK_POOL=16` changed nothing → pool size is not the residual. RE-MEASURE QUIET before judging; harness `validation/t9a_concurrency.sh` |

Accuracy/determinism runs on GPU 2 (A6000, idle; same sm_86 as the 3080 Ti) because GPU 1 had
only ~4.2 GiB free beside the live stack; concurrency/VRAM legs ran on GPU 1 with a standalone
`diar-server:t9a`. Known ORT-CUDA teardown crash (§5) still fires at diar-cli exit, after
results flush — unchanged by this work.

**Correction to the handoff gate table:** `HANDOFF_T9A_SHARED_SESSIONS.md` §5 lists Karpathy
exclusive 6.545% — that is the **pre-§7.7-fix** number (§7.4, 660 segments). The post-fix
exclusive path produces 766 segments and scores **6.188%** (better than the fork's 6.161%
pre-fix protocol equivalent), and T9a's exclusive output is bit-identical to the recorded
post-fix artifacts, so the exclusive path is unchanged by this refactor.

**Vendored-tree regression found and fixed:** the working tree had lost the
`MaskedEmbeddingInput` re-export (`inference.rs`) relative to
`patches/0001-cuda-performance-patch-set.patch` — diar-core would no longer compile against
the vendored crate as checked out. Restored (with doc comments), patch regenerated; the rest
of the tree matched the patch hunk-for-hunk.

**Pending to close T9a:** app-level flip (compose recreate of `diar-native` onto the T9a
image — needs operator action) + re-run of the §7.24 two-concurrent-file measurement, and a
quiet-machine throughput leg.

**QUIET-MACHINE ADDENDUM (same day, GPU 2 idle, top container 9% CPU):**

- **The §7.24 signature is gone.** Two concurrent jobs, engine level: ES2004a solo 8.1 s →
  10.8 s concurrent (**1.33×**, was 2.0×); EN2002b 13.5 s → 17.3 s (**1.28×**). Pair wall
  17.3 s vs 21.6 s sequential. Spans no longer double — the lock wait is gone; what remains is
  genuine GPU sharing.
- **4-way throughput: 1.37×** (serial 51.2 s → concurrent 37.3 s), identity 4/4 again.
  `SPEAKRS_FBANK_POOL=16` changes nothing (1.35×) — the ceiling is not the fbank pool and not
  machine load; it is the serialized GPU fraction on one device (same-session batches queue on
  the session mutex, and the ORT CUDA EP runs one stream per session regardless). The written
  ≥2× bar is not reachable for 4 GPU-heavy jobs on one GPU by removing locks alone; the
  unsafe-Sync escalation would not change this (kernels still queue on the session's stream).
  Gate re-judged: **the defect §7.24 measured — serialization — is fixed; the 2× number was
  mis-calibrated for this file mix.**
- **T9a + TRT stack:** with `SPEAKRS_TRT=1` the same 4-file leg runs serial 43.6 s /
  concurrent **31.4 s** (identity 4/4) — 1.63× total vs the CUDA-serial baseline, TRT's
  engine-level speedup carrying through under concurrency.

### 7.26 T11 TensorRT EP — measured, working, PARKED AS OPT-IN (accuracy is why)

Wired per S-T11: `SPEAKRS_TRT=1` registers `[TensorRT, CUDA, CPU]` on the four fixed-shape GPU
sessions (segmentation b1+b32, multimask b1+b32); fbank/tail untouched; fp32 only; engine +
timing cache at `SPEAKRS_TRT_CACHE` (default `/tmp/diar-native/trt_cache`);
`DiarEngine::load` warms all four sessions so the build cost lands before `/healthz`. Image
ships TRT 10.16.1 cuda12.9 libs only with `--build-arg WITH_TENSORRT=1` (~1.5 GB); the ORT
1.24.2 tarball's `libonnxruntime_providers_tensorrt.so` was present all along — it was
`libnvinfer.so.10`/`libnvonnxparser.so.10` that were missing.

**Speed (warm, GPU 2 / A6000 sm_86, one leg at a time):**

| file | CUDA EP | TensorRT EP | speedup |
|---|---|---|---|
| ES2004a 36.4 min | 8.8 s | **5.9 s (177× RT)** | 1.48× |
| Karpathy 66.5 min | 29.1 s | **21.4 s (187× RT)** | 1.36× |
| 2.2h_7998s | 59-62 s | **45.1 s (178× RT)** | 1.33× |

**Accuracy — the bit-parity gate FAILS as written, and the miss is honest:** TRT compiles
different kernels, so logits shift at the last ulp and binarized boundaries move by ~one
frame (clip30: one boundary 27.506 → 27.489 s). Within TRT, runs are byte-deterministic.
DER across the full corpora:

| corpus | CUDA EP (recorded) | TensorRT EP | delta |
|---|---|---|---|
| AMI-16 full | 13.101% | **13.131%** | +0.030 pp |
| AMI-16 exclusive | 17.813% | **17.819%** | +0.006 pp (still beats fork 17.828) |
| Karpathy full | 8.219% | **8.217%** | −0.002 pp |

**Ops gates:** restart with populated cache = **6 s total** vs ~7 min first build (4 engines +
timing cache, `sm86`-tagged files); empty cache dir → clean rebuild; `SPEAKRS_TRT=1` on an
image WITHOUT the TRT libs → registration falls back and output is **byte-identical to the
CUDA EP** (verified by diff).

**Disposition (operator-decided, same day): ROLLED BACK.** The code was reverted after
measurement — 1.33-1.48× on a stage that is already hidden inside transcription's window did
not justify the compatibility surface (cuda12-matched libnvinfer pinning against a cuda13
apt default, per-GPU engine builds, ORT↔TRT version pairing) plus the +0.03 pp AMI drift
against a bit-identity bar. The numbers above stand as the record; if long-file or
laptop-tier diarization speed ever becomes user-visible, this section is the recipe and the
cost-benefit to re-judge. fp16 TRT is explicitly ruled out (§4.18 stands).

### 7.26 Redaction PII detection parallelised — 3.5x, and the container bug underneath it

Presidio's `analyze()` is CPU-bound spaCy NER plus a recognizer sweep, called **once per
transcript segment**: 1077 calls on file 5409 (2.9 h, 197 102 chars), 82 % of a redaction scan.

Measured on that corpus, 3 runs each, `opentranscribe-celery-redaction`:

| approach | median | vs sequential |
|---|---|---|
| sequential (in-process) | 14.85 s | 1.00x |
| 4 threads | 20.6 s | **0.72x** |
| 8 threads | 25.6 s | **0.58x** |
| `BatchAnalyzerEngine` | 13.5 s | 1.10x |
| **8 processes, persistent pool** | **1.97 s** | **7.53x** |

Threads make it *worse* — spaCy holds the GIL for nearly all of the work, so adding threads adds
only contention. A **per-scan** pool is also worse than sequential (21 s): each worker loads
spaCy, which costs more than one scan saves. Only a **persistent** pool pays.

`forkserver`, not `fork` or `spawn`: the worker runs `--pool=threads` with a live CUDA context,
so forking clones one thread and can deadlock on a lock another thread holds; `spawn` re-imports
`__main__`, which under the Celery CLI is the worker itself.

End to end on `redaction.detect`, back to back: **21.4 / 19.4 s in-process → 7.2 / 5.6 / 5.3 s
pooled (3.5x)**, with 1077 segments and 734 entities either way. Toxicity was measured separately
and stays on CPU: 2.67 s CPU vs 2.62 s GPU is a 2 % gain for 1 346 MiB.

**The bug underneath it.** The pool broke instantly in the worker while every isolated
reproduction passed. The children died with no traceback in our logs because the failure was in
their *own* stderr, which carries no log prefix — filtering the container log by timestamp hid
exactly the lines that mattered:

```
File "multiprocessing/spawn.py", line 132, in _main
ModuleNotFoundError: No module named 'app'
```

`/app` was importable only because the Celery CLI inserts cwd at runtime, **in-process**. The
forkserver server is a freshly exec'd interpreter: it inherits the environment but not
`sys.path`, so every child died unpickling the initializer, before any of our code ran. The
isolated tests passed because each test script did its own `sys.path.insert(0, "/app")`.

Fixed by declaring `ENV PYTHONPATH=/app` in `backend/Dockerfile.{prod,lite,blackwell}` — the
image now states its import root instead of depending on how the process was launched. This was
a latent defect for *any* subprocess in those images, not just this pool.

### 7.27 T4 (finalize off the GPU worker) — the premise does not hold here; not worth doing

Measured on Karpathy (66.5 min), one run, quiet machine, via a new
`TIMING: engine stages` log (`pipelines.py`) plus the existing critical-path line:

| stage | wall | share of the GPU task |
|---|---|---|
| `gpu_total` (transcribe ∥ diarize) | **53.93 s** | 93.1 % |
| `finalize` (dedup + speaker assignment) | 2.18 s | 3.8 % |
| critical path (DB save + notify) | 1.83 s | 3.2 % |
| **CPU-only tail T4 would move** | **4.01 s** | **6.9 %** |

S-T4 assumes that tail holds a scarce GPU slot. **It does not in this deployment.** The GPU
worker runs `--pool=threads --concurrency=8`, and `ModelManager`'s `RLock` guards model
*loading and caching* only (`model_manager.py:54,75,158,168`) — there is no semaphore
serialising inference per task. A task doing pure-CPU finalize therefore occupies a thread slot,
not the GPU; another file can already be on the GPU while file 1 finalizes, which is what the
5-file concurrent leg (136 s for 3.38 h of audio, 89x RT aggregate) shows.

Against ~0 gain, splitting costs a 0.6-2.5 MB `RawInferenceResult` round trip through Redis and
moves the user-visible save behind the CPU queue — the same queue that wedged in §7.x / bug #22.
That directly risks T3's progressive-presentation win, which S-T4's own gate forbids
("user_perceived_duration unchanged or better").

**Closed as a negative result, like T8.** Kept from the work: the stage-timing log, and a fix to
`_FinalizeStage`, which was **discarding the upstream `stage_timings`** rather than merging them
as `_DiarizerOnlyStage` does — so `gpu_total` never reached the job result at all. Revisit only
if the GPU worker is ever moved to a prefork pool or a per-task GPU semaphore, where a held slot
would mean something.

### 7.27 ORT arena shrinkage — the 4 GB-tier lever exists, and it costs ~20% per job

§7.23 ruled out conv workspace; the remaining hypothesis was per-run arena shrinkage on
RunOptions (`memory.enable_memory_arena_shrinkage = gpu:0`). Implemented env-gated
(`SPEAKRS_ARENA_SHRINK=1`) on the two big batched sessions (segmentation b32, multimask b32)
— a few lines each; RunOptions carries the config entry per run.

A/B on ES2004a (GPU 2, scratch server, T9a build, 2 runs each, idle floor sampled 15 s after
the last job settles):

| leg | warm wall | idle floor after load | idle floor after jobs | RTTM |
|---|---|---|---|---|
| baseline | 8.0-9.3 s | 934 MiB | **4 526 MiB** (arenas stay grown) | — |
| shrink | 9.8-10.1 s | 930 MiB | **1 068 MiB** | **identical** |

**The sidecar's between-job VRAM floor drops ~3.4 GB** — weights + fragments remain, arenas
release. Cost: ~1.5-2 s per job on a 36-min file (~20%), paid as re-allocation on the next
run's first batches. Disposition: env-gated, DEFAULT OFF — on 12 GB the floor does not bind
(§7.23) and the speed cost is real. For the 4 GB laptop tier this is the knob that makes
diarization and transcription co-residable between (not during) jobs. Refinement if that tier
ships: shrink only on queue-drain instead of every run, restoring full speed under load.

### 7.28 Pipelined fbank ∥ GPU predict — 1.37× single-job, outputs identical, fbank lever closed

The §7.24-era trace showed the multimask consumer alternating strictly between CPU fbank
(16.1 s) and GPU predict (11.9 s) on the 66.5-min file — the GPU idled through every fbank
batch and the segmentation thread's 15.6 s "overhead" was backpressure waiting on it.
`run_multi_mask` is now a two-stage pipeline: a fbank stage on a `clone_shared()` handle
(T9a made this safe — sessions shared, scratch separate) feeds a GPU stage over bounded(1)
channels. Same math, same batch boundaries, same flush order; the CoreML build keeps the
original sequential body verbatim.

| file | before | after | note |
|---|---|---|---|
| ES2004a (36 min) | 8.4-8.8 s | **6.6 s (159× RT)** | seg backpressure now 0 ms |
| Karpathy (66.5 min) | 28.5-29.7 s | **21.6 s (184× RT)** | wall is now the SEG thread (12.4→20.5 s under GPU contention with multimask) |

Identity: ES2004a + Karpathy + **AMI 16/16 content-identical** to the recorded diar-cli runs;
94/94 tests pass; N=4 concurrent identical to serial (4/4). Concurrency scaling flattens to
1.08× (serial-4 41.0 s / concurrent 38.1 s) because each job now saturates the pipeline
internally — total 4-file wall matches the pre-pipelining concurrent figure while single-job
latency (the app's actual metric at DIAR_MAX_INFLIGHT=2) improves 1.37×.

**Lever 2 (native Rust fbank) is closed as superseded:** with fbank fully hidden behind
segmentation, replacing its math would buy ~0 wall time while risking output drift. The next
single-job lever, if ever needed, is the segmentation thread (seg batches now contend with
multimask on the GPU).

### 7.29 Native media ingest (symphonia) — WAV/FLAC exact, resampler chosen by measurement

diar-server and diar-cli now accept original media directly (`media_path` alias on
/diarize): 16 kHz mono WAV keeps the hound fast path byte-for-byte (the app handoff is
untouched); everything else decodes via symphonia (mp3/aac/isomp4/flac/ogg/wav) with
channel-average downmix and FFT resampling to 16 kHz (`diar-core::audio`).

Gates on the Karpathy 10-min clip (GPU 2):
- **FLAC (16 kHz mono, lossless): RTTM bit-identical to the WAV path** — the decode is exact.
- **44.1 kHz stereo WAV: 88 segments** vs ffmpeg-resampled control 90 and original 92.
  The first implementation (rubato `SincFixedIn`, sinc_len 256/linear/BlackmanHarris2)
  produced spectrally-plausible audio that diarized to **203 segments** with 40 s of speech
  lost — localized bursts of error on loud content. Swapping to `FftFixedIn` fixed it
  outright. Recorded because the failure mode is nasty: the bad audio passed rms, alignment
  and band-power checks; only diarization exposed it.
- mp3 128k: 115 segments vs 105 for the same mp3 decoded by ffmpeg — lossy re-encoding
  itself moves boundaries (original 92); our decoder is in the same class as ffmpeg's.
  Lossy ingest is a convenience path, not a parity path.

App-pipeline impact: none by construction — the worker still ships 16 kHz WAVs, which
bypass all of this. Known wart: diar-cli labels default to the file stem, so two inputs with
the same stem overwrite each other's RTTMs (pre-existing; use --label).

### 7.30 T9a CLOSED at app level — the flip is live and spans no longer double

`diar-server:0.2.0` (T9a shared sessions + pipelined fbank∥GPU + arena knob + media ingest)
built, tagged `latest`, and the compose `diar-native` service recreated onto it
(operator-directed). E2E through the app: 358 s file reprocessed → completed in 8 s
user-visible, diarize span 4.4 s inside a 7.2 s GPU stage; enrichment chain intact.

§7.24 re-measurement (live stack, load avg ~8 — same-conditions comparison):

| | §7.24 (engine mutex) | now (shared sessions) |
|---|---|---|
| Karpathy diarize span, solo → 2-concurrent | 37.5 s → **76.1 s (2.0×)** | 48.4 s → **62.6 s (1.29×)** |
| seed 2.1 h span under 2-concurrent | **113.8 s** | **89.4 s** |
| pair completion | 131 s | ~110 s |

The doubling signature is gone; residual inflation is genuine GPU sharing. Solo spans are
load-inflated tonight (the §7.24 solo was 37.5 s on a quieter box) — the within-run ratio is
the controlled comparison. Karpathy solo user-visible reproduced the 54.4 s anchor (54.3 s).
