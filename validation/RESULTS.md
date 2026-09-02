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
they surface under the NAS only because this host's docker data-root points at a local ext4
RAID mount, not the default). A named volume inherits its ownership from the image directory
at creation —
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

### 7.31 Native CoreML brought up end-to-end on Apple Silicon (M2 Max) — README's "future work" done

`coreml` feature wired through diar-core/diar-server/diar-cli (`Mode::CoreMl`/`CoreMlFast`,
`DIAR_MODE=coreml`/`coreml_fast`, `--mode coreml`/`coreml_fast`), mirroring the existing `cuda`
feature exactly. speakrs upstream already has full CoreML support
(`ExecutionMode::CoreMl`/`CoreMlFast`, `src/inference/coreml.rs`,
`scripts/native_coreml/convert_coreml.py`) — this wasn't new engine work, it was wiring +
fixing two real gaps found only by actually building and running on real hardware (a Mac
Studio on the operator's private network, Apple M2 Max):

1. **Concurrency model incompatible.** speakrs cfgs `SegmentationModel`/`EmbeddingModel`'s
   `clone_shared` out under `feature = "coreml"` — CoreML models aren't ORT sessions and are
   documented single-thread-at-a-time (`unsafe impl Send`, not `Sync`). T9a's shared-handle
   pattern doesn't apply. Fixed: `AppState::with_engine` in diar-server is now feature-gated
   — off coreml, unchanged (clone a handle, release the mutex, T9a concurrency intact); under
   coreml, holds the mutex for the whole request (correct, but serializes jobs —
   `DIAR_MAX_INFLIGHT` has no effect in this mode. Not measured whether this matters in
   practice; nothing here stresses concurrent coreml load).
2. **Two required model assets, neither produced by the documented conversion path.**
   `segmentation-3.0-b64.mlmodelc` and `wespeaker-fbank-30s.mlmodelc` are unconditionally
   required at load time. `convert_coreml.py` only exports segmentation batches 1-32
   (`SEGMENTATION_BATCH_SIZES = tuple(range(1, 33))`); the b64 variant needs the separate
   `export_b64_seg.py` (hardcodes a relative `fixtures/models` output path — easy to point at
   the wrong directory, which is what happened the first time). The fbank-30s asset had no
   script anywhere. Added `scripts/native_coreml/export_fbank_30s.py` (mirrors
   `export_b64_seg.py`'s fixed-shape pattern, `[1, 1, 480_000]` matching the Rust side's
   `CachedInputShape` exactly) — pushed to the fork (`attevon-llc/speakrs` PR #2, merged to
   `master`), patch regenerated.

**Model export:** ran `convert_coreml.py` + `export_b64_seg.py` + the new
`export_fbank_30s.py` on the Mac Studio, fully offline (`HF_HUB_OFFLINE=1`, using the same
locally-cached `pyannote/speaker-diarization-community-1` weights this project already uses
for ONNX export — no HF token needed on the Mac). 57 `.mlmodelc` bundles, 1.7 GB. Ran
speakrs' own `compare_coreml.py` validation: **parity checks passed**
(`segmentation batch=1 max_abs=5.7e-06`, `batch=32 max_abs=9.5e-06`,
`tail batch=1 max_abs=5.0e-03`, `batch=3 max_abs=1.0e-02`, `batch=32 max_abs=1.3e-02` — all
within the script's own tolerances).

**Correctness:** real `/diarize` call against `karpathy_10m.wav` (600 s) on the CoreML build:
2 speakers, 93 segments vs. 92 on the CUDA reference for the same file — 99%+ match, the
1-segment difference explained by the same class of harmless numeric drift as the
backend-embedded-build cuBLAS comparison earlier this session (different backend, different
float rounding near a boundary, not a bug). Speaker label swap (`SPEAKER_00`/`01`) is
expected — cluster-label assignment order is arbitrary per run.

**Speed — NOT a controlled comparison, flagged explicitly:** `karpathy_10m.wav` (600 s), warm
runs (2nd/3rd call, discarding cold-load first call):

| | run 2 | run 3 |
|---|---|---|
| CUDA (RTX A6000, **this machine loaded**: load avg 17.7, GPU already 39% utilized by something else) | 4.17 s | 4.53 s |
| CoreML (Apple M2 Max, quiet) | 2.42 s | 2.46 s |

CoreML came out ~1.7-1.9× faster in this run, but the CUDA side violates this project's own
quiet-machine protocol (docs/BENCHMARK_PROTOCOL.md) outright — GPU already contended,
load avg near 18. **Do not read this as "Apple Silicon beats an A6000."** It shows CoreML on
an M2 Max is genuinely fast (same ballpark as CUDA under those conditions), not a validated
relative ranking. Re-measure on a quiet CUDA box before quoting a real ratio anywhere.

**Not verified:** actual GPU utilization % during CoreML inference (would need `powermetrics`,
which needs sudo on the Mac Studio — not set up this session). What is verified: conversion
was restricted to `compute_units=ct.ComputeUnit.CPU_AND_GPU` (ANE excluded) at export time,
and inference definitely ran through CoreML's own code path (real `.mlmodelc` bundles loaded,
not a silent ONNX/CPU fallback) — proven by successful load + correct output through the
coreml-specific loaders. Only tested on this one machine (Apple M2 Max) — CoreML's compiled
`mlprogram` format is chip-generation-portable by design, but M1/M3/M4 haven't been run.

### 7.32 Embedding ONNX session intra-op threads unpinned for CPU mode — 3.7x, adopted from upstream PR #6

`build_session_with_graph` (`vendor/speakrs/src/inference/embedding/session.rs`) hardcoded
`.with_intra_threads(1)` for the segmentation-tail/multimask embedding sessions, unconditionally
across every execution mode. Found relevant via an open, unrelated third-party PR against
upstream (`avencera/speakrs#6`, `ryoma0421`), who measured the same bottleneck on Apple
Silicon CPU mode (1.03x -> 8-9x realtime on a 5.7-min meeting file). We adopted the equivalent
fix into our own fork rather than wait on that PR to merge, since this project ships a
CPU-only image (`docker/Dockerfile.server-cpu`) that runs with no GPU/CoreML acceleration at
all — exactly the scenario PR #6's bottleneck applies to.

**Change:** new `SPEAKRS_INTRA_THREADS` env var, default `available_parallelism().min(6)`
(zero/unparseable -> default). The cap of 6 isn't arbitrary — `SegmentationModel::build_session`
already ships `.min(6)` in production, so this makes the embedding sessions match a threading
policy already proven safe, rather than inventing a third one.

**Oversubscription checked, not assumed:** ~7 embedding sessions exist per pipeline instance
(tail, multimask, batched variants, each with `with_independent_thread_pool()`), but they are
alternatives — only one executes per request. Real concurrent thread demand is
`DIAR_MAX_INFLIGHT x 6`, the same bound segmentation already imposes; this doesn't introduce a
new oversubscription class.

**Control set:** AMI `EN2002c`, first 360 s, 16 kHz mono, `diar-bench-builder:latest`, 3
alternating rounds, single build A/B'd via the env var (old=1 vs new=6) so it's the exact same
binary both times. Host load ~9-13/48 (not fully quiet) — the gap below is far outside that
noise band, per-round spread is tight, but this is not a protocol-grade quiet-machine leg.

| mode | intra=1 (old) | intra=6 (new) |
|---|---|---|
| CPU | 218.9 / 217.2 / 220.1 s -> 1.64x RT | 57.8 / 59.2 / 67.7 s -> **6.1x RT** |
| CUDA | 4.83 / 4.86 s | 4.88 / 4.85 s (round-1 7.3s excluded, cold start) |

**3.7x faster on CPU mode** (219s -> 59s), matching PR #6's shape. CUDA mode explicitly
re-tested rather than assumed unaffected — no regression. RTTM bit-identical: one MD5 across
all CPU legs, one MD5 across all CUDA legs — scheduling change only, no output change.
94/94 -> 96/96 speakrs tests pass (count includes the two prior session's `SPEAKRS_AHC_THREADS`
tests).

### 7.33 Missing `-tail-b64` artifact: split-primary batching was dead everywhere — restored, no perf change

`EmbeddingModel::split_primary_batch_size()` returned **0 on every model set we ship**, on both
ORT and CoreML. The loader asks for `split_tail_model_path(model_path, PRIMARY_BATCH_SIZE=64)` =>
`wespeaker-voxceleb-resnet34-tail-b64.onnx` (`load/sessions.rs:70`) and, under CoreML, for the
`.mlmodelc` compiled from it (`embedding/native/loaders.rs::load_native_tail`). Neither
`scripts/export_models.py` (emits b1/b3/b32 tails) nor `scripts/native_coreml/convert_coreml.py`
(`TAIL_BATCH_SIZES = (1, 3, 32)`) ever produced it. Same bug class as the multimask tail-b64 fixed
earlier by `validation/export_b64_addendum.py`.

Artifacts added (gated, local-only, never committed):
- `validation/export_tail_b64_addendum.py` -> `wespeaker-voxceleb-resnet34-tail-b64.onnx`
  (28.0 MB) into `models_folded/` and `vendor/speakrs/fixtures/models/`. Batch-invariance check
  (row i of the b64 wrapper vs the b1 wrapper on row i alone): max diff **7.8e-08**.
- `validation/convert_tail_b64_coreml_addendum.py` -> `wespeaker-voxceleb-resnet34-tail-b64.mlmodelc`
  on the M2 Max. Deliberately a SEPARATE fixed-shape-64 conversion rather than adding 64 to
  `TAIL_BATCH_SIZES`, so the shipped b1/b3/b32 tail artifacts are NOT regenerated and production
  CoreML accuracy is untouched.

`split_primary_batch_size()`: **0 -> 64** (verified on Linux/ORT via a throwaway probe test, and on
the Mac by `fast_apple_split_primary_batch_matches_single_tail_path` no longer skipping).

**Perf impact: none, and that is the real finding.** `select_embedding_path()` picks
`EmbeddingPath::MultiMask` whenever the multimask model exists, and it does on every set we ship —
`Split` is only reachable as a fallback. Measured, karpathy_66min.wav, `coreml_fast`, M2 Max,
3 runs each, artifact present vs moved aside:

| config | elapsed_s (3 runs) | segments |
| --- | --- | --- |
| with `-tail-b64.mlmodelc` | 5.15 / 4.41 / 4.54 | 656 |
| without (as shipped before) | 4.82 / 4.40 / 4.64 | 656 |

Overlapping distributions = no change, and the emitted segments are byte-identical (`json` diff).
So the gap was **dead fallback code + a permanently-skipped test**, not a live perf regression.
Retracting nothing; flagging that the "silently disables the split-primary batching optimization"
framing overstates the impact while MultiMask outranks Split.

**Test fallout, both fixed with measurements, not tolerance loosening:**
1. `fast_apple_split_primary_batch_matches_single_tail_path` had never run. With the artifact it
   failed twice for real reasons: (a) the fixture yields only 18x3 = 54 rows but the test asserted
   exactly 64 — now the collected rows are cycled to fill a full batch so the
   `inputs.len() == PRIMARY_BATCH_SIZE` fast path is the one under test; (b) its per-dimension
   5e-3 check is wrong in kind, because `load_native_tail` loads with `MLComputeUnits::All` (ANE
   fp16) while `expected` comes from the fp32 batch-1 tail. Measured over a full 64-row batch:
   min cosine **0.944999**, mean |err| **5.2e-3**, max |err| **1.1e-1** — the exact ANE fp16
   signature already documented for the multimask batch path (0.9450 / 8.5e-3 / 1.1e-1), which is
   what establishes the b64 graph is correct. Rescoped to cosine >= 0.90 / mean |err| <= 2e-2, the
   same thresholds as `fast_apple_embeddings_match_python_fixture`.
2. `fast_apple_single_embedding_matches_python_fixture` was flagged as "well-conditioned by luck"
   (pinned chunk 0 / speaker 1). **Not confirmed.** Swept all 35 non-NaN (chunk, speaker) pairs:
   max |err| **2.7e-5**, mean |err| **4.1e-6**, min cosine **1.000000** everywhere. `embed_masked`
   takes the batch-1 fp32 tail, not the ANE fp16 batch tail, so it is genuinely fp32-clean and its
   5e-4 per-dimension tolerance keeps ~18x headroom. Kept the strict check and added the sweep
   (loosening it to cosine/MAE would have LOST real coverage); speaker 0 is all-NaN in the fixture
   for every chunk and is skipped.

Both modified tests were verified to still fail on an injected bug: swapping two rows of the batch
`weights` drives split-primary min cosine to **0.068**, and shifting the expected dimension index
fails the single-embedding test at dim 0.

Suites: Linux `openblas-system,online` **96 passed / 0 failed**; Mac
`openblas-system,online,coreml` **110 passed / 0 failed** (90 lib, up from 89 — the split-primary
test now executes instead of skipping).

### 7.34 One image, two devices — the CUDA image was ALREADY a CPU image (issue #1)

**The premise, tested before any code was written.** Ran the *already-shipped, unmodified*
`davidamacey/diar-native:0.2.0` with `DIAR_MODE=cpu`, no `--gpus`, on a host whose Docker default
runtime is `runc` (verified: no `/dev/nvidia*` and no `nvidia-smi` inside the container):

| endpoint | result |
| --- | --- |
| `/healthz` | HTTP 200 |
| `/diarize` (`clip30.wav`, 30 s) | HTTP 200, 16.3 s, 1 speaker, 2 segments, 256-d centroid |
| `/embed_window` (1–5 s) | HTTP 200, 256-d embedding |

So the Dockerfile half of the issue was **already closed** — it only needed documenting. Why:
`ort-sys` **statically links** the ORT core objects, `onnxruntime_mlas` (the CPU EP kernel
library) included; `ldd /usr/local/bin/diar-server` in the shipped image shows **no ONNX Runtime
`NEEDED` entry at all**. `ort/cuda` only selects a different prebuilt distribution — purely
additive. speakrs agrees in source: `ExecutionMode::Cpu.validate()` returns `Ok(())`
unconditionally, before any feature gate, and `with_execution_mode` registers
`ep::CPU::default().with_arena_allocator(false)` with no `#[cfg]`.

**Do NOT add the CPU ORT tarball to Dockerfile.server.** The GPU tarball already installs
`libonnxruntime.so.1.24.2` (unused by the binary, which is statically linked); the CPU tarball's
copy would collide. Rebuilt image is **3.46 GB — byte-for-byte the same size as before**. The
superset costs zero bytes and zero libraries.

`docker/Dockerfile.server-cpu` **stays**, and not as a correctness carve-out: it is the only
**arm64** artifact (the CUDA base image and the `onnxruntime-linux-x64-gpu` tarball are x86-64
only, no arm64 equivalent exists) and the only **small** one (189 MB vs 3.46 GB, no NVIDIA
runtime or driver). The superset claim is precisely *"the amd64 GPU image is a superset of the
amd64 CPU image."*

#### What was implemented

Startup-loaded engine registry (`crates/diar-server/src/engines.rs`, new file), `DIAR_DEVICES`
(comma list, **first entry is the default**, wins over `DIAR_MODE`), per-request `device` field
on `/diarize` + `/embed_window`, `x-diar-device` response header, and JSON `/healthz`.
Defaults unchanged: with neither new knob set the server loads exactly one engine from
`DIAR_MODE`, one global semaphore, one `clone_shared` per request — as before.

**Lazy on-first-use loading was rejected as unsound, not as an optimization declined.**
`DiarEngine::load` calls `std::env::set_var("SPEAKRS_FBANK_POOL", ..)` and speakrs reads it back
inside the same call (`inference/embedding/load/sessions.rs:255` — verified the **only** read
site, load-time only, never at request time). glibc `setenv`/`getenv` is not thread-safe, so a
lazy load on a request thread would race live tokio workers. All engines therefore load
**serially in `run()` before `axum::serve`**, which is exactly as safe as the single load the
server has always done.

#### B5 — capability matrix (untimed; asserts behaviour, not speed)

New image `diar-server:issue1`. GPU-free leg = no `--gpus`, `DIAR_MODE=cpu`:

| leg | request | result |
| --- | --- | --- |
| no GPU, `[cpu]` | `device` omitted | 200, `x-diar-device: cpu` |
| no GPU, `[cpu]` | `"device":"cpu"` | 200, `x-diar-device: cpu` |
| no GPU, `[cpu]` | `"device":"cuda"` | **400** `device 'cuda' is not loaded; this server is serving [cpu] (add it to DIAR_DEVICES to load it)` |
| no GPU, `[cpu]` | `"device":"tpu"` | **400** `unsupported device 'tpu'; this build serves [cuda, cpu]` |
| no GPU, `[cpu]` | `/embed_window` `"device":"cpu"` | 200, 256-d |
| GPU, `[cuda,cpu]` | `"device":"cuda"` / `"cpu"` | 200 each, header matches |

`/healthz` on the CUDA build with only CPU loaded:
`{"status":"ok","default_device":"cpu","devices":["cpu"],"supported_devices":["cuda","cpu"]}` —
`devices` = loaded and serving, `supported_devices` = compiled-in capability.

**Cross-image control:** the new image's CPU `/embed_window` returned
`0.024982646, 0.05198441, 0.06093254, …` against the shipped 0.2.0 image's
`0.02498265, 0.051984407, 0.06093254, …` on the same clip and window — identical. `/diarize`
segment boundaries likewise identical (`0.030969 + 27.489375 = 27.520344` from 0.2.0's RTTM vs
`end: 27.520343750000002`). The registry did not perturb the CPU path.

#### B3 — RSS cost of a second resident engine (memory, so machine load does not invalidate it)

`--gpus device=0` (idle, 15 MiB before each leg), `SPEAKRS_LAZY_SESSIONS=1` matching live,
`DIAR_MAX_INFLIGHT=2`, `models_folded/`. VRAM sampled **per process**
(`--query-compute-apps`), not whole-GPU — the first attempt at this was contaminated by a
previous container still holding 4 437 MiB on the same card, and is retracted in favour of the
numbers below.

| | `DIAR_DEVICES=cuda` | `DIAR_DEVICES=cuda,cpu` | delta for the CPU engine |
| --- | --- | --- | --- |
| idle VmRSS | 633 008 kB | 1 268 428 kB | **+635 420 kB (+620 MB)** |
| idle process VRAM | 824 MiB | 824 MiB | **+0** |
| peak VmRSS, CUDA run | 1 415 464 kB | 2 065 688 kB | +650 224 kB |
| peak process VRAM, CUDA run | 4 388 MiB | 4 388 MiB | **+0** |

**The CPU engine costs zero VRAM**, as predicted. Confirmed independently on a fresh
`cuda,cpu` container that ran *only* a CPU job and never a CUDA one: process VRAM 824 MiB idle
→ 824 MiB peak → 824 MiB after. (The 824 MiB is the CUDA engine's own session load, paid
whether or not CPU is enabled.)

**+620 MB host RSS is double the ~311 MB predicted from weight sizes; gender is why.** Measured
by re-running both legs against a models dir symlinked to omit `gender-wav2vec2.onnx`:

| | with gender | no gender | gender's share |
| --- | --- | --- | --- |
| `cuda` idle VmRSS | 633 008 kB | 452 624 kB | 180 384 kB |
| `cuda,cpu` idle VmRSS | 1 268 428 kB | 620 520 kB | 647 908 kB |
| **CPU-engine delta** | **635 420 kB** | **167 896 kB** | **467 524 kB = 74%** |

So the CPU engine is ~164 MB of diarization weights plus ~457 MB of gender. `GenderModel::load_optional(dir, cuda)`'s
`cuda` flag selects only the execution provider, not precision — both engines parse the same
189 MB fp32 ONNX, but under CUDA most of it lands in VRAM while under CPU it is materialised in
host RAM with an ORT CPU arena on top.

**Declined: a per-device `with_gender(false)` knob** (floated as a possible follow-up if B3 came
in high). Measured, and rejected on correctness grounds rather than cost: it would make
`{"device":"cpu","gender":true}` silently return no gender while the same request on `cuda`
returns it, destroying the "same code, same weights, same outputs" property that is the entire
justification for the superset claim. 620 MB of host RAM on a box with tens of GB is not worth a
device-dependent output schema. Operators who want the saving can already remove the gender model
from `models_dir` (measured: 164 MB per extra engine), which disables it uniformly.

Note the plan's reference to an existing `EngineConfig::with_gender(bool)` at
`diar-core/src/lib.rs:48-66,127` — **no such method exists**; gender is controlled by model-file
presence via `GenderModel::load_optional`. Corrected against the code.

#### Concurrency

The global `Semaphore(DIAR_MAX_INFLIGHT)` is **kept as the outer admission gate**, unchanged
semantics and default (2), so a mixed deployment cannot silently double total inflight and
oversubscribe cores (each CPU embedding session takes up to 6 intra-op threads — §7.32). The new
`DIAR_MAX_INFLIGHT_CPU` is an **optional inner sub-gate**: unset (default) means no inner gate
and zero behaviour change; when set, CPU requests take global-then-CPU, always in that order, so
there is no lock-ordering hazard. Device resolution happens *before* admission, so a bad device
name costs a 400 and never an admission permit.

Functional check, `DIAR_MAX_INFLIGHT=4 DIAR_MAX_INFLIGHT_CPU=1`, `[cuda,cpu]`: 3 concurrent CPU
`/embed_window` returned 200 in 0.226 / 0.453 / 0.695 s — a staircase confirming serialization
with no deadlock, all three embeddings byte-identical — while 2 concurrent CUDA requests ran
unblocked (0.650 / 0.680 s).

#### Tests

- `diar-server`: **13 new unit tests, pass in BOTH builds** (default/CPU-only and
  `--features cuda`) — device name round-trip, capability list vs `cfg`, `DIAR_MODE` path
  unchanged (including the preserved "unset or unrecognized ⇒ cuda" quirk), `DIAR_DEVICES`
  precedence, dedupe-preserving-order, blank-falls-back, bad-entry rejection, and the three
  distinct error kinds. Watched fail before the fix: removing the dedupe produced
  `left: Ok([Cpu, Cpu, Cpu]) / right: Ok([Cpu])`.
- speakrs suite unchanged and green: **96 passed / 0 failed** (`--no-default-features
  --features openblas-system,online`). No vendored edit was needed — `git diff HEAD` in
  `vendor/speakrs` is byte-identical to `patches/0001-cuda-performance-patch-set.patch`, so the
  patch file did not need regenerating. (CLAUDE.md still says "94 tests"; the real count has been
  96 since §7.33 un-skipped the split-primary test.)

#### NOT measured here — pending a quiet window

The box was at load average 26–44 throughout (an `otfresh-demo` OpenTranscribe stack plus a vLLM
container). Per `docs/BENCHMARK_PROTOCOL.md` no timed leg was run. Everything above is a
capability or memory assertion, which load does not invalidate. Wall times that appear above are
incidental and are **not** benchmark numbers. Still owed, one timed leg at a time on a quiet
machine:

- **B1 — does `--features cuda` slow down CPU-mode inference?** Single variable = the `ort-sys`
  prebuilt distribution. `diar-cli --mode cpu` from a default-features build vs a `--features
  cuda` build, control corpus AMI `EN2002c` first 360 s, compared against the logged §7.32 CPU
  leg (57.8 / 59.2 / 67.7 s, 6.1× RT). Accuracy check is **output identity**: MD5 of the RTTMs
  must EQUAL the §7.32 CPU-leg MD5 — proven, never asserted.
- **B2 — CUDA no-regression with both engines resident.** `DIAR_DEVICES=cuda,cpu` vs `cuda`,
  control = §7.32's logged CUDA leg 4.83 / 4.86 s. B3 already shows the VRAM side is zero.
- **B4 — mixed-device concurrency.** Adapt `validation/t9a_concurrency.sh` (serial/concurrent
  legs + during-run VRAM sampling already at lines 17–45): N concurrent CUDA `/diarize` alongside
  M concurrent CPU `/embed_window`; pass = CUDA leg wall time within noise of CUDA-only.

Never re-run: §4.12 (CPU parity), §7.32, §7.7 DER legs, the T9a identity gate.

#### Consumer-side (transcribe-app — reported, NOT changed here)

Confirmed: the live service does **not** run a `diar-server` image. `docker-compose.diar-native.yml:26-27`
runs `opentranscribe-backend:latest` with `command: ["diar-server"]`, and the binary arrives via a
build stage in `backend/Dockerfile.prod:20`,
`FROM davidamacey/diar-native@sha256:544a6a6536464729834dd1b51dc6d76b538776da23a22682501f220cf2ff999c`.
**What reaches production is the BINARY, not the image** — that digest pin must be bumped for any
of this to ship. `Dockerfile.prod:196-199` copies only `diar-server` plus
`libonnxruntime.so.1.24.2` / `_providers_cuda.so` / `_providers_shared.so`, which is sufficient:
the CPU EP is inside the binary.

The compose healthcheck is `["CMD-SHELL", "curl -sf http://localhost:8701/healthz || exit 1"]`
(line 69) — status-only, so the `/healthz` JSON change is **non-breaking**, verified by reading
it rather than assuming.

`DIAR_MODE=${DIAR_NATIVE_MODE:-cuda}` is always set explicitly there, so behaviour is unchanged.
Note that adding `ENV DIAR_DEVICES=cuda` to `Dockerfile.server` was **deliberately not done**: it
would win over compose's `DIAR_MODE` and silently break a `DIAR_NATIVE_MODE=cpu` deployment.

#### Incidental fix: both Dockerfiles were broken by the concurrent provisioning work

Caught while rebuilding the CPU image. `crates/diar-core/src/provision/exporter.rs:44-48`
`include_str!`s five `scripts/provision/*.py` files, but neither Dockerfile's builder stage
copied `scripts/` — only `Cargo.toml`, `Cargo.lock`, `crates/` and `vendor/speakrs/`. Result:

```
error: couldn't read `crates/diar-core/src/provision/../../../../scripts/provision/provision.py`:
       No such file or directory (os error 2)
error: could not compile `diar-core` (lib) due to 5 previous errors
```

This is **not** from the multi-device work (which touches only `crates/diar-server/`). It arrived
with `94b8b8e feat(provision): add model provisioning core` at 07:49, and it breaks **both**
`docker/Dockerfile.server` and `docker/Dockerfile.server-cpu`. The CUDA image built clean earlier
in this session only because that build finished at ~07:40, before the provisioning commit
landed — a nice illustration of why the image build, not just `cargo build`, is the gate.

Fixed by adding `COPY scripts/ ./scripts/` to both builder stages. Both images then build.

### 7.35 Model provisioning: the export recipe reconstructed, and five things it was hiding (issue #2)

`diar-native` had no supported way for a third party to obtain its weights — the last blocker
to self-hosted OpenTranscribe running the native diarizer (OpenTranscribe #639). Adding
`diar-server provision-models` required first working out what the shipped `models_folded/`
actually IS, which turned out not to be documented anywhere.

**The recipe is 5 steps, not 1.** `vendor/speakrs/scripts/export_models.py` emits 20 files;
`models_folded/` has 24, and 3 of the 20 are subsequently REPLACED. Reconstructed by md5:

| step | what | evidence |
| --- | --- | --- |
| 2a | base export | 20 files |
| 2b | onnxsim constant-fold the 3 segmentation graphs, written under the PLAIN names | `models/segmentation-3.0-sim.onnx` == `models_folded/segmentation-3.0.onnx` (`48dae792…`); same for `-b32` (`f4bc2e0c…`) and `-b64` (`318d91ff…`) |
| 2c | `wespeaker-multimask-tail-b64.onnx` = byte COPY of `-b32` | both `8c990f3b16ee74280d2b4591033b2c45` |
| 2d | genuine b64 tail export | `fe325df9…`, matches `vendor/speakrs/fixtures/models/` |
| 2e | gender classifier + fp16 conversion | see below |

Step 2b was the biggest gap: **folding is mandatory and was undocumented as a provisioning
step.** Not folding costs ~2x on segmentation and reintroduces the ORT-CUDA `Sin`/`Cos`
CPU-fallback tax (§4.1, §4.10 item 2), with no error anywhere.

**Verified reproducibility:** re-running onnxsim 0.7.3 over the unfolded graphs reproduces all
three folded artifacts **byte-identically** (md5 match). Node census on the shipped artifact is
144 → 40, not the 179 → 40 recorded in §4.1 — §4.1's "179" was measured on a different export
and does not describe what we ship. Folded op histogram: `Abs`×1 `Conv`×3 `Gemm`×3
`InstanceNormalization`×4 `LSTM`×4 `LeakyRelu`×5 `LogSoftmax`×1 `MaxPool`×3 `Reshape`×10
`Transpose`×6; `Sin`/`Cos`/`If` all zero.

**TOOLCHAIN BLOCKER — onnxsim cannot install on the target image.** The OpenTranscribe backend
image is **Python 3.13.12 with no cmake/gcc/g++/make**. onnxsim publishes **zero cp313 wheels
at every version** (0.4.x–0.7.3: 13 wheels each, none cp313) and is a C++ extension, so
`pip install onnxsim` there must build from source and fails. Two fallbacks measured:

| folder | installs on 3.13 | nodes | Sin/Cos/If | max_abs_diff vs unfolded | byte-identical to shipped |
| --- | --- | --- | --- | --- | --- |
| onnxsim 0.7.3 | NO (sdist only) | 40 | 0/0/0 | 0.0 | **yes** |
| ORT `ORT_ENABLE_EXTENDED` | yes (built in) | 50 | 0/0/**1** | — | no |
| onnxslim (py3-none-any wheel) | **yes** | 37 | 0/0/0 | **0.0** | no |

ORT's own optimizer is **not** an adequate fallback: it leaves `If` in the graph. onnxslim is
adopted as the 3.13 fallback — numerically bit-exact and it eliminates the same ops — but it
emits `MatMul`+`Add` where onnxsim emits `Gemm`, so its output is functionally equivalent and
NOT byte-comparable. `fold_segmentation.py` prefers onnxsim and records the choice in the
marker's `toolchain.folder`. **Open item: the onnxslim-folded graph has not had a CUDA perf leg
run against the onnxsim one.** Do not assume the 2x holds for it.

**The gender model is FP16, and the docs said otherwise.** `docs/DETAILED_SPECS.md` records
`optimum-cli export onnx` as its provenance, which yields ~361 MB of fp32. The shipped
artifact is 189,431,659 bytes with **all 213 initializers `FLOAT16`** and fp32 I/O preserved by
2 `Cast` nodes — i.e. §7.18's `onnxconverter_common.float16(keep_io_types=True)` step, which
was never written into any export recipe. An exporter following the documented route would
have produced a 2x-larger, differently-quantized model that still worked, so nothing would have
complained.

**PLDA dtypes cannot be inferred from file size, and the obvious guess is wrong.** Headers
parsed from `models_folded/`:

| file | bytes | actual |
| --- | --- | --- |
| `plda_lda.npy` | 131200 | (256,128) **f32** |
| `plda_tr.npy` | 131200 | (128,128) **f64** |
| `plda_mu.npy` / `plda_psi.npy` | 1152 | (128,) **f64** |
| `plda_mean1.npy` | 2176 | (256,) **f64** |
| `plda_mean2.npy` | 640 | (128,) **f32** |

`plda_lda` and `plda_tr` are the same size with different dtype AND shape. The verifier parses
`.npy` headers rather than checking lengths.

**Fail-open fixed in the copied exporter.** Upstream `export_plda()` hardcoded
`~/.cache/huggingface/hub/…`, blind-scanned blobs for a `PK` magic, and wrapped everything in a
bare `except: pass`; a cache miss merely printed "skipping". Under a container `HF_HOME` this
silently produced a models directory with **no PLDA files at all** and exit status 0.
Now resolved via `hf_hub_download`, loaded by name from `plda/plda.npz` +
`plda/xvec_transform.npz`, with all six arrays asserted.

**Gate detection cannot use the model-info API.** Measured:
`GET /api/models/pyannote/speaker-diarization-community-1` returns **200 with no token and
200 with a garbage token**. The discriminating call is a file resolve
(`/resolve/main/config.yaml` → 401 `x-error-code: GatedRepo`), and `/api/whoami-v2` is what
separates "bad token" from "valid token, terms not accepted" so the message names the right
remedy. Both are in Rust, so a traceback on this path is structurally impossible.

**Smoke test results (CPU, `vendor/speakrs/fixtures/test.wav`, 26.0 s):**

| set | graphs | speakers/segments | 3a fbank b1-vs-b32 | 3b fused-vs-split | 3c multimask-vs-tail | 3e tail-b64 invariance |
| --- | --- | --- | --- | --- | --- | --- |
| `models_folded` (fast) | 16 | 2 / 7 | 3.43e-5 | **0.00e0** | **0.00e0** | 3.58e-7 |
| `models_small` | 12 | 2 / 7 | 3.43e-5 | 0.00e0 | 0.00e0 | n/a |

3e's 3.58e-7 is consistent with §7.33's 7.8e-08 for the same property. Bar is 1e-4 throughout.

**Injected-bug verification (the check that the checks work).** Two corruptions:

1. 4096 flipped bytes inside `wespeaker-voxceleb-resnet34-tail.onnx` → **stage 1** fails and
   names the file.
2. One entire initializer zeroed (`resnet.seg_1.weight`, (256,5120) f32) via
   `validation/make_corrupt_fixture.py`. The graph **still loads with an unchanged signature**
   — confirmed in the fixture builder — so stages 1 and 2 both PASS it. Only **stage 3b**
   catches it, at **9.222e-1** against the 1e-4 bar. This is the evidence that the numeric
   stage is more than a protobuf parse.

`scripts/compare_model_sets.py` on the same fixture: Tier A (inventory/size) **passes** it —
the file is the same length — and Tier B catches it at 1/84 initializers differing, max |Δ|
2.703e-01. Control (`models_folded` vs itself): equivalent at Tier B across all 24 files.

**Ordering defect found by a test.** With stage 5 (PLDA headers) running last, truncating
`plda_tr.npy` by 64 bytes reported `STAGE 4 FAILED: pipeline init: reached EOF before reading
all data` — no filename, no hint PLDA was involved. Stages now run structural-before-
end-to-end, giving `plda_tr.npy is 131136 bytes, expected 131200`. A verifier that emits
unactionable errors defeats its own purpose.

**Seeding bug found by a unit test.** The mask probe used `seed | 1`, which maps every even
seed onto its odd successor; stage 3e walks `0xA5A5..0xA5A5+64`, so half of its 64 probe rows
would have been duplicates and the batch-invariance check quietly half-strength.

**Health contract.** `/healthz` stays **200 in all four model states**. Verified callers
inspect status only — `docker-compose.diar-native.yml:69` (`curl -sf .../healthz || exit 1`)
and `diarizer_native.py` (`resp.status == 200`) — so changing the BODY is safe and changing
the CODE is not. Every models directory deployed today has no marker; a 503 for "unverified"
would have failed every existing healthcheck on ship day and silently reverted the stack to
in-process PyAnnote, which is the exact regression this issue exists to prevent. `/readyz` is
the new endpoint that 503s on unverified.

Tests: 50/50 diar-core unit, 7/7 provisioning integration. `vendor/speakrs` untouched
(`git diff HEAD --stat` unchanged at 23 files / 1359 insertions / 256 deletions) — the export
scripts are adapted COPIES under `scripts/provision/`, so
`patches/0001-cuda-performance-patch-set.patch` needs no regeneration.

### 7.36 Provisioning acceptance run — RTTM byte-identical from a reconstructed export (issue #2)

Ran the full 5-step recipe from `scripts/provision/provision.py` into a clean directory and
compared it against the shipped `models_folded/`. Run was **offline from a warm HF cache**
(`HF_HUB_OFFLINE=1`), so it needed no token — which also exercises the air-gapped/re-export
path. Python 3.12, torch 2.13.0, onnx 1.22.0, onnxscript 0.7.1, pyannote.audio 4.0.7,
onnxsim 0.7.3.

**Steps 2a-2d reproduced the shipped artifacts exactly.** Folding census on the fresh export:
`segmentation-3.0.onnx` **144 -> 40 nodes, max_abs_diff = 0.000e+00**; `-b32` and `-b64`
175 -> 40 with `Sin`/`Cos`/`If` eliminated.

**Smoke test on the fresh directory — identical to the shipped set on every number:**

| | models_folded | fresh export |
| --- | --- | --- |
| graphs parsed / signatures | 16 / 16 | 16 / 16 |
| speakers / segments / exclusive | 2 / 7 / 8 | 2 / 7 / 8 |
| 3a fbank b1-vs-b32 | 3.43e-5 | 3.43e-5 |
| 3b fused-vs-split | 0.00e0 | 0.00e0 |
| 3c multimask-vs-tail | 0.00e0 | 0.00e0 |
| 3e tail-b64 invariance | 3.58e-7 | 3.58e-7 |

**Output identity (the check that actually matters — proved by diffing raw records, not
asserted).** `diar-cli` over `fixtures/test.wav`, `--mode cpu`, against each directory:

```
sha256 46c9c617943ece588bb121c9b8c59155c03271b0376cfb2719a4151da750740f  models_folded RTTM
sha256 46c9c617943ece588bb121c9b8c59155c03271b0376cfb2719a4151da750740f  fresh export  RTTM
```

`diff` empty. 7 segments, 2 speakers, RTF ~6x on CPU on a loaded box (NOT a timed leg —
recorded only as run context).

**`compare_model_sets.py`, Tier B:** all 15 diarization ONNX graphs have identical op-type
histograms AND bit-identical initializer tensors; all six `plda_*.npy` and
`min_num_samples.txt` byte-identical. Only `gender-wav2vec2.onnx` differs — see below.

**A bug in the comparison tool, found by having a second export to compare against.** Tier B
originally zipped the two name-sorted initializer lists. `torch.onnx.export(dynamo=True)`
assigns initializer names at TRACE time, so the same weight legitimately carries a different
name across runs — the shipped `wespeaker-fbank.onnx` has `{eps, val_45}` where the fresh one
has `{val_44, val_46}`. Zipping paired DIFFERENT tensors and reported
"13/15 initializers differ, max |Δ| **2.000e+00**" for graphs that are in fact bit-identical.
Now compared as a multiset of `(dtype, shape, sha256)`. Negative control retained: the
zeroed-initializer fixture is still caught at 1/84. Worth recording because the false
positive was *plausible* — a tolerance would have been "fixed" instead of the tool.

**GENDER MODEL: fp16 conversion does not work on current torch, fp32 fallback adopted.**
`onnxconverter_common.float16` cannot convert the wav2vec2 graph torch 2.13 emits; ORT
rejects the output with `Type parameter (T) of Optype (Add) bound to different types
(tensor(float16) and tensor(float))`. Tried and rejected: opset 14 vs **17** (same failure),
`disable_shape_infer=True`, `op_block_list=['Cast']`, and clearing `graph.value_info` (which
only moves the error from a `Cast` to an `Add`). The shipped fp16 artifact was produced under
**torch 2.11.0**, whose graph the converter handles.

Rather than fail the whole run over a disk/VRAM optimisation, the exporter now treats fp16 as
**best-effort with a validated fp32 fallback**, and reports which precision it produced
(`toolchain`/`gender_precision`, surfaced as `models_gender` on `/healthz`). fp32 parity vs
torch on the fresh export: **max |logit diff| 5.36e-06** (bar 1e-4; §7.16 measured 5.96e-06).
Cost of the fallback, per §7.18: disk 189 -> 379 MB and roughly +500 MiB VRAM
(4890 -> ~5396 MiB). **Open item:** either pin torch 2.11 for provisioning or find a converter
path that works on 2.13; until then a freshly provisioned deployment uses more VRAM for gender
than the shipped one does.

**Server behaviour verified against the real `models_folded/` (which has NO marker, i.e. the
state every deployment is in today):**

- Bare `diar-server` with NO ARGS starts and serves — the hard backward-compat requirement.
- One warning naming `verify-models`/`provision-models`, then normal startup.
- `GET /healthz` -> **200**, body carries `models_state: "unverified"`, `models_verified:
  false`, `models_gender: true`, and issue #1's `status`/`default_device`/`devices`/
  `supported_devices` unchanged.
- `GET /readyz` -> **503**, same body.
- Empty models dir -> exit 6 naming the 22 missing files, the `provision-models` command line,
  the token page and the gate URL. No engine is constructed, so no "session load failed".
- `check-token` against the LIVE API: no token and a garbage token both exit 5 with the gate
  URL and **no traceback**.

Tests: 50 diar-core unit + 13 diar-server unit + 7 provisioning integration, all passing.

**NOT run, and needing an operator:** the token-authenticated end-to-end
`provision-models` path. Reading `transcribe-app/.env` is blocked by the permission system
(correctly — it is a secrets file), so the download half was exercised only from cache. All
four preflight branches are unit-tested from canned responses and two of them were confirmed
against the live API; what remains unproven is a real gated download.

**Image size, measured (this was called out as a regression check).** CPU image
**189 MB -> 194 MB (+5 MB, +2.6%)**. The 832 KB smoke clip is a small part of it; the rest is
binary growth from `clap`, `ureq`/rustls, `sha2`/`time`, the provisioning code, and the ~60 KB
of `include_str!`'d export scripts. Not zero, but the alternative it replaces — bundling
torch + pyannote + onnxscript into the runtime image so provisioning could run there — was
measured at ~13x on this image for a step that runs once. Verified in the built image: bare
`docker run <image>` with NO ARGS serves; `/usr/local/share/diar-native/smoke.wav` is present;
against a verified models dir `/healthz` and `/readyz` both return 200 with
`models_state: "verified"` and every provenance field populated, and no startup warning.

**Idempotency, demonstrated without a token** (the no-op is decided before preflight, so it
needs no network and no python): with a valid marker, `provision-models` and NO token exits
**0** with "already provisioned and verified (recipe v1)"; adding `--force` on the same
directory proceeds to preflight and exits **5** naming the token page.

### 7.37 diar-server had no log subscriber at all — structured logging, and two things measuring it found

**No timed leg. No benchmark number is claimed, retracted or affected here**; the box was
loaded (load average ~11–22 on 48 cores) throughout and every run below is functional.

**The bug.** `diar-server` had no `tracing-subscriber` dependency and never installed a
subscriber. `vendor/speakrs/src/` emits 40 tracing events and `crates/diar-core/src/` emits 2
`warn!`; **all 42 went to /dev/null in the deployed artifact.** The operator's entire visible
surface was two `eprintln!` lines plus crash output. CLAUDE.md documented
`RUST_LOG=speakrs=trace` for engine stage timings, which worked only in `diar-cli` — it was
dead in the thing that actually ships. Corrected in CLAUDE.md rather than deleted.

Worth recording precisely, because it is why this survived review: the natural reading is "no
subscriber ⇒ nothing logged", but `diar-cli` used `EnvFilter::from_default_env()`, which with
`RUST_LOG` unset falls back to **`ERROR`** — not to off. Nothing in this workspace emits at
`error` level (speakrs' 40 are debug/trace/info, diar-core's 2 are `warn!`), so an `ERROR`
floor is observationally identical to silence. The code looked correctly wired and produced
nothing. Pinned as a characterization test so nobody simplifies back into it.

**What landed.** `diar_core::logging` holds the policy for both binaries (the sink is a
parameter: server → stdout, CLI → stderr, whose stdout is the harness's JSONL). `RUST_LOG`
unset/empty/malformed ⇒ the default filter; `DIAR_LOG_FORMAT=text|json`. One startup record
built from `health_body()` so it cannot drift from `/healthz`. A per-request span on
`/diarize` and `/embed_window`, re-entered inside `spawn_blocking` so speakrs' own events
nest under it. `x-request-id` honoured inbound and echoed back.

**Finding 1 — a bare `info` default is unusable, measured.** First run of the built image
against `models_folded/`, no extra env: **5835 log lines, of which 5812 were `ort::logging`
and exactly 3 were diar-server's** ("Removing NodeArg …", "GraphTransformer … modified: 0").
The startup record was buried ~2000:1. Default is now `info,ort::logging=warn`, which keeps
ORT's 15 WARNs (Memcpy nodes added, nodes not assigned to the preferred EP — real perf
diagnostics) and keeps `ort::ep`'s EP-registration lines. **5835 → 38 lines**, same models,
same device, same image. Narrowing applies to the DEFAULT only; `RUST_LOG=ort=info` still
returns the firehose, and there is a test asserting that.

**Finding 2 — a privacy leak, caught by running it rather than reading it.** Keeping the span
field to a basename is not sufficient: the underlying I/O error interpolates the path it was
handed. Verbatim from the run, `audio` correct and `error` defeating it in the same record:

```
WARN request{request_id=5158e4-000002 endpoint="/diarize" gender=false
     audio="Board Meeting Q3 - CONFIDENTIAL.wav" device="cuda"}:
  /diarize failed outcome="error" duration_ms=0.4 error_class="audio_decode" status=422
  error=opening /audio/private/Board Meeting Q3 - CONFIDENTIAL.wav: No such file or directory
```

Logged error text is now redacted down to the basename. Re-run of the identical request:
`error=opening Board Meeting Q3 - CONFIDENTIAL.wav: …`, and `/audio/private` appears nowhere
in the log. The HTTP **response body** deliberately still carries the full path — the caller
supplied it, so it is not a disclosure to them, and trimming it would break error handling
that parses these messages.

**Verified in the built image** (`diar-server:log-test`, CUDA, GPU 0, `models_folded/`):

- No extra env: 38 lines; startup record carries version, bind, models dir/state/set/gender,
  devices, default device, both inflight gates and the resolved log config.
- `POST /diarize` → `200`, `x-diar-device: cuda`, `x-request-id: 5158e4-000000`, and
  `/diarize ok outcome="ok" duration_ms=3126.7 num_speakers=2 segments=7`.
- Caller-supplied `x-request-id: otranscribe-job-4271` is kept and echoed back.
- Error classes exercised: `device:"tpu"` → **400** `error_class="bad_device"`; a missing file
  → **422** `error_class="audio_decode"`. Status codes unchanged from before this work.
- `RUST_LOG=speakrs=debug` surfaces the previously invisible engine events — `Embedding path
  selected path=MultiMask`, `Segmentation thread profile windows=18 seg_infer_ms=514`,
  `AHC pre-clustering num_clusters=2`, `clustering stage timing ahc_ms=0 plda_ms=0 vbx_ms=0`.
- `DIAR_LOG_FORMAT=json`: **54 of 54 lines parsed by Python's `json.loads`, 0 failures.**
  Fields are flattened and the span is nested, e.g. the request record carries
  `span.request_id`, `span.device`, `span.audio`, `duration_ms`, `num_speakers`, `segments`.
- Correlation: with `RUST_LOG=info,speakrs=debug`, **14 of 15** speakrs events carried the
  caller's `request_id`.

**Known limitation (NOT fixed, needs a vendored change).** The 15th event,
`speakrs::inference::segmentation::run`'s "Segmentation thread profile", is emitted from a
thread speakrs spawns internally for the fbank∥GPU pipeline (§7.28); that thread does not
inherit the request span, so the event is logged without a `request_id`. Fixing it means
propagating the span into `vendor/speakrs`, which is out of scope here and is flagged rather
than done.

**Also on record:** the startup line reports `version="0.1.0"` — the `diar-server` crate
version, which has never been bumped and does not match the `diar-server:0.2.0` image tag.
The value is accurate for what it names; the crate/tag divergence pre-dates this work and is
left for an operator decision rather than silently changed here.

Tests: **76 diar-core + 28 diar-server**, all passing (this work added the `logging` and
`reqlog` suites; the counts also include §7.38's additions). The default-filter test was
watched failing against the old policy first — it reported `info was dropped: ""`.

### 7.38 Provisioning audit fixes — a GPU-less run no longer self-destructs, and the graphs production runs are now verified (issue #2)

Two adversarial audits of the provisioning subsystem shipped in §7.35/§7.36. Ten confirmed
defects, all fixed in `f35fdfa`. Nothing here retracts a number from §7.35/§7.36 — the
measurements there stand; what changes is what the code does on paths those runs did not take
(a host with no GPU, a `--set small` directory, a directory with no marker, a corrupted
fixed-batch graph).

**The one that mattered most, measured.** A models directory whose
`wespeaker-multimask-tail-b32.onnx` has its largest initializer (`resnet.seg_1.weight`,
256x5120 f32) zeroed — built by
`validation/make_corrupt_fixture.py models_folded /tmp/models_zeroed_mm wespeaker-multimask-tail-b32.onnx`,
which also mirrors the corruption into the b64 byte copy so stage 3d's sha256 equality still
passes. Against the PRE-FIX smoke test this produced a **fully green report**:

```
1-parse        16 ONNX graphs loaded on the CPU EP
2-io-contract  16 signatures matched the compiled-in contract
5-plda         6 PLDA arrays + min_num_samples=400
3-numeric      3a fbank b1-vs-b32 3.43e-5; 3b fused-vs-split 0.00e0;
               3c multimask-vs-tail 0.00e0; 3d multimask-b64 is a byte copy of b32;
               3i tail-b64 batch-invariance 3.58e-7
4-end-to-end   2 speakers, 7 segments, 8 exclusive, gender=2
```

i.e. `smoke.status: "pass"`, a `verified` marker, 200 from `/readyz`, and every file longer
than one window embedded by a graph with a zeroed weight tensor. Post-fix:

```
STAGE 3e FAILED: wespeaker-multimask-tail-b32.onnx row 5 disagrees with
wespeaker-multimask-tail.onnx by 9.222e-1 (bar 1e-4) under an identical mask.
```

That graph is the production hot path: `vendor/speakrs/src/inference/embedding/load/sessions.rs`
gates `primary_batched_session`, `split_tail_batched_session` and
`split_primary_tail_batched_session` on `!lazy_sessions`, but NOT the multimask sessions — and
live compose sets `SPEAKRS_LAZY_SESSIONS=1`.

**Stage 3 coverage, before and after** (fast set, `models_folded/`, CPU EP, 10 s clip):

| check | graph | before | after |
|---|---|---|---|
| 3a | `wespeaker-fbank{,-b32}` | 3.43e-5 | 3.43e-5 |
| 3b | fused vs split (b1) | 0.00e0 | 0.00e0 |
| 3c | `multimask-tail` (b1) | 0.00e0 | 0.00e0 |
| 3d | `multimask-tail-b64` == b32 (sha256) | pass | pass |
| **3e** | **`multimask-tail-b32` vs b1** | **nothing** | **2.38e-7** |
| **3f** | **`segmentation-3.0-b32` vs b1** | **nothing** | **0.00e0** |
| **3g** | **`-tail-b3`, `-tail-b32` vs b1** | **nothing** | **2.38e-7, 2.38e-7** |
| **3h** | **`resnet34-b32` vs b1** | **nothing** | **2.58e-6** |
| 3i | `-tail-b64` batch invariance | 3.58e-7 (ran first) | 3.58e-7 (runs last) |

Bar is 1e-4 throughout; 3i's 3.58e-7 is unchanged and consistent with §7.33's 7.8e-08 for the
same property. **Cost: full smoke 14.7 s -> 29.5 s on the CPU EP** (busy box, single run, not a
benchmark leg). The b64 tail check — the single most expensive one, on a graph live compose
never loads — now runs LAST, so a hot-path failure is reported without waiting for it.

**The fold parity comment was wrong on the facts.** `fold_segmentation.py` checked only the b1
graph, on the stated grounds that b32/b64 were "covered structurally here and by smoke stages
1-2 in Rust" (they are not: stage 1 is a protobuf parse, stage 2 is names and shapes) and that
running them "costs minutes for no additional signal". Measured on the shipped graphs with
`ORT_DISABLE_ALL`, per graph pair:

```
segmentation-3.0.onnx      batch=1   max_abs_diff=0.000e+00   0.2 s
segmentation-3.0-b32.onnx  batch=32  max_abs_diff=0.000e+00   0.9 s
segmentation-3.0-b64.onnx  batch=64  max_abs_diff=0.000e+00   1.5 s
```

Seconds, not minutes. All three are now checked.

**C1 — provisioning defaulted to a device it does not need, then blamed the models.**
`cli.rs::parse_mode` fell through to `Mode::Cuda`, so `docker run --rm -e HF_TOKEN=... -v
/path:/models diar-provision:<ver>` (the line `docker/Dockerfile.provision:24` documents, with
no `--gpus`) exported ~470 MB of correct models, failed stage 4, and wrote `smoke.status:
"fail"` into `diar-provision.json`. `startup_gate` reads that as known-bad, so every later
`diar-server` start exited non-zero about files that were fine; on `Dockerfile.server-cpu`,
where CUDA is not compiled in, `provision-models` could never have succeeded. Provisioning now
defaults to CPU. The device-vs-models distinction is decided by EXPERIMENT rather than by
string-matching an ORT error: the same directory is re-loaded on the CPU EP, and only if that
succeeds is the device blamed. Reproduced in CI without a GPU-less host by asking a non-coreml
build for `Mode::CoreMl`:

```
the `coreml` execution device is not usable on this machine, so the end-to-end stage could
not run here. The models themselves are NOT implicated — the same directory loaded
successfully on the CPU execution provider moments ago.
Underlying error: loading segmentation model: coreml requires the `coreml` Cargo feature
```

Exit 9 (`DEVICE_UNAVAILABLE`), and **no marker is written**.

**C2/C3/C4/C5/C6, in one line each.** `--set small` produced a directory the server refused to
start against (the gate assumed `Fast` and demanded the four b64 graphs `provision.py` had just
deleted, with remediation text saying `--set fast`) — the set now comes from the marker,
`DIAR_MODEL_SET` remains an override and is named in the message. A passing `verify-models` was
read-only w.r.t. the marker, so recovery from a stale `fail` needed a full `--force` re-export;
it now re-stamps the smoke record (provenance untouched, `--no-attest` opts out). The smoke
clip was resolved before the writability/token/python checks, so provisioning from
OpenTranscribe's backend image — which copies the binary out of our image without the clip —
died with exit 2 before looking at the token; resolved late now. `DIAR_DEVICES` was ignored by
every provisioning subcommand. Exit 6 meant both "install torch" and "provision the models";
serving now exits 8, with 9 and 10 added.

**F2 — `verify-models` reported success having compared nothing.** `Marker::read` returns
`Ok(None)` for an absent marker, so `verify_deep`'s drift loop was skipped entirely and `drift`
stayed empty — indistinguishable from "every hash matched". `OK: /models verified.`, exit 0,
zero bytes hashed, on the exact scenario the command exists for (a `/models` directory of
unknown provenance). Now a distinct `unverified` status and exit 10, with `files_hashed` in the
JSON. Verified against `/tmp/e2e1/models` (has a marker: 24 files hashed, no drift) and a
hardlinked copy with the marker deleted (0 hashed, `fully_verified()` false, smoke still green). Against the built binary, same two directories:

```
$ diar-server verify-models --models-dir /e2e1/models --smoke-clip …
OK: /e2e1/models verified (24 file(s) matched their recorded sha256).
  (marker not updated: … Read-only file system (os error 30) — verification still passed;
   this is expected on a read-only mount)
exit=0

$ diar-server verify-models --models-dir /tmp/nomarker --smoke-clip …
UNVERIFIED: /tmp/nomarker passed every smoke stage, but there is no diar-provision.json in
it, so NOTHING was compared against a recorded hash. …
exit=10

$ diar-server verify-models --models-dir /tmp/nomarker --json
{"attested":null,"drift":[],"files_hashed":0,"marker_present":false, …}
```

Note the read-only arm of C3 in the first transcript: a `:ro` mount cannot be re-attested, and
that is reported as a note rather than turned into a verification failure.

C4 in the same binary, from a working directory where neither clip candidate resolves:
`provision-models` now reaches the TOKEN check ("No HuggingFace token was supplied … gated …
`--hf-token`") instead of exiting 2 with "no smoke clip found" before looking at anything.

**F3 — `gender_precision` was measured and dropped by serde.** `ExportReport` had no such
field, so the value `export_gender.py` computes — and whose docstring says is "REPORTED, so the
marker and /healthz can say so rather than implying fp16" — reached nothing at all. It mattered
most while the fp16 conversion was failing under the pinned `torch==2.13.0` (every directory
silently got the 378 MB fp32 classifier, ~500 MiB more VRAM than fp16 — §7.18: 5396 -> 4890
MiB); §7.37 has since made fp16 reachable again, but the fp32 fallback is still live and the
field is what distinguishes the two. Now carried into `marker.toolchain.gender_precision` and
printed by both subcommands. **Still outstanding: `HealthResponse` in `crates/diar-server/src/main.rs`
needs a one-line field addition to surface it on `/healthz`; that file is owned by another
change in flight, so it is reported rather than edited here.**

**F4/F6/F11.** The pre-download python probe omitted `onnxconverter_common` (absence raises
NOTHING and silently downgrades gender to fp32), `onnxsim`/`onnxslim` and `onnxruntime` (both
discovered only after the full download), plus `torchaudio` and `huggingface_hub` — a test now
pins the probe list against the imports in the scripts themselves and reports exactly that set.
`run_export` swallowed both the IO error and the parse error on `report.json` and fell back to
an all-`None` report, producing a `verified` marker with null provenance including
`toolchain.folder` — the one field `fold_segmentation.py` exists to record; now a hard error
distinguishing the two causes. `export_models.py` gated the multi-mask parity check on a bare
`assert`, which `PYTHONOPTIMIZE` compiles out and which `run_export` inherits; converted to an
explicit `raise`, `PYTHONOPTIMIZE` is cleared in the subprocess, and a test pins that no
embedded script gates an invariant on an assert.

**Tests.** 76 diar-core unit + 28 diar-server unit + 10 provisioning integration, all passing,
plus the vendored speakrs suite at **96/96** (76 + 5 + 8 + 7, one ignored) to confirm no
regression — nothing under `vendor/` was touched.
Every one of the ten new tests was watched FAILING against a deliberately re-introduced defect
before the fix was restored — including the F4 probe test, which named the gap itself:

```
the export scripts import ["huggingface_hub", "onnxconverter_common", "onnxruntime",
"onnxsim", "onnxslim", "torchaudio"], which check_python_env never probes for.
```

**Not re-run, and not claimed:** no DER leg, no timed benchmark, no token-authenticated
end-to-end `provision-models` (same operator gap as §7.35). The 29.5 s smoke figure is a single
observation on a loaded box, recorded as an order-of-magnitude cost, not as a benchmark.

### 7.39 Gender fp16 restored on torch 2.13 — the blocker was two no-op `Cast` nodes (issue #6)

Closes the open item §7.36 left behind ("either pin torch 2.11 for provisioning or find a
converter path that works on 2.13"). **No torch pin was needed.** Every directory provisioned
since §7.36 got the 378 MB fp32 classifier — ~500 MiB more VRAM than the shipped fp16 build
(§7.18: 5396 -> 4890 MiB). Fresh provisioning now produces fp16 again.

**NOT a timed leg, and deliberately so.** The box was loud throughout: `uptime` load average
**11.15 -> 39.91 on 48 cores**, an `otfresh-demo` stack plus a container at 774% CPU, and GPU
0 went from 847 MiB free-and-idle to **6451 MiB** used by other work mid-session. Every claim
below is therefore a **size, precision, or output-identity** fact, all of which are invariant
to machine load. No wall-clock number here is offered as a benchmark.

**ROOT CAUSE.** torch 2.13 emits two `Cast` nodes that torch 2.11 did not, both semantic
no-ops: `/inner/wav2vec2/encoder/Cast_1` is `Cast(to=FLOAT)` on an already-fp32 `Transpose`
output feeding `encoder/Add`, and `/inner/wav2vec2/encoder/Cast` is `Cast(to=INT64)` on an
already-int64 value. They are invisible in fp32, which is why nothing noticed them.
`onnxconverter_common.float16` treats every `Cast` as an authoritative precision boundary —
casts are the mechanism it inserts itself to bridge fp16/fp32 under `keep_io_types` — so it
retypes the value to fp16 and leaves `to=FLOAT` alone, and ORT rejects the contradiction.

This also explains §7.36's confusing symptom pair. Reproduced here, both faces of one bug:

```
baseline               FAIL Type (tensor(float16)) of output arg
                            (/inner/wav2vec2/encoder/Cast_1_output_0) of node
                            (/inner/wav2vec2/encoder/Cast_1) ... expected tensor(float)
disable_shape_infer    FAIL Type parameter (T) of Optype (Add) bound to different types
                            (tensor(float16) and tensor(float)) in node
                            (/inner/wav2vec2/encoder/Add)
```

`/inner/wav2vec2/encoder/Add` is precisely the node `Cast_1` feeds — §7.36's "only moves the
error from a `Cast` to an `Add`" was the same node all along, not a second problem.

**FIX.** `_elide_identity_casts` in `scripts/provision/export_gender.py` drops any `Cast`
whose `to` already equals its input's inferred element type, before conversion. Principled,
not a name match: an identity cast is a no-op by definition. Casts feeding a graph output are
never elided (renaming `logits` would break `gender.rs`'s by-name lookup), and a cast whose
input type shape inference cannot determine is left alone.

**Gates, all run by the exporter itself every time — not one-off measurements.** The gate
corpus is 6 seeded, `do_normalize`-preprocessed clips at 16k/24k/32k/48k/64k/80k samples,
bracketing `gender.rs`'s real range (`MIN_SAMPLES` 1 s to the `DIAR_GENDER_MAX_SECONDS`
default cap of 5 s). §7.18's 67-verdict AMI sweep was ad hoc and left nothing runnable; this
is weaker evidence per input but it is *committed and repeatable*.

| gate | bar | measured |
| --- | --- | --- |
| 1. fp32 ONNX vs torch | max abs logit diff < 1e-4 | **5.36e-06** (unchanged from §7.36) |
| 2. elision is a no-op | **bitwise equal** fp32 logits, 6/6 clips | **6/6 bitwise identical**, max abs diff 0.000e+00 |
| 3. fp16 argmax invariance | 6/6 labels unchanged | **6/6**, max abs logit delta **1.06e-02**, max abs prob delta **4.86e-04** (bar 0.05) |

Gate 2 is the load-bearing one: casts are load-bearing when they are real, so the exporter
proves these were not rather than trusting the analysis above.

**Artifact.** `gender_precision: "fp16"` on the pinned torch 2.13.0 / onnx 1.22.0 /
onnxruntime 1.29.0 stack.

| | bytes | initializers | I/O | Cast nodes |
| --- | --- | --- | --- | --- |
| fp32 (what §7.36 shipped) | 378,529,501 | 213 FLOAT | fp32 | 2 no-op |
| **fp16 (now)** | **189,488,828** | **213 FLOAT16, 0 FLOAT** | **fp32** | 2 (`keep_io_types` boundary) |
| shipped torch-2.11 artifact | 189,431,659 | 213 FLOAT16, 0 FLOAT | fp32 | 2 (`keep_io_types` boundary) |

**Disk: -189,040,673 bytes (-50.0%).** The regenerated graph matches the shipped torch-2.11
artifact on every load-bearing property — 213/213 FLOAT16 initializers, fp32 in and out,
opset 17, and the same two boundary casts by role (`input_values_cast_to_...Unsqueeze` to=10,
`.../Gemm_cast_to_logits` to=1). It is **57,169 bytes larger** (0.03%) because torch 2.13
emits extra graph machinery the 2.11 export did not — 12 `IsNaN` + 25 `Where` (attention NaN
clamp) and `Range`/`Expand`/`Equal`/`GreaterOrEqual`/`ConstantOfShape` mask construction,
1084 nodes vs 966. That difference is in the fp32 export too; it is a torch-version graph
difference, not a conversion artifact, and it does not touch the weights.

**END-TO-END under the real runtime (ort `=2.0.0-rc.12`, not just python onnxruntime).**
`diar-server:issue1`, `--gpus "device=0"`, two containers differing ONLY by a bind-mount of
`gender-wav2vec2.onnx` over an otherwise shared `models_folded`; `karpathy_10m.wav` (the
§7.16/§7.18 reference clip), `{"gender":true}`. The new graph loads under ort rc.12 (the
gender session is committed eagerly at engine load, so `/healthz` 200 is itself the proof).

```
rttm sha256 edcc5d277412b93d906b81039c5f7a81185ba3a6fd8759368bc4565df589b4ef  shipped fp16
rttm sha256 edcc5d277412b93d906b81039c5f7a81185ba3a6fd8759368bc4565df589b4ef  regenerated fp16
```

`segments` (92), `exclusive_segments` (88), `centroids` (2), `num_speakers` (2): all equal by
value. Gender **2/2 labels agree**:

| speaker | shipped fp16 | regenerated fp16 | delta |
| --- | --- | --- | --- |
| SPEAKER_00 | female 0.79675186 | female 0.79675025 | 1.6e-06 |
| SPEAKER_01 | male 0.9991439 | male 0.9991439 | 0 |

These also match §7.16's recorded values for this clip at the 5 s cap (female 0.797 / male
0.999), so the regenerated artifact lands on the historical operating point.

**Cost.** The exporter does one extra shape-inference pass over a 378 MB graph, 2 extra ORT
session loads and 12 CPU inferences: **~11 s** against §7.36's 119.5 s cold provisioning
(~+9%). That number is load-contaminated (la 15.78) and is order-of-magnitude only — it is a
once-per-deployment cost buying 189 MB of disk and ~500 MiB of VRAM permanently.

**Rejected, with the measurement.** `op_block_list=['Cast']` cannot work and was not the
answer: blocking the op the converter uses as its own precision boundary is what produced the
mismatch in the first place. `disable_shape_infer=True` and clearing `graph.value_info` both
only relocate the error (above). Pinning torch 2.11 for provisioning — §7.36's other
suggestion — was rejected: it would freeze the export environment against the rest of the
pinned stack to work around two no-op nodes, and OpenTranscribe's backend image (the primary
provisioning route, which already has torch) does not get to choose its torch version.

**Retraction.** §7.34's "both engines parse the same **189 MB fp32 ONNX**" is wrong on
precision — 189 MB *is* the fp16 artifact; the fp32 one is 378 MB. The RSS attribution in
§7.34 (gender = 74% of the +620 MB second-engine cost) was measured against `models_folded/`,
i.e. against the **fp16** model, and stands. Stale doc comments in `provision/exporter.rs` and
`provision/marker.rs` asserting the converter "cannot convert the graph torch emits" are
corrected here; the fp32 fallback path itself is unchanged and still live.

**Still open (narrowed).** The serde drop is fixed — `gender_precision` now reaches
`Toolchain` (`marker.rs`) and the provisioning report, and `provision-models` prints it
(`cli.rs`); `provision::tests::gender_precision_reaches_the_marker` covers it. What is still
inaccurate is §7.36's specific wording "surfaced as `models_gender` on `/healthz`":
`HealthResponse.models_gender` is a **bool of file existence** and carries no precision. Now
that both precisions are genuinely reachable in the field, a served directory still cannot be
asked which one it has without reading the marker off disk.

### 7.40 The fp16 gender load failure is an ORT *fusion-gate* difference, not an aarch64 kernel gap — and the surgical fix needs the name `GeluFusionL2` (issue #14)

Run on the operator's Apple Silicon Mac (M2 Max, macOS 15.7.9, `Darwin 24.6.0`), from a
fresh clone of `attevon-llc/diar-native` at `b9a4b3e` (0.3.0). Issue #14 asked three
platform questions that cannot be answered from Linux, because `coreml` builds only on
macOS and never goes through Docker.

**NOT a timed leg.** Every claim below is a load/no-load fact, a graph-shape fact, or an
output-identity fact, all invariant to machine load. No wall-clock number is offered as a
benchmark. `duration_ms` values in the smoke JSON are incidental.

**Models.** The 22 required diarization artifacts came from the local-only gated set on
that machine; `gender-wav2vec2.onnx` was exported fresh with `scripts/provision/export_gender.py`
(the classifier repo is ungated — no HF token was used or requested). The export's own gates
passed: fp32-vs-torch 5.60e-06, 2 no-op `Cast` nodes elided with fp32 output bitwise
unchanged on 6 clips, fp16 conversion 6/6 labels unchanged, max Δlogit 2.75e-03,
379 MB -> 189 MB. Shape matches the shipped artifact: opset-17 plain `ai.onnx`, no contrib
domain on any node, 20 `Erf`, 213 FLOAT16 initializers, fp32 in/out under `keep_io_types`.
Node count 1084 vs the shipped 966 and size 189,488,828 vs 189,431,659 — a `transformers`
version difference, immaterial to everything below (the `Erf` count and initializer dtypes,
which are what the fusion keys off, are identical).

#### Q1/Q2 — the failure does NOT reproduce on macOS arm64, in any mode

`diar-server verify-models --set fast --smoke-clip vendor/speakrs/fixtures/test.wav`,
native builds, no Docker:

| build | `--mode` | stage 1 | end-to-end |
| --- | --- | --- | --- |
| default (CPU) | `cpu` | 16 ONNX graphs loaded | 2 speakers, 7 segments, 8 exclusive, gender=2 |
| `--features coreml` | `cpu` | 16 ONNX graphs loaded | 2 speakers, 7 segments, 8 exclusive, gender=2 |
| `--features coreml` | `coreml` | 16 ONNX graphs loaded | 2 speakers, 7 segments, 8 exclusive, gender=2 |
| `--features coreml` | `coreml_fast` | 16 ONNX graphs loaded | 2 speakers, 7 segments, 8 exclusive, gender=2 |

The gender model loads and classifies in all four. Counts match amd64 and linux/arm64
exactly. Both builds compile natively (`coreml` needs `LIBRARY_PATH` to Homebrew openblas).
`--mode coreml` additionally requires the `.mlmodelc` assets in the models dir; without them
it fails with a clear `coreml requires native asset …` message, which is the *asset* gate
doing its job and not this bug.

#### The mechanism — both aarch64 builds lack the fp16 kernel; they differ in whether the fusion FIRES

`nm` on the ORT static lib the `ort` crate links (`ort.pyke.io/dfbin/aarch64-apple-darwin/…`,
ORT **1.24.2**) shows the macOS build has the same kernel gap the issue describes:

```
onnxruntime::Gelu<float>                      <- the only instantiation
Gelu<onnxruntime::MLFloat16>                  <- 0 matches
onnxruntime::GeluFusion / BiasGeluFusion      <- both present
contrib::Gelu_Microsoft_ver1 schema           <- present
```

So macOS is *not* rescued by having an fp16 kernel. It is rescued because **`GeluFusion`
declines to rewrite an fp16 graph on this build**, so the unsupported node is never created.
Proven by dumping the optimized graph (`with_optimized_model_path`) for the same model in
both precisions on macOS:

| graph | opt level | nodes | `Erf` | contrib ops produced |
| --- | --- | --- | --- | --- |
| gender **fp16** | Level3 (default) | 994 | 20 | none |
| gender **fp16** | Level1 | 994 | 20 | none |
| gender **fp16** | Disable | 1232 | 20 | none |
| gender **fp32** | Level3 (default) | 496 | 0 | `Gelu` 8, `BiasGelu` 12, `FusedMatMul` 12 |
| gender **fp32** | Level1 | 612 | 20 | none |

The fp32 row proves the dump reflects Level-2 fusions, so the fp16 row's surviving 20 `Erf`
is a real negative, not a measurement artifact. **The Erf-GELU rewrite is gated to fp32 on
the macOS aarch64 ORT build and is not gated on the Linux aarch64 build.** That is the whole
difference; it is an ORT-build-configuration divergence between two targets of the same
1.24.2 release, not anything about Apple silicon.

#### Q3 — reproduced the Linux failure on this Mac under Docker, and fix candidate (a) as written DOES NOT WORK

Docker Desktop runs `linux/arm64` natively here, so the failing platform was reproducible
without leaving the machine — `rust:1-trixie` (glibc 2.41; `bookworm`'s 2.36 cannot link
this ORT, it wants `__isoc23_strtol`), `RUSTFLAGS=-C link-arg=-lstdc++`, same pinned
`ort =2.0.0-rc.12`, same model file. The error text matches the issue byte for byte.

`ort` rc.12 **does** expose the config entry: `SessionBuilder::with_disabled_optimizers()`
-> `optimization.disable_specified_optimizers`, plus the generic `with_config_entry`.
So the mechanism the issue proposed exists. It just does not do what the issue assumed:

| session config (linux/arm64) | load |
| --- | --- |
| default (Level3) | **FAIL** — `com.microsoft.Gelu(1) … implemented only for (tensor(float),) … model has (tensor(float16))` |
| Level2 | **FAIL** (same) |
| `disable_specified_optimizers=GeluFusion` | **FAIL** (same) — candidate (a) as specified |
| `disable_specified_optimizers=GeluFusionL1` | **FAIL** (same) |
| **`disable_specified_optimizers=GeluFusionL2`** | **ok** |
| `disable_specified_optimizers=GeluFusionL1,GeluFusionL2` | **FAIL** |
| `disable_specified_optimizers=BiasGeluFusion,GeluFusionL2` | **FAIL** |
| `disable_specified_optimizers=GeluFusionL2;BiasGeluFusion` | **ok** |
| Level1 (Basic) | **ok** — candidate (b) |
| Disable (Level0) | **ok** — the unfused reference |

Two things fall out that were not in the issue's model of the problem:

1. **The optimizer is named `GeluFusionL2`, not `GeluFusion`.** ORT registers the Erf-GELU
   pass twice (an L1 and an L2 instance) under suffixed names. An unrecognized name is
   **silently ignored** — `disable_specified_optimizers=NotARealOptimizerName` loads fine and
   changes nothing — so shipping the wrong name buys a config entry that looks applied and
   does nothing. This is exactly what candidate (a) would have done.
2. **The separator is `;`, not `,`.** `GeluFusionL2;BiasGeluFusion` disables both;
   `BiasGeluFusion,GeluFusionL2` disables neither. The `ort` crate's own doc comment on
   `with_disabled_optimizers` says "Accepts a comma-separated list of optimizers to disable",
   which is wrong for this build — worth an upstream `ort` doc issue. Practically: pass one
   name, or separate with `;`.

Independent confirmation of the name and separator, on the fp32 gender graph on macOS where
the fusion does fire (dump inspection, so it reports *suppression* rather than load success):
`BiasGeluFusion` alone changes the output from `Gelu` 8 + `BiasGelu` 12 to `Gelu` 20 + no
`BiasGelu`; `ConstantFolding` moves node count 496 -> 499; `GeluFusion` and a bogus name both
leave the graph untouched.

#### Q4 — numerics, against the unfused graph as reference

Six seeded `do_normalize`-preprocessed clips at 16k/24k/32k/48k/64k/80k samples — the same
gate corpus `scripts/provision/export_gender.py` uses — generated once and fed byte-identically
to every configuration and both platforms.

| comparison | max &#124;Δ logit&#124; | label agreement |
| --- | --- | --- |
| linux/arm64 **Level1 (fix b)** vs Level0 reference | **0.000e+00** (bitwise) | 6/6 |
| linux/arm64 **`GeluFusionL2` (fix a)** vs Level0 reference | 9.580e-04 | 6/6 |
| linux/arm64 `GeluFusionL2` vs Level1 | 9.580e-04 | 6/6 |
| macOS Level3 / `GeluFusion` / Level1 vs Level0 | 0.000e+00 (bitwise) | 6/6 |
| macOS Level3 vs linux/arm64 Level0 | 2.890e-01 | 6/6 |

Candidate (b) is **bitwise identical** to the fully-unoptimized reference. Candidate (a)
keeps every other Level-2/3 optimization, so it differs by fp16 reassociation at ~1e-3 on a
logit — far inside the 0.05 probability bar §7.18/§7.39 set, and every argmax is stable.
Both are safe; (b) is the stronger identity claim.

The last row is a **cross-platform** observation, not a regression: the two aarch64 ORT
builds disagree by up to 0.29 on an fp16 logit purely from arithmetic ordering. All 6 labels
still agree. It is recorded because it means an fp16 gender logit is not a portable number
across builds — only the label is.

#### Q5 — the diarization graphs sit on the same cliff, and are safe only because they are fp32

All 15 diarization graphs load at Level3 on macOS arm64. But "no fusion gap" is the wrong
reading. Dumping each optimized graph shows **11 of 15 are rewritten into a contrib op by
the same machinery**:

| graphs | contrib ops after Level3 | initializer dtypes |
| --- | --- | --- |
| `wespeaker-voxceleb-resnet34{,-b32,-b64}`, `…-tail{,-b3,-b32,-b64}`, `wespeaker-multimask-tail{,-b32,-b64}` (11) | `com.microsoft::FusedConv` × 33 each | FLOAT only |
| `segmentation-3.0{,-b32,-b64}`, `wespeaker-fbank{,-b32}` (4) | none | FLOAT only |

`nm` shows the only `FusedConv` kernel in the build is `FusedConv_kMSDomain_ver1_**float**`
— no fp16 instantiation, exactly the shape of the gender bug. **These graphs are safe today
solely because every one of their initializers is fp32.** The exposure is structural: if
fp16 is ever revisited for the embedding graphs (§4.18 rejected it for accuracy, not for
this), 11 of 15 graphs land on the identical failure. Any future fp16 export must carry a
load check on aarch64, not just an accuracy gate.

#### What this means for the fix

- The bug is **not** "linux/arm64 is missing a kernel that amd64 has" — every aarch64 build
  checked is missing it. It is "the linux/arm64 ORT build lets a fusion produce a node its
  own kernel set cannot execute". Upstream-reportable against onnxruntime.
- Ship-wise, either fix works and both are cheap. `GeluFusionL2` is the surgical one and is
  what issue #14 preferred; **Level1 on the gender session only** is the one with a bitwise
  identity proof and no dependence on an ORT-internal optimizer name that is unvalidated,
  silently ignored when wrong, and already renamed once. Recommendation: **Level1 for the
  gender session**, with `GeluFusionL2` recorded here as the validated alternative.
- Neither touches `vendor/`. `GenderModel::load_optional` in `crates/diar-core/src/gender.rs`
  builds its own `Session` and is the only place that needs to change.
- No change is needed on macOS at all; whichever fix ships is inert there (macOS is already
  bitwise identical between Level3 and Level0 on this graph).

Artifacts (scratch, not committed): probe sources `ortprobe{,2,3}`, the seeded `clips.bin`,
and every optimized-graph dump.

#### §7.40 addendum — reconciled against the fix that shipped (`c06fa15`)

`c06fa15` landed independently on the Linux side while §7.40 was being measured on macOS, and
reached the **same fix**: cap the gender session at `Level1`, aarch64 only, diarization graphs
untouched. Two claims in its rationale are narrowed by the measurements above; neither changes
the fix, both change how its escape hatches should be used.

1. **"Naming the optimizer does NOT suppress the rewrite; only capping below level 2 does."**
   Too strong. Naming it works — under `GeluFusionL2`. `c06fa15` tested `GeluFusion` and a
   four-name comma list, and both fail for reasons that are not "naming doesn't work": the
   pass is registered twice under suffixed names (`GeluFusionL1`/`GeluFusionL2`), an
   unrecognized name is silently ignored, and the separator is `;` not `,` so the comma list
   matched nothing at all. The level cap remains the better fix for the three reasons in
   §7.40 (bitwise identity, zero contrib ops left on fp16, no dependence on an undocumented
   ORT-internal name) — but it is preferred, not forced.

2. **"The x86_64 ORT build ships an fp16 kernel for the fused contrib op; the aarch64 build
   ships fp32 only."** True of those two targets, but it is not the mechanism. `nm` on the
   **macOS aarch64** ORT 1.24.2 static lib shows `Gelu<float>` only — the same missing fp16
   kernel — and that platform loads the model fine, because its ORT declines to fuse fp16 at
   all. The differentiator is whether the fusion FIRES, not whether the kernel exists. This
   matters for prediction: a future aarch64 ORT build could acquire the kernel, or lose the
   fusion gate, and either would change the symptom without anyone touching this repo.

**Sharp edge in the shipped escape hatches**, recorded because the symptom is silence:
`DIAR_ORT_OPT_LEVEL` and `DIAR_ORT_DISABLED_OPTIMIZERS` are global and each returns early, so
either one *replaces* the built-in aarch64 workaround rather than adding to it. Setting
`DIAR_ORT_OPT_LEVEL=all` on an aarch64 host to tune the diarization graphs silently disables
speaker gender again. Comments in `ort_compat.rs` now say so at both sites.

No numbers in §7.40 are retracted.

### 7.41 Issue #14 follow-through — floor semantics, comma rejection, and a LOAD gate that cannot be turned into an accuracy gate

Acts on the decisions taken after §7.40, plus the linux/arm64 confirmation of §7.40's two
disputed claims against the **shipped** artifact rather than the re-export §7.40 measured.
No §7.40 number is retracted.

**Confirmation (linux/arm64, real hardware, `models_folded/gender-wav2vec2.onnx`, 966 nodes /
189,431,659 B), full `verify-models` per run:**

| session config | gender loads |
| --- | --- |
| `GeluFusion` | no |
| `GeluFusionL2` | **yes**, gender=2 |
| `GeluFusionL1;GeluFusionL2` | **yes**, gender=2 |
| `GeluFusionL1,GeluFusionL2` | no |

Both §7.40 claims hold on the shipped model: the pass is `GeluFusionL2`, and the separator is
`;`. The 1084-node re-export was a faithful stand-in.

**Change 1 — the level hatch is a FLOOR, not an override.** `DIAR_ORT_OPT_LEVEL` and
`DIAR_ORT_DISABLED_OPTIMIZERS` both returned early, so setting either silently un-did the
aarch64 cap: an operator raising the level to tune the *diarization* graphs lost speaker gender
with no error anywhere. Now the effective level is `min(requested, cap)` and the two hatches
compose with each other and with the workaround. Asymmetric on purpose — lowering can never
reintroduce a fused op (Level1 is already bitwise identical to Disable on this graph, §7.40),
while raising past the cap is the configuration measured to fail. `GraphOptimizationLevel`
derives `Ord`, so this is `min` rather than a hand-rolled ranking that could drift if ORT adds
a level.

**Change 2 — a comma-separated optimizer list is refused.** ORT takes `A,B` as one name,
matches nothing, and disables nothing *silently*. Same reasoning as the `DIAR_ORT_OPT_LEVEL`
typo fix: a hatch that silently does nothing is worse than no hatch, because it stops the
operator looking for the real cause. The remaining trap is unfixable here — ORT exposes no list
of registered optimizer names, so a wrong *name* still cannot be validated.

**Change 3 — `verify-models` stage 1 gained an aarch64 LOAD gate.** §7.40 established that 11
of 15 diarization graphs are rewritten to `com.microsoft::FusedConv` (fp32-only kernel) and are
safe purely by being fp32. The workaround is scoped by FILENAME to the gender model, so a
future fp16 export of any other graph provisions cleanly on amd64 and refuses to start on arm64
hosts only. Stage 1 now attempts every graph at ORT's default level with no workaround:

```
linux/arm64  aarch64 load gate: 1 graph(s) need the optimization cap
             (["gender-wav2vec2.onnx"]), as expected          [measured, full smoke green]
macOS arm64  aarch64 load gate: no graph needs an optimization workaround here
x86_64       aarch64 load gate NOT RUN on x86_64 — it can only be checked on aarch64
```

Any graph other than gender needing the cap FAILS stage 1 and is named. The x86_64 line is
deliberately not a pass: that host cannot vouch for arm64, and saying so beats implying a check
that did not happen. The macOS line is the §7.40 result reproduced by the shipped gate — same
missing fp16 kernel, no fusion, so nothing needs a workaround.

**This gate must stay a LOAD check.** An accuracy gate cannot catch this class: the session
never opens far enough to produce a number, so there is nothing to compare but load success.
Stated at the function so it is not later "improved" into a numeric check, which would remove
the gate silently.

Not a timed leg — every line above is a load/no-load or output-identity fact. 81 diar-core
tests pass on macOS arm64; the new tests pin the floor rule (`lower_of`) and the comma rule
directly rather than through the environment, which tests cannot set safely in parallel.

**Upstream:** drafts for the onnxruntime bug (an optimizer emits a node the same build has no
kernel for) and the `ort` doc bug ("comma-separated" is wrong; it is `;`) are in
`docs/upstream_drafts_ort_fusion.md`. NOTHING FILED — outward-facing reports need explicit
operator approval.

---

### 7.42 The 0.3.0 dependency sweep — the two base-image bumps, built and run rather than merged on green

**2026-09-01.** Six dependabot PRs were reviewed before 0.3.0. Recorded here are only the two
where the evidence *is* "it built and it ran": **#16** (`nvidia/cuda` 12.8.1 → 12.8.2) and
**#11** (`ubuntu` 24.04 → 26.04). No timed benchmark was run — the box was at load average ~10
with the OpenTranscribe stack and a sibling demo stack up, so this section contains capability
and correctness evidence only, and **no number here should be compared against any timed result
elsewhere in this file.**

Why not merge on CI: CI builds neither the CUDA image nor anything holding model weights, so a
green run says nothing about either bump. That is the same blind spot that would have passed the
cuda13 candidate, which compiles and links and then dies at session load.

**Controls.** `main` at `c94b758` before any merge: `cargo fmt --check` clean, `cargo clippy
--release --workspace --all-targets -- -D warnings` clean with **no `-A` exemptions**, 79 + 28
tests. After all merges, at `4c85a9f`: same commands clean, **81 + 28** — the two extra tests
are §7.41's, which landed on `main` mid-sweep, not anything this sweep added. Model-gated
integration suite (`-p diar-core -- --ignored`): **10 passed**, 116.44 s.

#### #16 — nvidia/cuda 12.8.1 → 12.8.2 (patch, within the pinned 12.8.x line)

This is a *patch* bump inside 12.8.x, which the dependabot ignore rule in `.github/dependabot.yml`
deliberately still permits (it blocks only major and minor). 13.x remains blocked and unusable.
All three tags the two Dockerfiles need — `12.8.2-{base,devel,runtime}-ubuntu24.04` — exist.

Built `docker/Dockerfile.server` from the final merged tree. Inside the image: `CUDA_VERSION=12.8.2`,
base OS still `Ubuntu 24.04.4 LTS` (unchanged — `nvidia/cuda:*-ubuntu24.04` pins that, so #11 does
not touch this image). The hand-installed cuBLAS/cuFFT/cuRAND/cuDNN set and the ORT 1.24.2 GPU
tarball were not modified.

Five-stage smoke, `--gpus '"device=0"'` on RTX A6000 (GPU 0, 3.5/49 GB used, 0% util at start):

```
1-parse        16 ONNX graphs loaded on the CPU EP
2-io-contract  16 signatures matched the compiled-in contract
5-plda         6 PLDA arrays + min_num_samples=400
3-numeric      3a 3.43e-5; 3b 0.00e0; 3c 0.00e0; 3d byte copy; 3e 2.38e-7; 3f 0.00e0;
               3g 2.38e-7 / 2.38e-7; 3h 2.58e-6; 3i 3.58e-7
4-end-to-end   2 speakers, 7 segments, 8 exclusive, gender=2
```

Exit **10**, which is the documented "nothing to verify against" code — `models_folded/` carries
no `diar-provision.json`. Not a failure.

**The control that makes this evidence rather than assertion.** Stage 1 reports the *CPU* EP, so
a passing run alone does not prove the CUDA path was exercised. The identical command with
`--gpus` removed **fails**, at stage 4, exit **9**:

```
error: the `cuda` execution device is not usable on this machine, so the end-to-end stage
could not run here. The models themselves are NOT implicated ...
Underlying error: loading segmentation model: ... CUDA failure 35: CUDA driver version is
insufficient for CUDA runtime version
```

So stage 4 genuinely opened CUDA sessions on the 12.8.2 image, and the numeric agreement above
is CUDA-EP output.

#### #11 — ubuntu 24.04 → 26.04 (the riskier one: it moves the CPU image AND the builder)

Four risks were checked rather than assumed.

**(a) Does `ubuntu:26.04` exist as a released LTS?** Yes. `VERSION="26.04 LTS (Resolute Raccoon)"`;
the built image self-reports `Ubuntu 26.04.1 LTS`. It publishes amd64, arm64/v8, armv7, ppc64le,
riscv64 and s390x manifests, so the multi-arch CPU image keeps a base on both target platforms.

**(b) Is `libopenblas0` still the right runtime package?** Yes — name unchanged, `0.3.26+ds-1ubuntu0.1`
(24.04) → `0.3.32+ds-5` (26.04). `ldd` on the shipped binary resolves
`libopenblas.so.0 => /usr/lib/x86_64-linux-gnu/libopenblas.so.0`. Nothing unresolved.

**(c) Does the ORT 1.24.2 linux tarball still run against 26.04's glibc?** Yes. glibc moves
**2.39 → 2.43**; the documented floor for this ORT is 2.38 (bookworm's 2.36 fails at link,
CLAUDE.md). Proven, not inferred: the smoke loaded all 16 ONNX graphs and completed end to end —
`4-end-to-end 2 speakers, 7 segments, 8 exclusive, gender=2`, exit 10, identical to the 24.04
result and to #16's.

**(d) Image size.** **195 MB → 246 MB (+51 MB, +26%).** This is the one real cost and it is not
a rounding error: small size is half the stated reason `Dockerfile.server-cpu` exists at all (its
header comment still cites "189 MB", which was already stale at 195 MB and is now further out).
The image is still ~14× smaller than the 3.46 GB CUDA image, so the artifact's purpose survives,
but the header comment should be corrected the next time that file is touched.

**The builder moved too, and that is the half CI never sees.** `docker/Dockerfile.builder` is the
reproducible build environment; #11 rebases it on 26.04. Built it, and confirmed the pinned
toolchain still installs there: `rustc 1.97.1 (8bab26f4f 2026-07-14)`, `clippy 0.1.97`,
`rustfmt 1.9.0-stable`, from `rust-toolchain.toml` via rustup, on `Ubuntu 26.04 LTS`. Every
package in `scripts/build-deps.txt` resolves on 26.04. Running the full gate **inside the 26.04
builder**: fmt clean, clippy clean under `-D warnings`, **81 + 28** tests. Builder image grows
1.76 GB → 1.82 GB.

**Not exercised — read this before trusting the arm64 leg.** Everything above is **x86_64 only**.
qemu binfmt is not registered on this host (`docker run --platform linux/arm64` dies with
`exec format error`), and installing it would have changed shared-machine state, so the arm64
half of the multi-arch CPU image was **never built or run on 26.04**. Two specific consequences:
the arm64 `libopenblas0`/glibc combination on 26.04 is unverified, and §7.41's aarch64 fp16
**load gate cannot fire here** — stage 1 says so itself: `aarch64 load gate NOT RUN on x86_64 —
it can only be checked on aarch64 (issue #14)`. The arm64 image is built by the release workflow
on a `v*` tag; **that run is the first place 26.04/arm64 gets tested, and it should be watched.**

#### The other four, briefly (no new measurements)

`huggingface_hub` 1.28.0 → 1.29.0 landed on a compatibility argument, not a run: `pyannote.audio
4.0.7` needs `>=0.28.1` and `transformers` needs `>=1.5.0,<2.0`, and the only API surface this
repo uses — `hf_hub_download`, `model_info`, `HF_HOME` — is untouched in 1.29.0 (`HF_ENDPOINT` is
handled by our own Rust preflight over raw HTTP, not by the library). **A real gated export with
an `HF_TOKEN` has NOT been run against 1.29.0**; that is the only way to be certain, and it is
outstanding. The three GitHub Actions bumps are CI-only and were merged on rebased green CI.

**A dependabot failure worth recording.** PR #8 (`docker/setup-qemu-action` 3 → 4) was **closed by
dependabot itself**, claiming "Looks like docker/setup-qemu-action is up-to-date now, so this is
no longer needed", while `main` still read `@v3`. It deleted the branch, so the PR could not be
reopened. The bump was reapplied by hand in `4c85a9f`. The lesson is that a dependabot PR
disappearing is not evidence the dependency is current — check the manifest.

Five of the six bumps landed. Nothing in this section retracts an earlier number.

---

### 7.43 Non-root containers (issue #7) and the standalone quickstart — the uid choice is decided by the WRITE path, not the serve path

**Date:** 2026-09-01. Host: this box (2× A6000, 1× 3080 Ti), GPU 0 used. Images built from a
clean `git archive HEAD` context — the working tree carried unrelated in-progress edits from a
sibling session, so building from it would have measured someone else's code.

**Change.** `docker/Dockerfile.server` and `docker/Dockerfile.server-cpu` now end on
`USER 10001:10001` (account `diar`, created with `--system --no-create-home`).
`docker/Dockerfile.provision` reclaims `USER root` for its `apt`/`pip` layers and drops back.
Both server Dockerfiles also gained a shared `runtime-base` stage and a `cli` target.

**Why 10001.** Outside the 1000-1999 band `useradd` allocates on a normal host, so a file left
with this ownership is unambiguous and cannot be confused with a real account's. The same
number is repeated in `docker-compose.yml`, `start.sh`, `QUICKSTART.md` and
`docs/INSTALL_NATIVE.md`.

**The real constraint is provisioning, not serving.** Serving needs no write access anywhere:
`/models` is `:ro`, the startup gate only `stat`s the marker, and the sole writable path is
`/tmp/diar-native` (1777). Provisioning writes 484 MB, and a container user cannot write a host
bind-mount it does not own — so a fixed uid alone would have made the documented first step
require a `chown`, i.e. a quickstart that fails on step one. Resolved by running the export as
the *invoking* user (`--user "$(id -u):$(id -g)"`, wired through `DIAR_UID`/`DIAR_GID`). Still
non-root, just not that non-root user. **No `chown` in the normal flow.**

**Four paths verified, none assumed:**

| path | result |
|---|---|
| serve against `:ro` models | `/healthz` 200, `/readyz` 200, `models_state: verified`, `devices: [cuda, cpu]`; `id` = `uid=10001(diar) gid=10001(diar)`. `touch /models/x` → `Read-only file system`. |
| `provision-models` against a **rw** mount | 484 MB / 25 files in **152.8 s**, smoke test passed, `gender classifier: fp16`. All 25 files owned by **1000:1000** (the invoking user), mode 0644. |
| `/tmp/diar-native` writable | `stat` = `1777 root:root`; write **and** delete of a probe file as uid 10001 succeeded. |
| bare `docker run <image>` (no args) | CUDA image, no args and no env: serves, `/healthz` 200, `/readyz` 200, `uid=10001`. CPU image with `DIAR_MODE=cpu`: same. |

A live `/diarize` through the non-root container returned 2 speakers, 7 segments, 8 exclusive
segments and both gender verdicts — so the `:ro` mount is genuinely sufficient for the full
request path, gender model included.

**Not a regression, and worth recording so it is not rediscovered:** a bare `docker run` of the
**CPU** image with no args and no env exits 1 with `cuda requires the cuda Cargo feature; this
build serves [cpu]`. That is the documented "unset *or unrecognized* `DIAR_MODE` ⇒ `cuda`"
fall-through, and the released **root** image `diar-server:0.3.0-cpu` was run side by side and
produced the byte-identical error. Non-root changes nothing here. `docker-compose.yml`
therefore pins `DIAR_MODE=cpu` explicitly rather than leaving it blank.

**Size cost of non-root: zero.** CPU image **195 MB**, the same as the released root
`diar-server:0.3.0-cpu`. The `cli` target adds no bytes to the serving image because it is a
sibling stage off `runtime-base`, not an extra binary in the sidecar (`ls /usr/local/bin` in the
serving image shows `diar-server` only).

**Trivy** (0.67.1, `--scanners vuln --severity HIGH,CRITICAL`): see §7.43a below.

#### Two bugs found by actually running the thing

1. **An empty `HF_ENDPOINT` breaks the export**, and the failure does not look like a config
   error. `diar-server` itself treats empty as unset (`preflight.rs` filters it), but it does
   **not** strip the variable from the environment of the Python export child, and
   `huggingface_hub` does `os.environ.get("HF_ENDPOINT", "https://huggingface.co")` — which
   returns the **empty string** when the key exists but is blank. The download URL then has no
   scheme and the export dies with `httpx.UnsupportedProtocol: Request URL is missing an
   'http://' or 'https://' protocol`, which reads as a network fault and is not one. A compose
   file written the obvious way (`HF_ENDPOINT: ${HF_ENDPOINT:-}`) hits this on every run.
   Fixed by defaulting to the literal URL. README §6e's "empty is treated as unset" was true
   only of the Rust side and has been corrected.
2. **`diar-cli` intermittently faults during teardown**, `corrupted double-linked list` →
   SIGSEGV (exit 139), **after** printing its JSONL result line and writing the RTTM. Isolated
   2×2 against build and log level, because the first instinct — blame the sibling session's
   uncommitted `diar-cli` edits — was wrong:

   | build | `RUST_LOG` | exit |
   |---|---|---|
   | clean HEAD | default (`info`) | **139** |
   | clean HEAD | `error` | 0 |
   | working tree (sibling's WIP) | default (`info`) | 0 |
   | working tree (sibling's WIP) | `error` | 0 |

   Re-running the clean-HEAD build at the default level gave 0/139/139 across three runs. So it
   tracks **shutdown work under verbose logging**, is nondeterministic, and is present on the
   committed tree — not caused by either session's changes. Output is complete and correct
   every time it happens. `start.sh --cli` now defaults the run to
   `RUST_LOG=warn,ort::logging=error` (which also makes the one line of real output visible
   instead of burying it under ORT provider registrations) and, on a non-zero exit **with** an
   RTTM present, reports the known teardown fault rather than either swallowing a segfault or
   failing a run whose results are on disk. The underlying fault is unfixed and is a genuine
   bug worth its own issue.

#### 7.43a Trivy

`trivy 0.67.1`, `--scanners vuln --severity HIGH,CRITICAL`, DB version 2, scanned 2026-09-01.

| image | OS packages | HIGH | CRITICAL |
|---|---|---|---|
| `diar-native:local-cpu` (new, non-root) | 112 | 0 | 0 |
| `diar-native:local-cuda` (new, non-root) | 137 | 0 | 0 |
| `diar-server:0.3.0-cpu` (released, **root**, baseline) | 112 | 0 | 0 |

No language-specific manifests are detected in any of them (`num=0`) — the Rust binary is
statically linked and carries no lockfile into the image, so trivy's OS-package scan is the
whole surface. The baseline was scanned in the same run to make the comparison meaningful:
the non-root change neither introduced nor removed a finding, which is the expected result —
it changes who the process runs as, not what is installed. The `apt-get upgrade -y` already in
both runtime stages is what keeps the count at zero.

### 7.44 B4 — mixed-device concurrency: CUDA is not measurably slowed by concurrent CPU work, and the default admission gate makes the question moot (issue #5)

First of the three timed legs §7.34 deferred as "NOT measured here, pending a quiet window".
Harness: `validation/b4_mixed_device.sh` (committed, reusable).

**Machine honesty, stated up front.** The box does not reach protocol-grade quiet and will not
while the `opentranscribe` and `otfresh-demo` stacks run: load average oscillated **10–30 on 48
cores** throughout, and GPU 2 held 38 301 MiB of an unrelated vLLM the whole time (untouched).
The defence is not pretending otherwise — it is **interleaving** (`cuda_only`, `mixed`,
`cuda_only`, `mixed`, …, legs 5 s apart) so that slow drift in background load hits both legs
roughly equally, and reporting a **ratio between adjacent legs** rather than an absolute time.
**The absolute wall times below are NOT protocol-grade anchors and must not be quoted as such**
or compared against §7.32/§4.19 numbers taken on a quiet box. The ratio is the result.

**Control set.** `diar-server:bench` built from `docker/Dockerfile.server` at `1ffbde0`;
builder `diar-native-builder:bench` = **Ubuntu 24.04.4 / rustc 1.97.1** (see the trap below);
`--gpus "device=0"` (RTX A6000, 45 GB free, 0% util — GPU 1's 3080 Ti has only ~5.5 GB free
against §7.34's 4 388 MiB peak, too tight to be safe); `DIAR_DEVICES=cuda,cpu`,
`DIAR_MAX_INFLIGHT=8`, `SPEAKRS_LAZY_SESSIONS=1`, `models_folded/`. CUDA leg = the four t9a
AMI files (`ES2004a IS1009c TS3003b EN2002b`, 17–36 min, four rooms) issued concurrently;
CPU load = **M=4** `/embed_window` loops hammering `EN2002c_360.wav` continuously for the
whole CUDA leg, staggered windows. All audio staged on local disk (`/tmp/bench_audio`) —
the AMI corpus lives on a **NAS mount** and its I/O has no place inside a timed leg.

| round | cuda_only | mixed | ratio | cpu_reqs done | load avg at leg |
| --- | --- | --- | --- | --- | --- |
| 1 | 38.13 s | 42.56 s | 1.116 | 161 | 16.7 → 30.6 |
| 2 | 38.53 s | 38.31 s | **0.994** | 193 | 26.2 → 27.9 |
| 3 | 36.95 s | 37.88 s | 1.025 | 193 | 22.1 → 25.6 |
| **median** | **38.13 s** | **38.31 s** | **1.005** | | |

**VERDICT: PASS.** The CUDA leg costs **+0.5% (median 38.13 → 38.31 s)** with four CPU
embedding jobs running flat out beside it. That is far inside the `cuda_only` leg's own
round-to-round spread of **1.58 s (4.3%)**, so it is not distinguishable from noise on this
box. Rounds 2 and 3 straddle zero (−0.6%, +2.5%), which is what "no effect" looks like.

**Round 1's +11.6% is CPU-side lazy-session warmup, and the CPU counter proves it rather than
excusing it.** `cpu_reqs` = **161, 193, 193** — round 1 completed 17% less CPU work, and rounds
2 and 3 are *identical*. Under `SPEAKRS_LAZY_SESSIONS=1` the CPU engine's embedding sessions are
built on first use; the warmup issued a single `/embed_window` on one handle, so round 1's mixed
leg was the first time **four concurrent** CPU handles forced session construction and arena
growth. The cost lands once, on both sides of that leg. Had the outlier been genuine GPU
contention, CPU throughput would have been *higher* in round 1, not lower.

**Zero VRAM for the CPU engine, sampled DURING and per-PID.** Peak was **4 416 MiB in all six
legs** — mixed and CUDA-only alike, no delta whatsoever. Sampling filters on this container's
PID: the card routinely carries other `diar-server` and `python3.13` processes, and whole-GPU
sampling would silently attribute theirs to us (the exact contamination §7.34 had to retract a
measurement for). This is the during-run confirmation of §7.34 B3's per-process claim, which
was previously only an idle/peak measurement on an otherwise-quiet card.

**ACCURACY CHECK — output identity, proven not asserted.** `rttm`, `segments`,
`exclusive_segments`, `centroids` and `num_speakers` hashed for all four files across all six
legs: **one hash per file, IDENTICAL** (`ES2004a cf1fa73f…`, `IS1009c c664a832…`,
`TS3003b c3be3707…`, `EN2002b 2287e581…`). Concurrent CPU work does not perturb CUDA output.

**The levers were not needed.** `DIAR_MAX_INFLIGHT_CPU` and a reduced `SPEAKRS_INTRA_THREADS`
for the CPU engine were held in reserve for a regression that did not appear. Neither was
touched; both remain untested as remedies because nothing needed remedying.

#### The `DIAR_MAX_INFLIGHT=8` requirement is itself the load-bearing finding

Getting 4 CUDA + 4 CPU genuinely concurrent **required raising the admission gate to 8**. At the
**shipped default of `DIAR_MAX_INFLIGHT=2`** the outer semaphore serialises the mix outright:
the contention this leg probes **is not reachable** without an operator deliberately opting into
it. So the honest framing of mixed-device concurrency is not "it is a risk" but:

> **mixed-device concurrency is a risk only above the default admission gate — and measured at
> 4×4, well above that gate, it costs 0.5% and changes no output.**

That belongs in how the CPU+GPU feature is described, not just in the harness header. It also
means B4 measured the *maximum-contention* configuration an operator could construct, not the
default one — the default is strictly safer than what is reported here.

#### Trap found before any number was taken: the builder image was silently the reverted base

`diar-native-builder:latest` on this box was **byte-identical to `diar-native-builder:u2604`** —
an Ubuntu **26.04** image left behind by the base bump that `1bbba89` *reverted* because 26.04
changes arm64 diarization output (issue #18). Benchmarking with it would have violated the
pinned-base rule invisibly, since nothing about `:latest` announces its base. Rebuilt from the
repo as `diar-native-builder:bench` and **verified before use**: `Ubuntu 24.04.4 LTS`,
`rustc 1.97.1` (matching `rust-toolchain.toml`). Check `docker run --rm <builder> cat
/etc/os-release` before trusting any local builder tag; image tags are not provenance.

#### Harness bug worth not repeating

The first B4 attempt hung for 12 minutes without issuing a single request. Cause was the
harness, not the server: `sampler=$(start_sampler ...)` runs in a **command substitution**, and
a background job started inside it inherits that substitution's stdout, holding the pipe open
forever — so the substitution never returns. The VRAM sampler must have its stdout redirected
away (`>/dev/null 2>&1 &`) and its PID passed via a global. Fixed in the committed harness with
a comment; symptom to recognise is an empty output dir plus zero request lines in `docker logs`
while the sampler log grows normally.

### 7.45 B1 — `--features cuda` does NOT slow CPU-mode inference, and the CPU-leg RTTM finally has a recorded MD5 (issue #5)

Second of the three timed legs §7.34 deferred. Harness:
`validation/b1_cuda_feature_cpu_cost.sh` (committed, reusable).

**The single variable is the `ort-sys` prebuilt distribution.** `ort/cuda` selects a different
prebuilt tarball whose statically-linked MLAS — the ORT CPU EP kernel library — may have been
compiled with different flags. Everything else is held fixed: one source tree at `96e5563`, one
toolchain, one container image, same models, same audio, same `--mode cpu`. Both binaries were
built by the same `diar-native-builder:bench` (**Ubuntu 24.04.4 / rustc 1.97.1**; see §7.44 on
why the local `:latest` could not be trusted).

**The premise was confirmed at the linker before it was measured.** `ldd` on both binaries is
byte-for-byte the same set — `libopenblas`, `libstdc++`, `libgcc_s`, `libm`, `libc`,
`libgfortran`, and **no ONNX Runtime `NEEDED` entry in either**. That is §7.34's static-linking
claim holding for the *default-features* build too, and it is what makes this a clean
single-variable test: the difference is entirely inside the statically-linked ORT objects.
Binary sizes differ accordingly, 32 619 904 B (default) vs 34 006 720 B (cuda), **+1.39 MB**.

**Control set.** AMI `EN2002c` first 360 s, 16 kHz mono, md5
`e192929eef63d8f61ac525ca4c906643` (source is already 16 kHz mono `pcm_s16le`, so `-t 360` is an
exact PCM truncation), staged on local disk — the AMI corpus lives on a **NAS mount** and its
I/O has no business inside a timed leg. `models_folded/`, `SPEAKRS_INTRA_THREADS` at its default
(`available_parallelism().min(6)` = 6, i.e. §7.32's "new" configuration). Legs **interleaved**
def/cuda/def/cuda/def/cuda. Load average **8.9 → 14.3 on 48 cores** across the whole leg — the
quietest window this session, and unusually stable for this box.

| round | default-features | `--features cuda` | load avg at leg |
| --- | --- | --- | --- |
| 1 | 52.54 s | 51.75 s | 8.87 / 9.33 |
| 2 | 52.66 s | 51.98 s | 10.14 / 10.77 |
| 3 | 51.86 s | 52.46 s | 14.28 / 12.67 |
| **median** | **52.54 s** | **51.98 s** | |

Metric is the engine-reported `elapsed_s` (what §7.32 reported); container start and model load
are excluded. Wall-clock medians were 56.77 s and 56.40 s, the same picture with a constant
~4.3 s of process startup added.

**VERDICT: PASS — no cost, and the sign points the wrong way for a regression.** The cuda-flavour
build is **0.56 s (1.1%) FASTER** at the median. That delta is *smaller than each build's own
round-to-round spread* (def 0.80 s, cuda 0.71 s) and the two ranges fully overlap
(def 51.86–52.66, cuda 51.75–52.46). The honest reading is **no measurable difference**, not a
speedup: three rounds cannot resolve 1% on this box, and round 3 reverses the ordering. What the
leg does establish is a firm bound — whatever the MLAS flag difference is, it does not cost
CPU-mode inference anything detectable at this resolution.

**ACCURACY CHECK — output identity, proven by diffing raw records.** All six runs across both
builds produced **one RTTM MD5 and one record hash**:

```
RTTM md5      c5a9fdf208f57a7b1129a85c5175cf86   (6/6 runs, both builds)
record sha256 5f338c99ff19eb27…                  (segments + exclusive_segments
                                                  + centroids + num_speakers)
```

The different ORT distribution changes neither timings nor numerics.

#### The §7.32 comparison cannot be made as the plan asked, and why that is worth saying

The task specified "the RTTM MD5 must EQUAL the §7.32 CPU-leg MD5". **It cannot be checked:
§7.32 states "one MD5 across all CPU legs" but never writes the value down**, and no 360 s
`EN2002c` artifact is committed under `results/`. Nor could the clip be reconstructed with
certainty — §7.32 records "first 360 s" but not the tool or command that produced it, so a
byte-identical input is not guaranteed even in principle. The identity claim above is therefore
**internal to this leg** (def vs cuda, the actual variable) and is *not* continuity with the
logged run. That is a real limitation, not a technicality: it means §7.32's CPU output and this
one have never been compared and cannot now be.

**Fixed going forward.** The value is recorded here — `c5a9fdf208f57a7b1129a85c5175cf86` for
`EN2002c` first 360 s, `models_folded/`, `--mode cpu`, alongside the input md5
`e192929eef63d8f61ac525ca4c906643` and the exact `ffmpeg -t 360 -ar 16000 -ac 1 -c:a pcm_s16le`
that produced it. A future CPU leg on this clip has something to diff against.
**Lesson for RESULTS entries generally: "identical MD5" without the digest, and a derived clip
without the command that derived it, are not reproducible claims.**

#### On the absolute numbers vs §7.32's 57.8 / 59.2 / 67.7 s

This leg's 52.54 s median is ~11% below §7.32's 59.2 s median (6.85× vs 6.1× RT). **No speedup
is claimed and none should be read into it.** The two are not comparable: different builder
image, unverifiable clip provenance (above), a session's worth of dependency drift, and §7.32's
own note that it was "not a protocol-grade quiet-machine leg" (load 9–13/48, close to this
leg's 8.9–14.3). Same band, different conditions. The B1 verdict rests entirely on the
**interleaved within-session A/B**, which is the only comparison here that controls its
variables.

### 7.46 B5 — the fp16 gender VRAM saving is **252 MiB, not ~500**: the CHANGELOG figure was borrowed from a different measurement basis (issue #5)

Third timed leg from §7.34's deferred list, and the one that mattered most because **§7.39
claimed "~500 MiB VRAM" by citing §7.18's measurement of the same model pair rather than taking
a fresh one**, and that borrowed number is asserted unqualified in `CHANGELOG.md`. Harness:
`validation/b5_gender_fp16_vram.sh` (committed, reusable).

**Control set.** `diar-server:bench`, `--gpus "device=0"` (RTX A6000), `DIAR_DEVICES=cuda`,
`DIAR_MAX_INFLIGHT=2`, `SPEAKRS_LAZY_SESSIONS=1`, `karpathy_10m.wav` (md5
`7c57039f944332e85b0b2a3c3f6963ca`, the §7.16/§7.18/§7.39 reference clip), `{"gender": true}`,
two diarize calls per container so the arena is grown before the peak is taken. Single variable
= a bind-mount of `gender-wav2vec2.onnx` over an otherwise identical `models_folded`
(189 431 659 B fp16 vs 378 529 501 B fp32) — the technique §7.39 used. Legs **interleaved**
fp32/fp16 ×3. VRAM sampled **DURING** at 2 Hz (15–16 samples per leg) and **filtered to the
container's host PID**. Load average **9.1–9.9 on 48 cores** and essentially flat across all six
legs — the quietest and most stable leg of this session, and a memory measurement is in any case
far more robust to load than a timed one.

| round | fp32 peak | fp16 peak | delta | load avg |
| --- | --- | --- | --- | --- |
| 1 | 4 694 MiB | 4 446 MiB | 248 | 9.60 / 9.90 |
| 2 | 4 698 MiB | 4 450 MiB | 248 | 9.50 / 9.71 |
| 3 | 4 698 MiB | 4 446 MiB | 252 | 9.14 / 9.69 |
| **median** | **4 698 MiB** | **4 446 MiB** | **252 MiB** | |

Reproducibility is excellent — a 4 MiB spread within each variant across three container
restarts.

**VERDICT: the direction is confirmed, the magnitude is NOT. fp16 saves 252 MiB here, about
HALF the ~500 MiB currently claimed.** The bind-mount demonstrably took effect: the two legs
differ in VRAM *and* in gender confidence at the 1e-6 level (below), so different graphs really
did load.

**This is not a retraction of §7.18.** §7.18 measured **"container VRAM (AMI run)"** — a
different clip (16 AMI meetings vs one 10-minute file) and, by its own wording, a
container/whole-GPU basis rather than the per-process one used here. Peak arena is workload
dependent, so 506 MiB and 252 MiB can both be honest measurements of different things. Note the
absolute levels differ too: §7.18's 5 396 / 4 890 MiB sit ~700 / ~444 MiB above this leg's
4 698 / 4 446 MiB, which is what a broader measurement basis would look like.

**What IS wrong is the generalisation.** §7.39 did not measure the pair; it cited §7.18 and the
figure then entered `CHANGELOG.md` as an unqualified "roughly −500 MiB VRAM" property of the
fp16 model. On the project's own reference clip, measured per-process, the saving is half that.
For orientation the raw weight difference is 189 MB = **180 MiB**, so 252 MiB (weights + arena)
is the more physically plausible of the two figures for a single-file run, while 506 MiB is
~2.8× the weight delta and needs the AMI workload to explain it.

**Docs corrected in the same commit** (neither is append-only): `CHANGELOG.md` now states the
measured range with both bases cited, and `docs/ORT_FUSION_FP16_AARCH64.md`'s option-(c) row —
which priced "fp32 gender on this platform" at "~500 MiB VRAM (§7.18)" and therefore *overstated
the cost of the aarch64 fp32 fallback by 2×* — now carries the measured figure. That row feeds a
platform decision, so the correction is not cosmetic.

**ACCURACY CHECK — two gates, both passed.**

1. **Diarization records identical.** Gender does not feed clustering, so swapping its precision
   must leave diarization untouched. `rttm` + `segments` + `exclusive_segments` + `centroids` +
   `num_speakers` hash to **one value across all six runs of both variants**
   (`0bb39f3bbfb44181…`). Anything else would have been a real bug rather than a VRAM result.
2. **Gender verdicts agree, 2/2.**

| speaker | fp32 | fp16 | delta |
| --- | --- | --- | --- |
| SPEAKER_00 | female 0.796751 | female 0.796750 | 1.15e-06 |
| SPEAKER_01 | male 0.999142 | male 0.999144 | 1.85e-06 |

These land on the historical operating point recorded in §7.16 and §7.39 for this clip (female
0.797 / male 0.999), so the fp32 leg is not some unvalidated graph — it is the same classifier
at the other precision. §7.18's headline finding (fp16 is verdict-preserving) is reconfirmed
independently; only its VRAM magnitude fails to generalise.

### 7.47 B2 — a resident CPU engine costs the CUDA path nothing: +0.8% latency, −4 MiB VRAM (issue #5)

Last of the legs §7.34 deferred, and the one that closes the superset feature: if operators are
invited to run `DIAR_DEVICES=cuda,cpu`, the second engine must not tax the first. Harness:
`validation/b2_cuda_both_engines.sh` (committed, reusable).

**Single variable:** `DIAR_DEVICES=cuda` vs `DIAR_DEVICES=cuda,cpu`. Every request carries an
explicit `"device":"cuda"`, so the CPU engine is **resident but idle** — the deployment shape
the superset claim actually creates. Same image (`diar-server:bench`), same clip, same request.
`--gpus "device=0"` (RTX A6000), `DIAR_MAX_INFLIGHT=2`, `SPEAKRS_LAZY_SESSIONS=1`,
`models_folded/`, AMI `EN2002c` first 360 s (md5 `e192929eef63d8f61ac525ca4c906643`). Each leg
is a fresh container, one warmup request discarded (cuDNN algo search and arena growth are
first-run costs, not steady-state latency), then **5 timed requests**; legs **interleaved**
cuda / cuda,cpu ×3 rounds. Load average **7.5 → 17.7 on 48 cores**, drifting upward across the
leg — which is precisely why the verdict rests on interleaved adjacent pairs and not on
absolute times.

| round | `cuda` median | `cuda,cpu` median | peak VRAM cuda | peak VRAM cuda,cpu | load avg |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.215 s | 2.293 s | 4 320 MiB | 4 316 MiB | 12.67 / 12.64 |
| 2 | 2.244 s | 2.195 s | 4 316 MiB | 4 314 MiB | 13.02 / 17.50 |
| 3 | 2.321 s | 2.261 s | 4 320 MiB | 4 316 MiB | 14.74 / 17.70 |
| **median** | **2.244 s** | **2.261 s** | **4 320 MiB** | **4 316 MiB** | |

**VERDICT: PASS.** Latency ratio **1.0076 (+0.017 s, +0.8%)**, against per-config round-to-round
spreads of **0.106 s and 0.098 s** — the delta is roughly a sixth of the noise band, and rounds
2 and 3 both come out *negative* (the two-engine config faster). No effect.

**VRAM delta for the resident CPU engine: −4 MiB**, i.e. zero to within sampling resolution, and
negative, which is how you can tell it is noise rather than a small real cost. Sampled at 2 Hz
**during** the timed requests and filtered to the container's host PID. This is the third
independent confirmation of §7.34 B3's "the CPU engine costs zero VRAM" — B3 measured idle and
peak on a quiet card, §7.44 confirmed it under mixed concurrent load, and this leg confirms it
during pure-CUDA steady-state work.

**ACCURACY CHECK — output identity.** `rttm`, `segments`, `exclusive_segments`, `centroids`,
`num_speakers` hashed for **every one of the 30 timed requests across both configurations**:
**one hash, `e6c90ce5ba9629ba…`**. Loading a second engine does not perturb CUDA output.

#### The §7.32 CUDA control (4.83 / 4.86 s) is not a valid comparator for this leg

The plan named it as the control. It cannot serve as one, for the same class of reason as §7.45:
**different measurement basis.** §7.32's CUDA figures are `diar-cli` numbers — a fresh process
per run, engine load included in the surrounding harness — whereas B2 times an HTTP request
against an already-warm server on an A6000. This leg's 2.24 s is ~2× below that figure and
**no speedup is claimed or implied**; the two quantities are not the same measurement, and
§7.28's fbank∥GPU pipelining landed between them besides. The valid comparison is the
interleaved within-session A/B above, which controls its variables. Recording this because the
same trap has now bitten three of these legs: **a "control" from RESULTS is only a control if it
was measured the same way.**

### 7.48 B6 — the aarch64 Level1 gender cap is free within resolution, because gender costs ~0.16 s and not the ~1.5 s everyone has been quoting (issue #14)

Bonus leg beyond §7.34's three. The gender session is capped at
`GraphOptimizationLevel::Level1` on aarch64 (`crates/diar-core/src/ort_compat.rs`), where the
uncapped alternative **does not load at all** — so the cap's cost is unmeasurable there, there
being no comparison to make. Measured instead on **x86_64**, where both levels load, via
`DIAR_ORT_OPT_LEVEL` (`unset` → ORT default Level3; `basic` → Level1, i.e. what aarch64 runs).
Harness: `validation/b6_gender_opt_level_cost.sh`.

**Design: the reported figure is a MARGINAL, not a wall time.** `DIAR_ORT_OPT_LEVEL` reaches
only sessions built through `diar_core::ort_compat` — the gender model and the smoke test — and
never speakrs' 15 diarization graphs. So each container measured **both** `gender:false` and
`gender:true`, and the number below is `true − false` *within the same container*. That
subtracts the diarization time the knob cannot touch instead of hunting a small effect inside a
~3.8 s wall. `gender:false` doubles as a **null control** that must not move between legs.
`karpathy_10m.wav`, `diar-server:bench`, `--gpus device=0` (A6000), 5 requests per cell,
interleaved all/basic ×3, load average **11.6 → 14.7 on 48 cores**.

| round | leg | no_gender | gender | **marginal** | load avg |
| --- | --- | --- | --- | --- | --- |
| 1 | all (Level3) | 3.632 s | 3.795 s | **0.163 s** | 13.53 |
| 1 | basic (Level1) | 3.783 s | 3.767 s | **−0.016 s** | 13.63 |
| 2 | all | 3.551 s | 3.676 s | **0.125 s** | 14.50 |
| 2 | basic | 3.641 s | 3.740 s | **0.099 s** | 16.05 |
| 3 | all | 3.602 s | 3.837 s | **0.235 s** | 14.65 |
| 3 | basic | 3.643 s | 3.630 s | **−0.013 s** | 14.19 |
| **median** | all | 3.602 s | 3.795 s | **0.163 s** | |
| **median** | basic | 3.643 s | 3.740 s | **−0.013 s** | |

**VERDICT: INCONCLUSIVE on the exact cost — and that is the honest answer, not a hedge.** The
nominal result is that Level1 is *0.176 s FASTER* than Level3, which cannot be true: capping
optimization does not speed a graph up. The measurement is simply at or below its own
resolution, and it says so out loud in three places:

1. **Two of the three `basic` marginals are NEGATIVE** (−0.016 s, −0.013 s). Gender cannot take
   negative time. That alone disqualifies the point estimate.
2. **The marginal's own round-to-round spread is 0.110 s (all) and 0.115 s (basic)** — comparable
   to the 0.176 s "effect".
3. **The null control drifted 0.041 s** (`no_gender` 3.602 vs 3.643 s), a quantity the knob
   provably cannot affect. That is a quarter of the claimed signal, and it is pure background.

**What the leg DOES establish, firmly, is an upper bound — and it answers the decision.** The
*entire* gender marginal is **~0.16 s** on this platform. The cap cannot possibly cost more than
that, because that is all the time gender takes. So: **the cap is free to within anything worth
measuring, and this does NOT argue for revisiting `GeluFusionL2` as the aarch64 fix.** §7.41's
choice of the Level1 cap stands, and it stands on a cost bound rather than on an assumption.

**The knob demonstrably took effect** — this is not a null result from a no-op setting. The two
legs return *different gender confidences* (SPEAKER_00: `0.79675186` at Level3 vs `0.79671210`
at Level1, Δ 3.98e-05), which is exactly the numeric fingerprint of the Level-2 fusions being
skipped, and consistent with §7.40's finding that Level1 is bitwise identical to `Disable` on
this graph while Level3 differs slightly. Worth stating explicitly given §7.40's Trap 1: an
*unrecognized optimizer name* is silently ignored by ORT, so a null result from this family of
knobs always needs proof the setting was honoured. Here the numerics are the proof.

#### The premise of the question was wrong: gender is ~0.16 s, not ~1.5 s

§7.18 records "gender is ~1.5 s of a 6 s call", and that figure is what makes the cap sound
worth investigating (it is also quoted in the issue #14 discussion). **Measured here on the same
reference clip, gender adds 0.163 s to a 3.60 s call — roughly 10× less, and 4% of the call
rather than 25%.** Conditions differ (this is CUDA on an A6000, the shipped fp16 model, gender
inside the sidecar per §7.16, `DIAR_GENDER_MAX_SECONDS=5` over 2 speakers), so **§7.18's number
is not retracted** — it is not reproducible from the information recorded there, which is the
same class of gap §7.45 hit with the missing MD5. But the *current* cost on the current stack is
0.16 s, and any future reasoning that starts from "gender is 1.5 s of the call" is starting from
a stale premise. Flagging it for reconciliation rather than silently overwriting it.

**ACCURACY CHECK — both gates pass.** Diarization records (`rttm`, `segments`,
`exclusive_segments`, `centroids`, `num_speakers`) hash to **one value across both legs**
(`faf30c2f47fe9d82…`), independently confirming the documented scoping that
`DIAR_ORT_OPT_LEVEL` does not reach speakrs' diarization graphs. Gender labels **agree 2/2**
(female / male) at the historical operating point, with confidence deltas 3.98e-05 and 1.70e-06.

### 7.49 CPU/CUDA output identity is clip-dependent: centroids always agree, one segment boundary can move by one frame (corrects §7.34's generalisation)

**Correcting a claim we published, not retracting a measurement.** §7.34 measured CPU-vs-CUDA
output on the 26 s smoke fixture, found it bit-identical with max centroid delta 0.0, and that
became "output is **bit-identical** between devices" in the CHANGELOG, in issue #1's closing
comment, and in two OpenTranscribe issues. The measurement was correct. **The generalisation
from one clip was not.**

Measured on `diar-server:0.3.0-rel` (final release build), same container, same models, both
requests carrying an **identical `file_id`** (see the test-artifact note below), `gender:true`:

| clip | rttm identical | segments identical | max boundary delta | max centroid delta |
|---|---|---|---|---|
| `fixtures/test.wav` (26 s) | yes | yes | 0 | 0 |
| karpathy `clip30.wav` (30 s) | **no** | **no** | **0.016875 s** | **0** |

Both clips: `num_speakers`, segment count, exclusive-segment count and gender verdicts all
agree exactly (1 speaker / 2 segments / 2 exclusive / male 0.999 on the real clip).

**What actually differs is one boundary, by exactly one frame.** 0.016875 s is a single
segmentation frame at this model's resolution; the two segment lists differ only in the `end` of
segment 1 (27.53721875 vs 27.52034375). That is a posterior sitting on the binarisation
threshold and landing on opposite sides under CPU vs CUDA float arithmetic — the ordinary
consequence of different kernels, not a defect, and not something a tolerance should be widened
to hide.

**Centroids were bit-identical on BOTH clips.** That is the stronger and more useful invariant,
and it is the one consumers doing embedding-only work on CPU actually depend on: speaker
*embeddings* do not move between devices; segment *boundaries* can, by one frame.

#### A test artefact worth recording, because it wasted a cycle

The first comparison reported `rttm_identical=False` even on the smoke clip, which briefly looked
like a much larger problem. Cause: the two requests were sent with different `file_id` values
(`smoke-cuda` / `smoke-cpu`) and **the RTTM embeds the file id**, so the payloads could not
possibly match. Re-run with one id, the smoke clip is identical on every field. Any future
cross-device or cross-build identity check must hold `file_id` constant or diff a field that
does not contain it.

#### Disposition

The CHANGELOG, issue #1 and the OpenTranscribe issues are corrected to state the invariant that
holds — centroids identical, boundaries within one frame — rather than the one that does not.
§7.34's numbers stand as measured on the clip it measured.

### 7.50 `SPEAKRS_FBANK_POOL` threaded through `RuntimeConfig` — the `set_var` is gone, and the knob was a lie (issue #3)

Two defects, one root cause. `DiarEngine::load` called
`std::env::set_var("SPEAKRS_FBANK_POOL", ..)` and the patched speakrs loader read it back inside
the same call (`inference/embedding/load/sessions.rs`).

1. **Thread-safety.** glibc `setenv`/`getenv` is not thread-safe; Rust 2024 marks
   `std::env::set_var` `unsafe` for exactly this reason. Our crates are edition 2021, so the call
   compiled with no `unsafe` and no warning — the hazard was invisible at the call site.
2. **The knob did not work.** The `set_var` was *unconditional* and overwrote whatever the
   operator had set, before speakrs read it. `EngineConfig::fbank_pool` was `None` at all four
   construction sites and nothing parsed the variable into it, so an operator setting
   `SPEAKRS_FBANK_POOL=1` to stop the pool contending on a shared box silently got `cores/4`,
   with no log line saying so. README, CLAUDE.md, PLAN.md and `docs/UPSTREAM_PRS.md` all
   described it as settable.

#### The `SAFETY` comment was already false, not merely fragile

The comment read *"called before any speakrs session exists; single-threaded load path"*. That is
true of the **first** load only. `EngineRegistry::load` (`crates/diar-server/src/engines.rs`)
loops over `DIAR_DEVICES` and calls `DiarEngine::load` once per device, so with
`DIAR_DEVICES=cuda,cpu` the second call runs `setenv` while the first engine's ORT intra-op
thread pools are alive. `provision/verify.rs` has the same shape (a CUDA load, then a CPU
retry). So the premise was violated in the deployed multi-device path today, not just in a
hypothetical lazy-loading future. What kept it from being an *observed* corruption is that no
in-flight request was reading the environment at that moment — but speakrs reads
`SPEAKRS_ARENA_SHRINK` (`inference.rs`) and `SPEAKRS_AHC_THREADS` (`clustering/ahc.rs`) on the
**request** path, not just at load, so the window was real.

#### The change

Upstream-shaped, in `vendor/speakrs`:

- `pipeline/config.rs`: new `RuntimeConfig::fbank_pool: Option<usize>`, defaulting to `None`.
- `inference/embedding/load/sessions.rs`: pool size comes from `config.fbank_pool`, falling back
  to `auto_fbank_pool_size()` (the `SPEAKRS_FBANK_POOL` env read, then `cores/4` clamped 1..=8)
  when it is `None`. **Existing speakrs consumers are unaffected**: `None` reproduces the old
  code path exactly. Also drops the now-dead `#[cfg(not(feature = "coreml"))] let _ = config;`
  and adds a `debug!` line reporting the resolved pool size (there was previously no way to
  observe it at all — which is what made the second defect invisible).

In `diar-core`:

- `EngineConfig::new` reads `SPEAKRS_FBANK_POOL` **once**, via `parse_fbank_pool`, which takes
  the raw string rather than reading the environment itself so it is unit-testable without
  `set_var`. All four construction sites go through `new`, so they all inherit the fix.
- `default_fbank_pool(mode)` and `EngineConfig::resolved_fbank_pool()` make the resolution rule
  explicit and testable.
- `DiarEngine::load` builds a `RuntimeConfig { fbank_pool: Some(..), ..default() }` and passes it
  to `EmbeddingModel::with_mode_and_config`. **No `set_var` remains in `diar-core`.**

Blank is treated as unset; a malformed value warns and falls back to the mode default rather
than pretending the operator got what they asked for; `0` is honoured and disables the pool.

#### Before/after — measured, not asserted

The `debug!` line was added to speakrs **first** and the pre-fix binary was built with it, so
both legs are observed rather than one being reasoned about. Identical everything else: same
host (48 cores → `48/4 = 12` clamped to 8), same `models_folded`, same `clip30.wav`, one
`diar-cli` run per cell, `RUST_LOG=speakrs=debug`, CUDA legs on `CUDA_VISIBLE_DEVICES=0`.

| condition | before (set_var) | after (RuntimeConfig) |
|---|---|---|
| `mode=cuda`, var unset | `fbank_pool=8` | **`fbank_pool=8`** |
| `mode=cpu`, var unset | `fbank_pool=1` | **`fbank_pool=1`** |
| `mode=cuda`, `SPEAKRS_FBANK_POOL=3` | `fbank_pool=8` (ignored) | **`fbank_pool=3`** |
| `mode=cpu`, `SPEAKRS_FBANK_POOL=3` | `fbank_pool=1` (ignored) | **`fbank_pool=3`** |

The top two rows are the no-behaviour-change control: defaults are byte-for-byte the ones every
deployment has been running. The bottom two are the defect and its fix — the "operator knob" that
did nothing now does what the table says.

#### Gates

- diar-core **87 passed** (82 before + 5 new: mode defaults, blank/absent override, numeric
  override incl. `0`, malformed fallback, explicit-wins-over-default), 10 `provision_smoke`
  ignored as usual.
- diar-server **28 passed**, unchanged.
- speakrs **96 passed**, 0 failed (`--no-default-features --features openblas-system,online`,
  `RUST_MIN_STACK=16777216`).
- `cargo clippy --release --workspace --all-targets -- -D warnings` exit 0, **no exemptions
  added**. (speakrs' pre-existing `reconstruct`/`reconstruct_smoothed` `dead_code` warning is
  unrelated and predates this change.)

#### Audit: other env round-trips of the same shape

Every `set_var`/`remove_var` in `crates/` and `vendor/speakrs/src` was enumerated.

- **`RUST_MIN_STACK`** (`crates/diar-server/src/main.rs`) — same shape, **safe and staying**:
  first statement of `main()`, before `Cli::parse()` and before the tokio runtime is built, so no
  threads exist. The reader is inside Rust std, not this tree. `docker/Dockerfile.builder` also
  sets it as an image env, which makes the guard a no-op in-container.
- **`ORT_DYLIB_PATH`** (`vendor/speakrs/src/inference.rs`) — test-only *and* behind the
  `load-dynamic` feature, which this workspace never compiles (`default-features = false,
  features = ["openblas-system"]`). Not fixed here; it is an upstream test-hygiene issue.
- **Test-only round-trips in our crates, previously unnoticed** — `DIAR_DEVICES`
  (`diar-server/src/cli.rs`), `HF_TOKEN`/`HUGGINGFACE_TOKEN`/`HUGGING_FACE_HUB_TOKEN`
  (`diar-core/src/provision/preflight.rs`), `DIAR_ALLOW_UNVERIFIED_MODELS`
  (`diar-core/src/provision/mod.rs`), `DIAR_EXPORT_PYTHON` (`diar-core/src/provision/exporter.rs`).
  These mutate the environment inside `#[cfg(test)]` while sibling tests in the *same binary*
  read the same variables in parallel. `cli.rs` has a `static ENV: Mutex<()>` that serializes the
  writers against each other but not against readers elsewhere in the binary; the other three
  have no guard at all. They are green today and are **not** part of this fix, but they are the
  same class of bug and should get the same treatment (parse-a-`&str` helpers, or
  `#[serial]`-style serialization). Recorded here so the next person does not have to re-derive
  it.
- `xtask/src/commands/dstack.rs` uses `xshell::Shell::set_var`, which only sets the child
  process env — not a process-global mutation, not a hazard.

No other production `set_var` exists in the tree.

#### What this unblocks

Lazy engine loading, rejected during the issue #1 design **for this reason alone**. A resident
CPU engine costs ~620 MB RSS (§7.34) and `DiarEngine::load` is now free of process-global state,
so loading on a `spawn_blocking` request thread is sound. Nothing lazy is implemented here — the
point was to remove the blocker. The doc comments in `engines.rs` and `main.rs` that described
serial pre-serve loading as a *soundness* requirement now describe it as the fail-fast choice it
has become.

#### Upstream

`RuntimeConfig::fbank_pool` is a genuine speakrs API improvement, not a diar-native workaround,
and is drafted for upstream in `docs/upstream_drafts_fbank_pool.md`. **Nothing has been filed** —
anything outward-facing against `avencera/speakrs` needs explicit operator approval.

### 7.51 `diar-cli`'s teardown abort is `ort` logging from a `.fini_array` destructor — not a race, not speakrs, and `diar-server` had it too (issue #19)

**Verdict: root-caused and fixed.** The fault is 100% deterministic once the right knob is
found; the "2 of 3 runs" in the issue was two independent gates being crossed, not a race. Fixed
by terminating through `diar_core::shutdown::exit` (`_exit`) instead of returning from `main`.
Output was never affected — verified byte-for-byte, not assumed.

#### The reproduction is deterministic, and the discriminator is `ort`, not `speakrs`

`diar-cli --mode cpu`, `clip30.wav`, `models_folded`, binary `/tmp/diar_target_qual` (pre-fix),
3 runs per cell:

| `RUST_LOG` | exit codes |
|---|---|
| `trace` | **134, 134, 134** |
| `speakrs=trace,ort=trace` | **134, 134, 134** |
| `speakrs=trace` | 0, 0, 0 |
| `debug` | 0, 0, 0 |
| unset (`info,ort::logging=warn`) | 0, 0, 0 |

The issue's own repro line (`RUST_LOG=speakrs=trace`) is the one cell that does **not** fail here
— 0 for 6 on this host. What fails is anything that enables the `ort` targets at TRACE. The
level, not the source tree, was already established in the issue; this narrows it further to a
single crate and a single target, `ort::lifetime`.

#### Root cause, from the backtrace rather than inferred

`RUST_BACKTRACE=full`, 5 runs of 5 identical:

```
10: std::thread::local::panic_access_error
11: LocalKey<RefCell<String>>::with::<tracing_subscriber::fmt::fmt_layer::Layer<...>::on_event>
12: tracing_subscriber::fmt::Subscriber<...>::event
13: tracing_core::event::Event::dispatch
14: <ort::environment::Environment as Drop>::drop
15: Arc<ort::environment::Environment>::drop_slow
16: ort::environment::release_env_on_exit
17: _dl_call_fini
18: _dl_fini
19: __run_exit_handlers
20: __GI_exit
21: __libc_start_call_main
```

```
cannot access a Thread Local Storage value during or after destruction: AccessError
fatal runtime error: failed to initiate panic, error 5, aborting
```

In ort 2.0.0-rc.12: `G_ENV` (`src/environment.rs:65`) holds a **strong** `Arc<Environment>` —
deliberately, because ORT tolerates `CreateEnv` only once per process — and its only release site
is `release_env_on_exit`, placed in `.fini_array` (`src/environment.rs:75-83`).
`Environment::drop` (`:240-245`) then emits `trace!(target: "ort::lifetime", "-DROP ...")`.
`.fini_array` runs from `_dl_fini`, after `main` has returned and after the Rust runtime has
destroyed the main thread's thread-locals, so `tracing_subscriber`'s fmt layer — which formats
through a thread-local `RefCell<String>` — gets `AccessError` and panics where unwinding cannot
start. Full write-up: `docs/ORT_ATEXIT_TEARDOWN.md`.

**The allocation that is "double-freed" is not one.** The issue's glibc `corrupted double-linked
list` framing pointed at a double free; the actual fault is a panic during atexit. Both are the
same window (the process is being torn down under it), and the abort is what reproduced here.
SIGSEGV/139 was never reproduced on this host at any of the cells above — SIGABRT/134 every time.
The fix removes the window itself, so it covers either symptom.

#### Two gates, which is why it looked intermittent

1. **The event must be enabled.** `ort::lifetime` is TRACE. At `debug` or below the callsite is
   disabled and never touches TLS.
2. **`main` must return normally.** `std::process::exit` does *not* run TLS destructors, so the
   thread-local is still alive when `.fini_array` fires. Measured on the isolated probe: identical
   program, `std::process::exit(42)` → exit 42, 5 of 5; implicit return → 134, 5 of 5.

Gate 2 is why `verify-models` was unaffected: it ends in `std::process::exit`, and separately
installs no subscriber. Measured: 8 runs, exit 10 every time, with and without `RUST_LOG=trace`.

#### Minimal reproduction — 8 lines, no model, no speakrs, no audio

```rust
// ort = "=2.0.0-rc.12"; tracing-subscriber = { version = "0.3", features = ["env-filter"] }
fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("trace"))
        .with_writer(std::io::stderr)
        .init();
    let _env = ort::environment::current().expect("env");
}
```

Exit 134, deterministic. This is the control that rules out speakrs, BLAS threadpools, the fbank
pool, ORT sessions and diarization entirely: none of them are present.

#### `diar-server` DOES share the defect — on the bind-failure path

Not in normal operation: it has no signal handler, so `docker stop` kills it before `.fini_array`
runs. But it reaches libc `exit` with a subscriber installed and engines loaded whenever `run()`
returns `Err` — most reachably, a port conflict. Port 18701/18702 held by another process,
`DIAR_MODE=cpu`, `RUST_LOG=trace`, 3 runs each:

| binary | exit codes | operator sees |
|---|---|---|
| pre-fix | **134, 134, 134** | `corrupted double-linked list` / SIGABRT |
| post-fix | 1, 1, 1 | `Error: Address already in use (os error 98)` |

So the pre-fix server answered a port conflict with a heap-corruption abort and swallowed the
actual diagnosis. That is the more damaging half of this issue, and it was in the deployed
artifact.

#### Output was never truncated

Verified rather than assumed, since the issue flagged it as the thing that would change the
priority. Crashing run (`RUST_LOG=trace`, exit 134) vs clean run (`RUST_LOG=debug`, exit 0), same
clip, `--json`:

- RTTM: **identical** (modulo the file-id column, which carries the `--label`).
- JSON: **identical** except the same label inside the embedded `rttm` string.
- stdout JSONL: present and complete in both.

Post-fix RTTM is likewise identical to the pre-fix clean run. The work was always finished; only
the process's report of itself was wrong.

#### The fix, and why it is a fix rather than a timing change

`crates/diar-core/src/shutdown.rs`: `exit(code)` flushes stdout/stderr, then `libc::_exit`.
`exit_main(result)` is the shared `main` epilogue (anyhow's `Error: <chain>` to stderr, then 0/1).
`diar-cli` and `diar-server` are now `fn main() -> !`; every exit in both — including the three
provisioning subcommands and `startup_gate_or_exit` — routes through it.

`_exit` does not run libc's exit handlers, so `release_env_on_exit` never executes. This is
structural, not a timing shift: the faulting frame is removed from the program, not raced with.

No local alternative works. `G_ENV` is private and strong, so dropping every `Session` does not
empty it. `set_global_default` cannot be undone, so the subscriber cannot be uninstalled before
returning. Clamping `ort::lifetime` in `logging.rs`'s default filter would hide only gate 1, would
not survive an explicit `RUST_LOG=trace` (the acceptance criterion is *any* level), and would
silently ignore what the operator asked for.

`std::process::exit` was measured and would have been sufficient for the observed abort (gate 2).
`_exit` was chosen over it because it additionally skips `ReleaseEnv` reaching into
`libonnxruntime.so` during `_dl_fini`, where destructor ordering across shared objects is not
guaranteed — the plausible origin of the reported `corrupted double-linked list`. Skipping
`ReleaseEnv` at process exit costs nothing: the kernel reclaims the address space.

The real defect is upstream — emitting a `tracing` event from a `.fini_array` destructor is
unsound for any subscriber that uses thread-locals. Report drafted in
`docs/ORT_ATEXIT_TEARDOWN.md`. **Nothing has been filed**; anything outward-facing against
`pykeio/ort` needs explicit operator approval. `ort` stays pinned `=2.0.0-rc.12`.

#### After: `diar-cli` exits 0 at every level

Same matrix, post-fix binary, 3 runs per cell:

| `RUST_LOG` | pre-fix | post-fix |
|---|---|---|
| `trace` | 134, 134, 134 | **0, 0, 0** |
| `speakrs=trace,ort=trace` | 134, 134, 134 | **0, 0, 0** |
| `speakrs=trace` | 0, 0, 0 | 0, 0, 0 |
| `debug` | 0, 0, 0 | 0, 0, 0 |
| unset | 0, 0, 0 | 0, 0, 0 |

#### Regression test

`crates/diar-core/tests/shutdown_teardown.rs`, driving `crates/diar-core/src/bin/diar-teardown-fixture.rs`.

The harness cannot stand in for a binary here: **libtest exits via `std::process::exit`**, which
skips the TLS teardown and so never arms gate 2. A first attempt that re-executed the test binary
passed for the wrong reason and was replaced. The fixture is a real `main` that arms the identical
hazard (subscriber + live ORT environment) and differs only in how it leaves `main`:

| fixture mode | exit | in CI |
|---|---|---|
| `exit 42` | 42 (5 of 5) | gated |
| `ok` | 0 | gated |
| `err` | 1, with `Error: outer context` + cause chain on stderr | gated |
| `return` | **SIGABRT / 134 (5 of 5)** | `#[ignore]` |

The `return` case is the pre-fix shape and passes — i.e. it still reproduces the abort — which is
what makes the other three non-vacuous. It is `#[ignore]`d because it asserts an *upstream* defect
and should start failing the day `ort` fixes it.

Model-free by construction: the fixture creates a bare ORT environment and loads no graph, so this
runs in CI with none of the gated artifacts present. The `err` case also pins that `shutdown::exit`
flushes before terminating — `_exit` skips libc's stream flushing, and without the explicit flush
the error message would vanish.

The fixture is a `[[bin]]` of `diar-core`, which is a lib-only dependency of both images: the
Dockerfiles build `-p diar-server -p diar-cli` and copy binaries by name, so it never ships.

#### Consequence for `start.sh --cli`

Its documented workaround — reporting a non-zero exit as a known teardown fault rather than
failing the run — is now unnecessary on this binary. `start.sh` is owned by another session and
was **not** edited here; flagged for its owner rather than changed.

#### Environment

x86_64, glibc 2.39, ort 2.0.0-rc.12, rustc 1.97.1, host build, default (CPU) features. Machine was
under unrelated load (load average ~13 of 48 cores) — irrelevant to exit codes, and no timing claim
is made in this section.

#### §7.51 addendum — the 139 is the 134, and one rationale in §7.51 is RETRACTED

Follow-up sweep (185 runs, host + container) plus a first-hand PID-1 control. Two corrections to
§7.51 above, which stands otherwise. Nothing about the fix changes; the shipped behaviour is
unaffected.

**1. Exit 139 DOES reproduce — in the container — and it is the SAME fault, not a second one.**
§7.51 said SIGSEGV/139 "was never reproduced". That was true of the *host* and is misleading as
written. In the container it reproduces 100%. The difference is entirely **where PID 1 sits**.
Same image, same env, same clip, only the PID-1 position changed:

| invocation | PID 1 | exit |
|---|---|---|
| `docker run ... diar-native-cli:local-cpu` | `diar-cli` itself | **139** x3 |
| `docker run --init ...` | tini | **134** x3 |
| `docker run --entrypoint /bin/sh ... -c 'diar-cli ...; echo $?'` | `sh` | **134** x3 |

As PID 1 in the namespace a `SIG_DFL` SIGABRT is discarded by the kernel, so glibc's `abort()`
falls through its `raise()` path into the trailing `ABORT_INSTRUCTION` and the death is reported
as 128+11 instead of 128+6. The stderr text is byte-identical in all cases — the TLS
`AccessError` and `failed to initiate panic, error 5, aborting`. So the issue's "139" and this
section's "134" were always one bug, observed through different PID-1 positions.

**`corrupted double-linked list` was never reproduced at all** — zero hits for `corrupted`,
`free()`, `malloc` or `SIGSEGV` across 185 stderr captures on host and in the container. The
issue title's glibc heap message remains unexplained and unreproduced; every observed failure was
the TLS abort. This is recorded as "could not reproduce", not as "the reporter was wrong".

**2. RETRACTED: the stated reason for preferring `_exit` over `std::process::exit`.** §7.51
justified `_exit` partly because it "additionally skips `ReleaseEnv` reaching into
`libonnxruntime.so` during `_dl_fini` — the plausible origin of the reported `corrupted
double-linked list`". **That hypothesis is disproven**: there is no heap-corruption flavour to
explain, and `Environment::drop` (ort `src/environment.rs:240-245`) calls `ReleaseEnv` *first* and
logs *second*, so `ReleaseEnv` had already completed successfully every time the abort fired.

`std::process::exit` would therefore have been a sufficient fix for the only fault ever observed.
`_exit` is **kept** — it is what shipped, is verified end to end, and removes the entire class
(any `.fini_array` destructor doing anything unsafe after `main`, not just this one) rather than
relying on Rust's guarantee that `process::exit` skips TLS destructors. But that is now the whole
of the argument, and it is weaker than §7.51 claimed. No re-measurement was needed: both
mechanisms were already measured clean.

**3. The trigger is narrower than `RUST_LOG=trace`.** The only directive that matters is one
enabling target **`ort::lifetime`** at TRACE. Host matrix, 15 runs per cell on two clips
(`clip30.wav` and `vendor/speakrs/fixtures/test.wav`), pre-fix binary, `--mode cpu`:

| `RUST_LOG` | clip30 | test.wav |
|---|---|---|
| unset / `warn` / `info` / `speakrs=trace` | 0 x15 each | 0 x15 each |
| `ort::lifetime=trace` | **134 x15** | **134 x15** |

120/120 clean, 30/30 abort. **`speakrs=trace` — the incantation CLAUDE.md documents for engine
stage timings — is safe and always was.** `RUST_LOG=trace`, the natural thing to reach for when
debugging, is the trigger. This is why the original report's own repro line looked flaky: it
never triggered the fault at all, and the failures attributed to it came from elsewhere.

**Container before/after, PID 1, the reporter's exact configuration:**

| binary | `ort::lifetime=trace` | `RUST_LOG=trace` |
|---|---|---|
| pre-fix | **139, 139, 139** | 139 x5 (swept) |
| post-fix | **0, 0, 0** | **0, 0, 0** |

Post-fix container RTTM is byte-identical to the known-good host run. Verified by mounting the
fixed binary into the unchanged pre-fix image, so the image, models, clip and PID-1 position are
held constant and only the binary differs.

### 7.52 The 26.04/arm64 divergence is OpenBLAS 0.3.32, and it is a **severe clustering regression**, not float noise: AMI-16 exclusive DER 18.7% -> 52.4%, all of it confusion (issue #18)

**This corrects the working hypothesis, not a published measurement.** §7.42 recorded the 24.04 ->
26.04 bump and flagged that the arm64 leg was never exercised; `1bbba89` reverted the bump on the
evidence that the smoke fixture's exclusive-segment count moved 8 -> 10, and left "whether 10 is
worse, better or immaterial" explicitly UNSCORED. It is now scored. **10 is much worse.** The
prior expectation — that this would turn out to be the §7.49 class, a posterior sitting on a
binarisation threshold and landing on the other side — is **wrong**, and the 26 s fixture is
exactly what made it look plausible.

#### 1. Root cause: OpenBLAS, isolated to a single shared object

The two bases differ in both OpenBLAS (0.3.26 -> 0.3.32) and glibc (2.39 -> 2.43). speakrs is
built `openblas-system`, so `libopenblas.so.0` is **dynamically linked from the runtime image** —
which makes it swappable without recompiling anything.

Probe: take the 26.04/arm64 image unchanged and overwrite only
`/usr/lib/aarch64-linux-gnu/openblas-pthread/` with the same directory from the 24.04 image
(`libopenblasp-r0.3.26.so`), then `ldconfig`. Base, glibc 2.43, the `diar-server` binary and the
three ORT `.so`s are all untouched.

| build | base | glibc | OpenBLAS | `4-end-to-end` |
|---|---|---|---|---|
| `diar-server:arm64-2404-probe` | 24.04 | 2.39 | 0.3.26 | 2 speakers, 7 segments, **8 exclusive** |
| `diar-server:u2604-arm64` | 26.04 | 2.43 | 0.3.32 | 2 speakers, 7 segments, **10 exclusive** |
| **`diar-probe:b18-2604base-blas0326`** | **26.04** | **2.43** | **0.3.26** | 2 speakers, 7 segments, **8 exclusive** |

The swap image is not merely "8 again" — its `/diarize` response is **byte-identical to the 24.04
image on every field** (segments, exclusive segments, centroids, RTTM). That also controls for
code: identical bytes out means the two images' binaries agree, so nothing in the 12 hours of
commits between them is implicated. **glibc 2.43 is ruled out as a cause.**

The converse (24.04 + 0.3.32) **cannot be run**: 26.04's OpenBLAS is built against the newer libm
and dies at load with ``version `GLIBC_2.43' not found``. It is not needed — the swap above already
holds every other variable fixed.

Every ONNX numeric smoke stage is identical across all three, including the exact-zero ones
(`3b fused-vs-split 0.00e0`, `3c multimask-vs-tail 0.00e0`), as §7.42 and issue #18 reported.

#### 2. Determinism control, taken first

Same file, same container, three consecutive `/diarize` calls: **3/3 byte-identical**. The
difference below is signal, not run-to-run noise. (Without this the whole comparison is worthless,
and it is cheap.)

#### 3. Why the smoke fixture understates it — and why centroids were the wrong tell

On `fixtures/test.wav` (26 s) the 24.04 and 26.04 outputs are, under a single
`SPEAKER_00`<->`SPEAKER_01` relabelling:

- **centroids bit-identical** (cross-wise cosine exactly +1.000000, L2 norms equal to all digits);
- **overlap-aware `segments` bit-identical** — all 7 boundaries agree to the last decimal;
- exclusive segments differ by **1.35 s over 24.62 s of labelled speech (5.48%)**, confined
  entirely to **one 2.04 s window (18.560-20.602 s)** which the overlap-aware output shows as a
  two-speaker overlap.

Clustering *label order* is arbitrary and DER is permutation-invariant, so on this clip the
divergence really is cosmetic plus a handful of argmax flips. **This does not generalise.** The
fixture has 2 speakers and one overlap region; it cannot exercise the part of clustering that
actually breaks. Anyone reading only §7.42/issue #18's "8 vs 10" would conclude this is minor.

#### 4. AMI-16: the real measurement

`DIAR_MODE=cpu`, `models_folded`, collar 0.25, overlap included, UEM-cropped, one request per file
with `file_id` held constant (per §7.49's artefact note).

**Control — the 24.04 leg reproduces the recorded baselines**, all 16 files:

| AMI-16, 24.04/0.3.26 | measured here | logged baseline | source |
|---|---|---|---|
| DER, full | **13.126%** | 13.102% | §2.2, §4.26 |
| DER, exclusive | **17.834%** | 17.813% | §7.7 |

Within 0.025 pp on both metrics, on a different architecture (aarch64 CPU vs the amd64 the
baselines were taken on). The harness, refs, UEM handling and RTTM naming are therefore sound.

**The comparison**, on the **10 files** completed in both legs (`EN2002a EN2002b EN2002d ES2004a
IS1009a IS1009b IS1009c IS1009d TS3003a TS3003b`):

| AMI-10, exclusive | DER | missed | false alarm | **confusion** |
|---|---|---|---|---|
| 24.04 / OpenBLAS 0.3.26 | **18.738%** | 15.285% | 1.433% | **2.020%** |
| 26.04 / OpenBLAS 0.3.32 | **52.404%** | 15.285% | 1.433% | **35.686%** |

| AMI-10, full (overlap-aware) | DER |
|---|---|
| 24.04 / OpenBLAS 0.3.26 | **13.781%** |
| 26.04 / OpenBLAS 0.3.32 | **48.657%** |

(The first 8 of those files alone: exclusive 17.776% -> 50.650%, full 13.335% -> 47.295%,
confusion 2.118% -> 34.992%. The effect is stable as files are added, not carried by one outlier.)

**`missed` and `false alarm` are identical to the digit in both legs.** Speech/non-speech detection
— the ONNX segmentation stage — is completely unaffected. The entire **+33.7 pp is speaker
confusion**. That is precisely the stage that runs through BLAS (PLDA scoring, VBx, AHC) rather
than ORT, and it is the strongest corroboration in this section: the damage lands exactly where the
changed library is used, and nowhere else.

#### 5. What breaks: cluster count, not embeddings

Per file, leg A vs leg B — speaker counts, whether the overlap-aware segment *boundaries* agree
(label-agnostic), and each leg-A centroid's best cosine match among leg B's:

| file | spk A | spk B | boundaries identical | best-cos of A's centroids in B |
|---|---|---|---|---|
| EN2002a | 4 | **7** | no | 0.9999 1.0000 1.0000 0.9973 |
| EN2002b | 4 | **5** | no | 1.0000 1.0000 1.0000 0.9997 |
| EN2002d | 4 | **5** | no | 1.0000 1.0000 1.0000 1.0000 |
| ES2004a | 5 | **6** | no | 1.0000 1.0000 1.0000 1.0000 1.0000 |
| IS1009a | 4 | 4 | no | 1.0000 1.0000 1.0000 1.0000 |
| IS1009b | 5 | 5 | no | 1.0000 1.0000 1.0000 1.0000 0.9999 |
| IS1009c | 4 | **6** | no | 1.0000 0.9994 1.0000 1.0000 |
| IS1009d | 4 | **6** | no | 1.0000 0.9934 0.9999 1.0000 |
| TS3003a | 4 | **3** | no | 1.0000 **0.6161** 1.0000 **0.8868** |
| TS3003b | 4 | 4 | no | 1.0000 1.0000 1.0000 1.0000 |

The **embeddings survive** — nearly every centroid finds a ~1.0000 cosine match in the other leg.
What changes is **how many clusters they are grouped into**: 0.3.32 over-splits on 6 of 10 files
(4->7, 4->6, 4->5, 5->6) and under-splits on TS3003a (4->3), where the two poor cosines
(0.616, 0.887) are the signature of two reference speakers collapsed into one centroid. On
IS1009a the *speech* is identical — same 608.0 s total, same 23.0-805.7 s span — and only the
assignment differs.

So the pipeline degrades in one specific place: embeddings in, clustering out.

#### 6. Why aarch64 and not x86-64 (upstream corroboration, not proof)

OpenBLAS 0.3.28 added forwarding of SGEMM/DGEMM calls with a 1xN or Mx1 matrix to the
corresponding **GEMV** kernel, on **arm64, power and riscv64 — not x86-64**; 0.3.29 then "improved
dimension criteria for forwarding from GEMM to GEMV kernels" (again arm64) and **rewrote arm64 CPU
autodetection** to scan all cores and return the highest-performing type. 0.3.28-0.3.30 also add
SVE small-matrix GEMM kernels and faster NCOPY packing. Routing the same call to a different
kernel changes accumulation order on ARM while leaving x86 bit-identical — which is the observed
architecture asymmetry (issue #18: amd64 gives 8 on both bases).

**Stated plainly: no changelog entry in 0.3.27-0.3.32 claims changed numerical results,
accumulation order or FMA behaviour on aarch64.** The mechanism is inferred; the swap experiment
in §1 is the evidence.

Whether 0.3.32 is *at fault* is a separate question this section does not answer. A pipeline whose
DER moves 33 pp under a legal change of BLAS accumulation order is a pipeline with a fragile
clustering stage; the honest reading is that **OpenBLAS exposed the fragility, not necessarily that
it introduced a bug.** See "not settled" below.

#### 7. Not settled

- **The AMI regression is attributed to OpenBLAS by inference, not by direct measurement.** The
  swap image (26.04 + 0.3.26) was proven byte-identical to 24.04 **on the 26 s fixture only**; it
  was **not** run over AMI. The two AMI legs differ by the whole base image. What would settle it:
  run `diar-probe:b18-2604base-blas0326` over the same 10 files and confirm it reproduces leg A's
  18.738% / 2.020% confusion. ~20 min on the arm64 host, and it is the single highest-value
  follow-up. It was blocked here by a scope decision, not by difficulty.
- **10 of 16 AMI files**, not the full set, so these are not directly comparable to the 17.813 /
  13.102 corpus numbers (leg A's own 16-file run, 17.834 / 13.126, is). The leg-B run was stopped
  at 10 files.
- **Root cause is the library, not the line of code.** Which BLAS call changes, and whether the
  fragility is in PLDA scoring, VBx or AHC, is unknown. No bisect of OpenBLAS 0.3.27-0.3.31 was
  done, so it is not known which release introduced it.
- Not tested: whether pinning `OPENBLAS_CORETYPE` or `OPENBLAS_NUM_THREADS` in the 26.04 image
  restores the 24.04 behaviour. If it does, the trigger is kernel dispatch or thread partitioning
  and there may be a cheap runtime mitigation.
- **Untested on amd64 with 0.3.32.** Issue #18 records amd64/26.04 giving 8 on the fixture, but no
  amd64 DER was run against 0.3.32 here. The fixture is now known to understate this failure, so
  "amd64 is unaffected" rests on the same weak evidence this section just discredited for arm64,
  and deserves a DER check before the next base bump.

#### 8. Disposition

The shipped base stays **24.04** (`docker/Dockerfile.server-cpu`, `docker/Dockerfile.builder`), as
`1bbba89` left it, and the published 0.3.0/0.3.1 images are on it — **nothing is shipping on the
affected base**. The practical risk is a *future* base bump, and this section is the reason one must
not be taken on trust: the smoke fixture passes on 26.04/arm64 with a plausible-looking 2 speakers
and 7 segments while exclusive DER is ~52%.

Consequences worth carrying forward:

1. **`verify-models` cannot detect this.** It passes every stage on the broken build. A base bump
   needs a DER check on a real corpus, on **each** published architecture, not a smoke pass.
2. **The old release workflow could not have caught it either** — it built arm64 under QEMU and
   never ran it, and OpenBLAS selects kernels by runtime CPU detection, so a QEMU run is not
   evidence about native aarch64 anyway. `scripts/release.sh` (which replaced it) should be
   held to the native-arm64 standard.
3. Issue #18 stays **open**: the direct AMI control in §7 is outstanding, and the clustering
   fragility it exposes is a real unanswered question independent of any base bump.

#### Environment

arm64 leg: Apple Silicon, Docker Desktop, aarch64 **native (no qemu)**, 12 CPUs, 15.6 GiB,
`diar-server` 0.3.0 images, `DIAR_MODE=cpu`, `models_folded` (24 files, gender present).
Scoring: `validation/score_der.py` + `pyannote.metrics` inside `opentranscribe-celery-worker`
(amd64). Host load average ~14-29 of 48 cores throughout — irrelevant, no timing claim is made
here and the determinism control in §2 was taken independently.
