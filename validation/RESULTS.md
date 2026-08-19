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

### 4.12 CPU leg (M7)

- fork eager CPU, 0.5h file: 1123 s = **1.7× RT** (machine under moderate load).
- speakrs `cpu` mode: running.

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
