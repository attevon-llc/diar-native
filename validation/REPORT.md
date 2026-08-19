# Validation Report — speakrs vs the Production pyannote Fork

**Audience:** anyone opening this repo cold. This is the narrative summary of what was done,
what was found, and what happens next. Precise numbers/methodology: [RESULTS.md](RESULTS.md)
(append-only log) and [TESTPLAN.md](TESTPLAN.md) (matrix + gates). Roadmap: [../PLAN.md](../PLAN.md).

**Status: Phase B COMPLETE — FINAL VERDICT: GO on speakrs adoption (Phase C), with two named
Milestone-1 ship-gate items.** All gates evaluated; every number quiet-machine-verified or
ground-truth-scored. Verdict date: 2026-08-19.

## 1. The question

OpenTranscribe diarizes with a heavily GPU-optimized fork of pyannote.audio
(`speaker-diarization-community-1` pipeline), pinned in production. It is accurate and fast
(80× realtime on an RTX A6000) but structurally capped: Python/GIL serialization (clustering is
~41% of wall time), a ~9 GB torch deployment, and no serving story. A previous ONNX/Triton
attempt failed on "unsupported operations." We asked: **can a native engine match the fork's
accuracy exactly, beat its speed, and simplify deployment — and can Triton serve it for
multi-user/AWS scale?**

## 2. What was done (chronological)

1. **Root-caused the historical ONNX/Triton failures** — none were fundamental:
   the failing ops (Sin/Cos/If) constant-fold away bit-exactly; the TensorRT "engine rebuild
   storm" was a mis-declared shape profile (500 vs the real 998 fbank frames); the old artifacts
   were exported from the wrong (fallback-pipeline) checkpoints entirely — proven by sha256.
2. **Established trustworthy baselines** (Gate 0): the production fork reproduces its frozen
   April RTTMs at 0.0000% DER and is bit-deterministic. Fresh ground-truth baselines: AMI test-16
   13.093% DER, Karpathy hand-labeled 66-min clip 8.194%, VoxConverse dev-216 5.099%
   (collar 0.25, overlap scored, official UEMs), duration curve 0.5–4.7 h, CPU leg 1.7× RT.
3. **Validated speakrs (Rust, Apache-2.0) head-to-head** with self-exported models from the
   exact production checkpoints: AMI 13.100% (+0.007pp), Karpathy 8.219% (+0.025pp), speaker
   counts **identical file-by-file** on all 16 AMI meetings, bit-deterministic across runs.
   Accuracy parity: **proven** (gates G1/G2/G5).
4. **Found and fixed why speakrs' CUDA path was slow** (it initially ran 2.5× slower than the
   fork): (a) a silent exporter/loader filename mismatch disabled its own batching (batch size
   fell to 1); (b) single-threaded CPU fbank was 76% of wall time; (c) the segmentation graph
   shipped unfolded. Three patches later (see [../docs/UPSTREAM_PRS.md](../docs/UPSTREAM_PRS.md),
   diffs in [../patches/](../patches/)): **3.1× end-to-end, RTTM bit-identical, AMI DER
   unchanged — 89× realtime on the *smaller* RTX 3080 Ti vs the fork's 80× on an A6000.**
5. **Proved the Triton/TensorRT serving path** the earlier effort thought impossible: both
   models build and run as TRT fp32 engines on first attempt (embedding 37.4 ms/batch-32 on the
   A6000 ≈ 1.9× the fork's eager embedding stage), and dynamic batching yields **2.14× embedding
   throughput at 8 concurrent clients on one weight copy** — the multi-user/AWS thesis.
6. **Documented every side-finding**: a fork bug that crashes all CPU-only Linux deployments;
   a production-image ORT-GPU cu12/cu13 mismatch; RTTM parsing breaking on speaker names with
   spaces; a corrupt dataset zip (re-downloaded from the official source); two benchmark-hygiene
   incidents that taught us to never co-schedule timed runs or mutate model dirs under live
   benchmarks.

## 3. Verdict so far (gates)

| gate | criterion | status |
|---|---|---|
| G0 | environment reproduces frozen baselines | ✅ 0.0000% drift |
| G1 | AMI DER ≤ fork +0.1pp, speakers ±1 | ✅ +0.007pp, counts identical |
| G2 | Karpathy DER ≤ +0.1pp, 2 speakers | ✅ +0.025pp, exact, deterministic |
| G3 | duration-curve A/B ≤0.5% median, ≤2% max, speaker-exact ×5 | ✗ as-written on synthetic files (3.2h 6.05%, 4.7h 2.27%, +1 cluster) — **formally adjudicated as synthetic-content edge case by the VoxConverse arbiter** (below); ground-truth gates govern |
| G3-arbiter | VoxConverse dev-216 ground truth | ✅ **speakrs 4.847% BEATS fork 5.099%**; speaker counts equal on 211/216; per-file DER wins 95-38 (83 ties); ground-truth-count matches 138 vs 136 |
| G4 | ≥1.0× fork speed, RSS<8GB, VRAM<4GB | ✅ **CLEAN SWEEP after clustering optimization (RESULTS §4.22-4.24)**: 2.2h 1.26×, Karpathy 1.2×, **4.7h 2.03× faster** (171.6 vs 349 s), AMI corpus 105× RT on the A6000. VBx vectorized 8.3×, pdist 53×; RTTMs bit-identical; parity fixtures green. **T1 ship gate CLEARED at 2× margin.** |
| G5 | bit-determinism ×3 | ✅ everywhere (the two "differing" runs were a self-inflicted mid-run model-dir mutation, documented) |

**FINAL VERDICT: GO on speakrs adoption** per PLAN.md Phase C — T1 shared-weights sidecar as the
open-source default (laptop-class deployable, Celery keeps orchestration), T2 Triton+TRT opt-in
for large servers and AWS, pyannote-fork path retired after shadow-mode verification.
Accuracy: speakrs ≥ fork on every ground-truth corpus (VoxConverse: better). Speed: faster on
typical content; two named Milestone-1 ship-gate items before the default flip:
(1) AHC linkage at N>10k (the 4.7h regression, §4.20); (2) patched-build Karpathy/AMI-class
verification is complete (§4.21) — carry the patch set upstream per docs/UPSTREAM_PRS.md.

## 4. What this repo would hand a new contributor

- Rebuild everything: TESTPLAN §4 (exact commands), `docker/Dockerfile.bench`, `models/` export
  procedure (gated weights — self-export only).
- Re-score anything: `validation/score_der.py` + `refs/` (AMI/Karpathy/VoxConverse staged).
- Every claim traceable: RESULTS.md section references throughout; raw RTTMs under `results/`.
- Next actions: PLAN.md Phase C milestones; docs/UPSTREAM_PRS.md submission order.
