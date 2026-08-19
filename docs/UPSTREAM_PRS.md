# Upstream Contribution Plan — avencera/speakrs

We adopted speakrs (Apache-2.0) as OpenTranscribe's diarization engine and found/fixed several
issues while validating it against the production pyannote fork. This file is the submission
plan: each PR's motivation, evidence, and change summary. The actual diffs live in
[`patches/`](../patches/) (generated from our vendored copy at pin `b0756b1`); all measurement
receipts are in [`validation/RESULTS.md`](../validation/RESULTS.md) (§ references below).

Ordering matters: PR-1 and PR-2 are independent; PR-3 is export-script only; PR-4 is a design
conversation to open as an issue first.

---

## PR-1 — Fix multimask batching: exporter/loader batch-size filename mismatch

**Bug (RESULTS §4.15).** `load/sessions.rs` loads the batched multimask tail via
`multi_mask_model_path(model_path, PRIMARY_BATCH_SIZE /* 64 */)` →
`wespeaker-multimask-tail-b64.onnx`, but `scripts/export_models.py` only writes
`wespeaker-multimask-tail-b32.onnx` (`MULTI_MASK_BATCH_SIZE = 32`). Result: the batched session
is silently `None`, `multi_mask_batch_size()` returns 1, and the CUDA pipeline flushes **one
chunk at a time** (trace: `flushes == chunks`) — per-chunk fbank and batch-1 GPU predicts.
Runtime buffers are sized for 32, so a *true* batch-64 graph under the expected name crashes the
embedding worker ("receiver disconnected") — verified empirically.

**Fix options** (either; we suggest a): (a) change the loader to request batch
`MULTI_MASK_BATCH_SIZE`; (b) export a `-b64`-named batch-32 graph. One-line either way, plus a
regression assert that the loaded session's batch dim matches the runtime buffer size.

**Evidence to attach:** trace excerpts (`flushes=22 chunks=22` before / `flushes=33 chunks=1041`
after), RTTM bit-identity between batch-1 and batched paths, crash repro with a real b64 graph.

## PR-2 — Parallel fbank: session pool + thread-cap override

**Problem (RESULTS §4.15/§4.16).** With batching fixed, CPU fbank is the CUDA-path bottleneck:
`fbank_ms=28182` of ~37 s wall (76%) on a 36-min meeting (ES2004a, RTX 3080 Ti) vs
`gpu_predict_ms=3572`. The single fbank session is capped at 4 intra-op threads
(`session.rs: .min(4)`), and intra-op threads don't scale it (4→24 threads: 35.4 s → 32.1 s).
`--chunk-emb-workers` has no effect on the CUDA path.

**Change (patches/0001).**
- `session.rs`: `SPEAKRS_FBANK_THREADS` env override for the intra-op cap.
- `load/sessions.rs`: build a `split_fbank_pool: Vec<Session>` (size `SPEAKRS_FBANK_POOL`,
  default `available_parallelism/4` clamped 1–8) when the split backend is available.
- `fbank.rs`: in `compute_chunk_fbanks_batch`, fan chunks across the pool with
  `std::thread::scope` (session-local input buffers; deterministic result ordering).

**Measured:** ES2004a E2E 39.4 s → **12.9 s (3.1×)** at pool=8, RTTM **bit-identical**;
AMI test-16 aggregate DER unchanged (13.100% → 13.101%, see PR-5 note on the 0.001pp);
full 16-meeting AMI corpus at **89× realtime on an RTX 3080 Ti** (was ~31–49×).
Upstream may prefer rayon or wiring this into `RuntimeConfig` instead of env vars — happy to
rework; the perf shape is the point.

## PR-3 — Export a constant-folded segmentation graph

**Problem (RESULTS §4.1/§4.2).** The exported segmentation ONNX retains SincNet's filter-synthesis
subgraph (`Sin`×2/`Cos`×2/`If`×1 computed from *frozen* parameters every forward). On ORT CUDA EP
these ops trigger CPU fallback + Memcpy insertion: 190 ms vs 96.6 ms per batch-32 (2.0×) for
folded vs unfolded in serving tests, and −7% E2E in the speakrs pipeline.

**Change:** run onnxsim (or equivalent constant folding) in `scripts/export_models.py` after
export. Folding is **bit-exact** (max_abs_diff 0.0, argmax mismatch 0; graph shrinks 179→40
nodes). Zero runtime changes — speakrs loads the same filenames.

## PR-4 (issue first) — Shared-session concurrent pipelines

**Goal.** N concurrent diarizations without N× VRAM: ONNX Runtime sessions are thread-safe for
concurrent `Run()` and weights load once per session. speakrs' `&mut` discipline protects its
scratch buffers, not the weights. Proposal: split per-request buffer state from `Arc`-shared
sessions so multiple `DiarizationPipeline` instances share one model set. This mirrors a
production pattern (Celery workers dispatching 2–8 concurrent jobs to one engine process).
Design-heavy — open as an issue with our use-case before any code.

## PR-5 / bug reports (no patch yet)

1. **ORT-CUDA teardown crash**: `corrupted double-linked list` (glibc) at process exit after
   results flush, `cuda` mode — ORT EP unload vs mimalloc interplay (RESULTS §5.1).
2. **Batched-fbank graph numeric deviation**: `wespeaker-fbank-b32.onnx` output differs slightly
   from the single-chunk `wespeaker-fbank.onnx` on identical audio (dynamo export difference);
   downstream this shifted 16/2994 RTTM lines and +0.001pp AMI DER. Suggest exporting the batched
   graph from the same traced function, or documenting the tolerance (RESULTS §4.16 area).
3. **`--chunk-emb-workers` is a no-op on the CUDA path** — either wire it up or document it as
   CoreML-only.

## PR-6 (flagship) — Vectorized VBx + threaded blocked pdist: 8× clustering, output-identical

**Problem.** At scale (4.7 h meeting → N=21,418 filtered embeddings, K=1,902 AHC seed clusters,
D=128), clustering dominates E2E wall: VB-EM 305 s (scalar O(N·K·D) loops ×20 iterations,
including a per-speaker penalty recomputed per sample), AHC pdist 64.5 s (per-pair scalar loop).
Clustering = 74% of the file's 474 s total.

**Change** (in `patches/0001-cuda-performance-patch-set.patch`):
- `vbx.rs`: M-step as one `gammaᵀ·rho` matmul; E-step as one `rho·alphaᵀ` matmul + penalty
  vector computed once per iteration; fused logsumexp/gamma row pass. Same f64 math, same
  iteration/early-stop semantics.
- `ahc.rs`: condensed distances via blocked Gram matmul (`d² = |a|²+|b|²−2ab`) with
  `std::thread::scope` over disjoint contiguous condensed ranges (lock-free, deterministic).
- `Cargo.toml`: ndarray `matrixmultiply-threading`.

**Evidence:**
| metric | before | after |
|---|---|---|
| VB-EM (N=21k, K=1.9k) | 305.1 s | **36.7 s (8.3×)** |
| AHC pdist | 64.5 s | **1.2 s (53×)** |
| clustering total | 348 s | **43.6 s (8×)** — also beats scipy (~145 s) on the same data |
| 4.7h E2E (A6000, quiet) | 474 s | **171.6 s (~99× RT)** |
| output | — | **RTTM bit-identical**; all 16 clustering tests incl. Python-parity fixtures pass (AHC==scipy order; VBx gamma/pi @1e-4/1e-5; PLDA fixture); AMI-16 corpus re-run: identical 13.101% aggregate, 16/16 RTTMs bit-identical |

## Consolidated E2E story (the cover-letter numbers, quiet-machine A6000)

Cumulative effect of the full patch set (multimask fix + fbank pool + folded seg + VBx/pdist)
vs stock speakrs `b0756b1`, using self-exported community-1 models:

| corpus / file | stock speakrs | patched | pyannote fork (GPU-optimized production baseline) |
|---|---|---|---|
| ES2004a (36 min AMI) | 39.4 s | **12.9 s (3.1×)** | ~27 s |
| 4.7h 8-speaker file | 474 s | **171.6 s (2.8×)** | 349 s |
| AMI test-16 corpus | ~31–49× RT | **105× RT** | 80× RT |
| accuracy (AMI/Karpathy/VoxConverse) | 13.100 / 8.219 / 4.847% | **identical (bit-level RTTMs)** | 13.093 / 8.194 / 5.099% |

Framing for upstream: the patch set makes speakrs-CUDA decisively faster than a heavily
GPU-optimized pyannote deployment while keeping its accuracy word-for-word — with the VoxConverse
number *better* than pyannote's. Every claim has a runnable reproduction in
`validation/TESTPLAN.md` §4 and raw artifacts under `results/`.

## Submission gameplan (execute when ready — est. 1-2 days total)

**Step 0 — prep (once):**
1. `gh repo fork avencera/speakrs --clone` into a work dir (NOT vendor/ — keep vendor pinned).
2. Rebase check: `git log b0756b1..origin/main` — if upstream moved, re-run the 4.7h + ES2004a
   A/B on the new tip BEFORE porting patches (their changes may overlap ours).
3. Open a short intro issue first: "We validated speakrs against a production pyannote
   deployment (AMI/VoxConverse/hand-labeled corpora) and have a patch series with benchmark
   receipts — filing over the next days." Sets context, signals seriousness, surfaces
   maintainer preferences early (env vars vs RuntimeConfig, rayon vs std threads).

**Step 1 — split the monolithic patch into per-PR branches** (from
`patches/0001-cuda-performance-patch-set.patch`; hunks map cleanly):
| branch | files | PR |
|---|---|---|
| `fix/multimask-batch-size` | `load/sessions.rs` (1 line) or `export_models.py` | PR-1 |
| `perf/vbx-vectorize-pdist-blocks` | `vbx.rs`, `ahc.rs`, `Cargo.toml` (ndarray threading) | PR-6 (flagship) |
| `perf/fbank-session-pool` | `session.rs`, `load/sessions.rs`, `fbank.rs`, `embedding.rs` | PR-2 |
| `feat/export-folded-segmentation` | `scripts/export_models.py` (+onnxsim dep) | PR-3 |

**Step 2 — per-branch validation (the isolation matrix; each ~30 min on our hardware):**
For each branch alone on upstream tip: (a) `cargo test --release` with fixtures mounted +
`RUST_MIN_STACK` (document that both quirks are pre-existing); (b) ES2004a + 4.7h E2E, RTTM
diff vs unpatched; (c) record isolated speedup. Attach per-PR numbers — maintainers trust
isolated effects over combined ones. Known isolated numbers so far: folded seg −7%;
batching+pool 3.1× (ES2004a); VBx/pdist 2.76× (4.7h) — re-confirm on upstream tip.

**Step 3 — submission shape (DECIDED 2026-08-19: consolidated to TWO PRs — a burst of 4+ PRs
overwhelms a single-maintainer project):**
1. **PR-A: the one-line multimask batching bug fix** (+ crash repro). Kept separate because a
   trivial merge should never be held hostage by discussion of the perf series — and it builds
   trust for PR-B.
2. **PR-B: "CUDA pipeline performance series"** — ONE PR, THREE clean commits reviewable
   independently: (i) VBx vectorization + threaded blocked pdist (the 8× clustering win,
   output-bit-identical), (ii) fbank session pool + thread override (note in the description:
   happy to rework env vars into `RuntimeConfig`), (iii) folded-segmentation export.
   Commit-per-change lets the maintainer drop/rework one piece without stalling the rest.
   Cover letter = the consolidated E2E story table above.
3. Bug **reports** (teardown crash, batched-fbank numeric deviation, chunk-emb-workers no-op)
   and the shared-sessions design proposal go as ordinary issues on their own schedule — issues
   are not review burden the way PRs are.

**Step 4 — PR body template** (each PR): problem → root cause → change summary → isolated
benchmark table (hardware named: RTX A6000 / 3080 Ti, quiet-machine protocol) → accuracy proof
(bit-identity / fixture tests / corpus DER) → reproduction commands. Link the intro issue.

**Contingencies:** upstream unresponsive after ~2-3 weeks → we're unblocked regardless (vendored
pin + patch file is our production path; PLAN decision #1). Maintainer wants different shape
(rayon, config plumbing) → mechanical rework, numbers unchanged. Upstream tip diverged heavily →
re-validate on tip; our harness makes each re-run ~30 min.

**Non-negotiables:** quiet-machine numbers only (RESULTS §4.11); never attach gated-model
artifacts (community-1 derivatives) to public PRs — fixtures/models stay local; benchmarks
reference our corpora by name, raw RTTMs available on request.
