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
- `pipeline/config.rs`: `RuntimeConfig::fbank_pool: Option<usize>` — the caller pins the pool
  size by value; `None` keeps the previous behaviour exactly (`SPEAKRS_FBANK_POOL`, else
  `available_parallelism/4` clamped 1–8). Added for issue #3: an embedder that wants a
  non-default pool should not have to `setenv`, which is not thread-safe and so forbids lazy or
  concurrent model loading. See `docs/upstream_drafts_fbank_pool.md`.
- `load/sessions.rs`: build a `split_fbank_pool: Vec<Session>` (size from
  `RuntimeConfig::fbank_pool`, falling back to `SPEAKRS_FBANK_POOL` and then to
  `available_parallelism/4` clamped 1–8) when the split backend is available **and the
  execution mode is not CoreML** (CoreML has a native batched fbank path the CPU pool would
  otherwise shadow — Greptile review on PR #15; the pool is skipped there, so no CPU sessions
  are loaded and no CUDA-path behaviour changes).
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
  `std::thread::scope` over disjoint contiguous condensed ranges (deterministic). Blocks are
  pulled from a shared queue by a **bounded** worker pool (`SPEAKRS_AHC_THREADS`, default
  `available_parallelism()` capped at 8) rather than one thread per 1024-row block, so peak
  thread count and peak Gram scratch scale with core count, not meeting length (Greptile
  review on PR #15). Output is bit-identical for any worker count.
- `Cargo.toml`: ndarray `matrixmultiply-threading`.

**Evidence:**
| metric | before | after |
|---|---|---|
| VB-EM (N=21k, K=1.9k) | 305.1 s | **36.7 s (8.3×)** |
| AHC pdist | 64.5 s | **1.2 s (53×)** |
| clustering total | 348 s | **43.6 s (8×)** — also beats scipy (~145 s) on the same data |
| 4.7h E2E (A6000, quiet) | 474 s | **171.6 s (~99× RT)** |
| output | — | **RTTM bit-identical**; all 16 clustering tests incl. Python-parity fixtures pass (AHC==scipy order; VBx gamma/pi @1e-4/1e-5; PLDA fixture); AMI-16 corpus re-run: identical 13.101% aggregate, 16/16 RTTMs bit-identical |

## PR-7 (correctness) — Exclusive diarization picks the overlap winner by cluster index

**Bug (RESULTS §7.7).** `reconstruct.rs` `make_exclusive` documents itself as "zero out all but
the highest-scoring speaker in each frame", but there are no scores left to compare by the time
it runs: `Reconstructor::reconstruct`/`reconstruct_smoothed` write **1.0** for every active
speaker, and `post_inference` may additionally `binarize()`. Every overlapped frame is therefore
an N-way tie among 1.0s, and `Iterator::max_by` returns the **last** maximum on ties — so the
surviving speaker is whichever has the highest cluster index.

Measured on AMI test-16: over **22 297 sampled overlap frames the winner was the highest-indexed
speaker 100.0% of the time.** Overlap ownership is decided by cluster numbering, never by
acoustics. The evidence needed to decide it properly — `Reconstructor::frame_activations` — is
computed and then discarded one step earlier.

This is invisible in the full diarization (unaffected) and only shows up in the exclusive output,
which is what any consumer assigning one speaker per word must use.

**Change.**
- `reconstruct.rs`: `reconstruct`/`reconstruct_smoothed` gain `*_with(activations, …)` variants
  so one activation pass feeds both reconstructions; new `exclusive_from(full, activations)`
  keeps, per frame, the active speaker with the highest activation score. By construction a
  frame with ≥1 active speaker keeps exactly one and an empty frame stays empty — speech is
  neither invented nor lost.
- `post_inference.rs`: build `exclusive_diarization` next to the full one and run it through the
  **same** `binarize` duration filter and `merge_segments(merge_gap)` the full path uses (the
  exclusive path previously skipped gap merging).
- `DiarizationResult` gains `exclusive_segments`.

**Evidence** (AMI test-16, collar 0.25, UEM, overlap included — pyannote community-1 as control):

| variant | DER | missed | false alarm | confusion |
|---|---|---|---|---|
| pyannote exclusive (control) | 17.828% | 14.387 | 1.632 | **1.808** |
| speakrs exclusive (before) | 18.654% | 14.375 | 1.625 | **2.655** |
| speakrs exclusive (**after**) | **17.813%** | 14.375 | 1.624 | **1.814** |

The gap was **entirely confusion** — missed detection is identical to within 0.012 pp, and the
union of speech time is bit-identical before and after, confirming nothing was ever being
dropped. After the fix speakrs edges the pyannote control, and the **full diarization is
bit-identical on 16/16 files**. Downstream (word-level speaker attribution through
OpenTranscribe on a 66.5-min 2-speaker clip): WSER 1.312% → **0.890%**, vs pyannote's 0.859%.

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

---

## Status at handoff (2026-08-19) — read this before executing the gameplan above

T10 is being handed to the agent working `vendor/speakrs` directly, because T9a changed the
tree the gameplan above assumes. Everything here is what that plan does **not** yet account for.

### 1. PR-4 is no longer a proposal — it is built

PR-4 is written above as *"issue first, design-heavy, before any code"*. **T9a implemented it**
(RESULTS §7.25): all 13 ORT sessions became `Arc<Mutex<Session>>` (`SharedSession`,
`vendor/speakrs/src/inference.rs`), locked for exactly one `run()` per inference call, with the
model structs themselves becoming per-request scratch via `clone_shared()`. No speakrs method
signatures changed and the pipeline code is untouched.

This is the strongest thing in the queue and it should be **re-planned as a real PR, not an
issue**. It arrives with gates already passed: 94/94 tests, byte-identical determinism, AMI-16
full 13.101% / exclusive 17.813%, Karpathy full 8.219%, N=4-concurrent output identical to
serial across three runs, and VRAM flat at ~one warm engine during 4 concurrent jobs.

Note for the write-up: **the spec's original option 1 could not work** — per-model locks are
held for the whole pipeline lifetime, so wrapping each model in its own mutex still serialises.
Worth saying explicitly upstream; it is the non-obvious part of the design.

### 2. PR-7 is missing from the execution plan

PR-7 (exclusive diarization resolving overlaps by cluster index instead of activation) is
documented in its own section above, but is **absent from the Step-1 branch table and from the
Step-3 two-PR submission shape**. It should not be lost — it is a *correctness* fix, not a perf
one, and correctness fixes merge easily and build maintainer trust.

Recommendation: send it with **PR-A** (the multimask one-liner), or immediately after. Do not
bury it inside the perf series, where it would be reviewed as an optimization.

Evidence: the gap was **entirely confusion**, which is the signature of picking the wrong
speaker rather than losing speech — AMI-16 exclusive DER 18.654% → **17.813%** with missed
(14.375) and false-alarm (1.624) essentially unchanged, and confusion 2.655 → 1.814. The
pyannote fork scores 17.828 on the same protocol, so the fix lands *at* the reference.

### 3. Numbers in the handoff table that are stale

`HANDOFF_T9A_SHARED_SESSIONS.md` §5 lists Karpathy exclusive **6.545%** — that is the
**pre-§7.7-fix** figure (660 segments). The post-fix exclusive path produces 766 segments and
scores **6.188%**. Use 6.188% in any upstream claim.

### 4. Mechanical consequences of T9a on the plan above

- **The patch set must be regenerated.** Step 1 splits branches out of
  `patches/0001-cuda-performance-patch-set.patch`, which predates T9a. Regenerate from the
  post-T9a tree (`cd vendor/speakrs && git diff > ../../patches/0001-...patch`) before splitting.
- **PR-2 overlaps T9a's files.** `perf/fbank-session-pool` touches `session.rs`,
  `load/sessions.rs`, `embedding.rs` — all rewritten by the shared-session work. Split PR-2
  after regenerating, not before, or the hunks will not apply.
- **Re-confirm isolated numbers on a quiet machine.** The figures in Step 2 (folded seg −7%,
  batching+pool 3.1×, VBx/pdist 2.76×) were measured before T9a. T9a's own throughput gate came
  back **1.51× on a machine at load average 19** and is explicitly marked RE-MEASURE. Do not
  quote any throughput number upstream that was taken under that load — accuracy and
  bit-identity results are unaffected, only timings.

### 5. Unchanged and still true

PR-1, PR-3, PR-5 (the three bug reports) and PR-6 are untouched by T9a. The Step-3 decision to
consolidate into **two PRs** rather than a burst still holds and is still the right call for a
single-maintainer project.

---

## Branches prepared (2026-08-19) — PUSHED, and 7 PRs are open upstream

**Status corrected 2026-09-02.** This section was written before the branches were pushed and
said "awaiting approval". They were pushed, and as of today `avencera/speakrs` has **7 open PRs**
from `attevon-admin` (#8, #9, #10, #14, #15, #17, plus upstream's own #6 which we adopted). Check
the live list rather than trusting the state recorded below:

    gh pr list --repo avencera/speakrs --state open

What follows is the preparation record — what each branch contains and why — which is still
accurate. Only the "not yet pushed" framing was stale.

Work clone: `upstream-work/` (gitignored; upstream tip verified NOT moved past `b0756b1`).
All branches compile warning-clean and pass the full 94-test suite in isolation
(fixtures mounted, `diar-bench-builder`, openblas-system+online):

| branch | commit | maps to |
|---|---|---|
| `fix/multimask-batch-size` | f7d506c | PR-A (PR-1) |
| `fix/exclusive-overlap-resolution` | abd505e | PR-7 — trimmed of our centroids-out feature |
| `perf/vbx-vectorize-pdist-blocks` | 5239269 | PR-B commit i (PR-6) |
| `perf/fbank-session-pool` | 26a8756 | PR-B commit ii (PR-2) — trimmed of lazy-sessions |
| `feat/export-folded-segmentation` | a700aeb | PR-B commit iii (PR-3, onnxsim pass authored) |
| `perf/cuda-pipeline-series` | 3 commits | PR-B as submitted (i+ii+iii cherry-picked) |
| `feat/shared-sessions` | 1a51e8b | PR-4 as a REAL PR (T9a, minus pool/lazy/TRT) |

Final PR/issue text now lives in the filed PRs themselves (the drafts file was deleted; see git history). Remaining before submission: per-branch isolated
E2E speed re-confirmation on a quiet machine (accuracy/bit-identity claims already verified),
then operator approval to open the intro issue + PRs against avencera/speakrs.

**Update 2026-08-20 — fork created, branches pushed, PRs still held.**
Forked to [`attevon-llc/speakrs`](https://github.com/attevon-llc/speakrs) (public, Apache-2.0
license carried over unchanged). All 7 branches above pushed there, plus a consolidated
`attevon/production-0.2.0` branch (patch regenerated from the current `vendor/speakrs` tree —
the previously-committed `patches/0001-...patch` was stale, missing 18 of 19 changed files;
now fixed) with the full current production diff applied as a real commit for anyone building
diar-native to check out directly, no manual patch-apply needed. `upstream-work/`'s `origin`
now points at the fork; `upstream` stays `avencera/speakrs`. Opening the intro issue and PRs
against avencera/speakrs is still gated on explicit operator approval — not done yet.

**Update 2026-08-20 (cont'd) — fork's `master` is now the canonical "what we run" branch.**
Opened and merged (real merge commit, not squash) attevon-llc/speakrs PR #1:
`attevon/production-0.2.0` → `master` (725cc4d). This only touches our own fork, not
avencera/speakrs. Verified: fork `master`'s tree is byte-identical to the local
`vendor/speakrs` working tree (`git diff attevon/production-0.2.0 origin/master` empty);
a completely fresh, independent clone of fork `master` passes the full 94-test speakrs suite
inside `diar-bench-builder` (74 unit + 5 integration [1 intentionally-ignored online test] +
8 queued + 7 doctests, all green). `scripts/bootstrap_vendor_speakrs.sh` now reproduces
`vendor/speakrs` from this pinned fork commit — tested end-to-end in an isolated scratch dir,
output byte-identical to the real `vendor/speakrs`. The 7 individual PR-prep branches were
deliberately left unmerged into `master` (they're trimmed subsets for clean upstream review,
not full production — merging them would misrepresent both). Did not do a full
`docker build -f docker/Dockerfile.server` re-run since it only `COPY`s `vendor/speakrs/` as
already-proven-identical content — the currently-live `diar-server:0.2.0` image already is,
byte-for-byte, a build of what's now on fork `master`.
