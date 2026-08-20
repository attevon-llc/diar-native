# Upstream PR drafts — avencera/speakrs (T10)

Ready-to-post text for each submission unit. Branches live in `upstream-work/` (local work
clone, gitignored; based on upstream tip `b0756b1`, which has NOT moved since our pin).
**Nothing is pushed** — per the standing constraint, opening the fork/issues/PRs waits for
explicit approval. Numbers below are quiet-machine only; the loaded-machine figures from
2026-08-19 (load avg ~19) are excluded per RESULTS §4.11.

Submission order (decided in UPSTREAM_PRS.md, still right):
1. intro issue → 2. PR-A (multimask fix) → 3. PR exclusive fix → 4. PR-B perf series →
5. PR shared sessions → 6. bug-report issues.

---

## 0. Intro issue

**Title:** Validated speakrs against a production pyannote deployment — patch series with
benchmark receipts incoming

> We adopted speakrs as the diarization engine for an open-source transcription app and
> validated it against a GPU-optimized pyannote community-1 deployment on AMI test-16,
> VoxConverse dev (216 files), and a hand-labelled 66.5-min corpus. Along the way we found
> one correctness bug, one silent-perf bug, and built a set of CUDA-path optimizations, all
> with DER/bit-identity receipts. Filing over the coming days:
> 1 correctness fix (exclusive diarization overlap resolution), 1 one-line bug fix
> (multimask batching), a 3-commit CUDA perf series (vectorized VBx, fbank session pool,
> folded segmentation export), and a shared-sessions change enabling N concurrent
> diarizations at one engine's VRAM. Preferences welcome on env vars vs `RuntimeConfig`
> plumbing and std::thread vs rayon — happy to rework shape, the numbers are the point.

## 1. PR-A — branch `fix/multimask-batch-size` (commit f7d506c)

**Title:** Fix multimask batching: load the batched tail at MULTI_MASK_BATCH_SIZE

Body = commit message + this evidence block:

> **Before:** trace shows `flushes == chunks` (batch size 1, per-chunk fbank).
> **After:** `flushes=33` for 1041 chunks on a 36-min AMI meeting; RTTM bit-identical to the
> batch-1 path. **Crash repro:** exporting a true batch-64 multimask graph under the
> expected `-b64` name kills the embedding worker ("receiver disconnected") because runtime
> buffers are sized 32 — reproduced on RTX 3080 Ti.

## 2. PR — branch `fix/exclusive-overlap-resolution` (commit abd505e)

**Title:** Exclusive diarization resolves overlaps by cluster index, not acoustics — fix by
keeping the activation scores

Body = commit message. Key table (AMI test-16, collar 0.25, UEM, overlap included, pyannote
community-1 as control):

| variant | DER | missed | false alarm | confusion |
|---|---|---|---|---|
| pyannote exclusive (control) | 17.828% | 14.387 | 1.632 | 1.808 |
| speakrs before | 18.654% | 14.375 | 1.625 | **2.655** |
| speakrs after | **17.813%** | 14.375 | 1.624 | **1.814** |

Full diarization bit-identical on 16/16 files; union of speech time identical before/after
(nothing dropped — the wrong speaker was being *named*). Downstream word-level attribution
WSER 1.312% → 0.890% on a 2-speaker 66.5-min clip.

## 3. PR-B — branch `perf/cuda-pipeline-series` (3 commits: 1f4a076, 90200c1, 7687da7)

**Title:** CUDA pipeline performance series: vectorized VBx + threaded pdist, fbank session
pool, folded segmentation export

Cover letter = consolidated table (quiet-machine A6000, self-exported community-1 models):

| corpus / file | stock b0756b1 | patched | GPU-optimized pyannote fork |
|---|---|---|---|
| ES2004a (36-min AMI) | 39.4 s | **12.9 s (3.1×)** | ~27 s |
| 4.7 h 8-speaker file | 474 s | **171.6 s (2.8×)** | 349 s |
| accuracy | AMI 13.100 / Karpathy 8.219 / Vox 4.847% | **bit-identical RTTMs** | 13.093 / 8.194 / 5.099% |

One commit per change so any piece can be dropped/reworked independently. Isolated numbers:
VBx/pdist 8× clustering (305→36.7 s VB-EM, 64.5→1.2 s pdist, 4.7 h file); fbank pool 3.1×
E2E (ES2004a); folded seg −7% E2E / 2.0× per batch-32 on ORT-CUDA. Env-var knobs
(`SPEAKRS_FBANK_THREADS`, `SPEAKRS_FBANK_POOL`) are the smallest surface — happy to move
into `RuntimeConfig`.

## 4. PR — branch `feat/shared-sessions` (commit 1a51e8b)

**Title:** Share ORT sessions across pipeline handles: N concurrent diarizations at one
engine's VRAM

Body = commit message. The non-obvious design point to state explicitly:

> Wrapping each model (or session) in a mutex without restructuring does NOT deliver
> concurrency: `DiarizationPipeline` borrows both models `&mut` for its whole lifetime, so
> any lock acquired through those borrows is held per-job and jobs still serialize. The
> weights/scratch split is the precondition. `clone_shared()` keeps every method signature
> and all pipeline code unchanged.

Evidence: full suite green; single-job outputs byte-identical; 4 concurrent jobs identical
to serial (3 independent runs); VRAM flat at one warm engine during 4 concurrent jobs; 2
concurrent jobs inflate each other 1.3× instead of 2.0×.

Note for review sequencing: textually overlaps the fbank-pool commit in
`load/sessions.rs`/`embedding.rs`/`fbank.rs` — whichever lands second rebases; we run the
combined form in production (pool becomes `Vec<SharedSession>`).

## 5. Issues (no code)

1. **ORT-CUDA teardown crash** — `corrupted double-linked list` at process exit after
   results flush (cuda mode, ort 2.0.0-rc.12 + ORT 1.24.2); reproduced with both mimalloc
   and glibc malloc.
2. **Batched-fbank graph numeric deviation** — `wespeaker-fbank-b32.onnx` differs slightly
   from the single-chunk graph on identical audio (dynamo export difference); shifted
   16/2994 RTTM lines, +0.001 pp AMI. Suggest exporting both from one traced function or
   documenting the tolerance.
3. **`--chunk-emb-workers` is a no-op on the CUDA path** — wire it up or document as
   CoreML-only.

## Status / remaining before submission

- [x] 7 branches created on upstream tip; all compile warning-clean
- [x] Full test suite per branch (fixtures mounted) — see matrix result in RESULTS/T10 notes
- [ ] Per-branch isolated E2E speed re-confirmation on a quiet machine (folded seg −7%,
      pool 3.1×, VBx 2.76× were measured pre-T9a; re-run before quoting in the PR)
- [ ] Operator approval → fork, push branches, open intro issue, then PRs in order
- Gated-model artifacts (community-1 derivatives) are never attached; corpora referenced by
  name with raw RTTMs available on request.
