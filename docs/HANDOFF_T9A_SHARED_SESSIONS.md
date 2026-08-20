# Handoff — T9a: shared sessions so diarization stops serialising

**Status: IMPLEMENTED 2026-08-19 — see RESULTS §7.25.** Engine-level gates passed (determinism,
DER parity full+exclusive, N=4 concurrency identity, VRAM flat); throughput leg needs a quiet
machine; app-level flip + §1 re-measurement pending. NOTE: the §5 Karpathy exclusive gate value
6.545 % below is the pre-§7.7-fix number — post-fix output scores 6.188 % and is bit-identical
to `results/exclusive_study/exclusive_fixed`.
**Repos:** `diar-native` (this repo) — vendored crate at `vendor/speakrs`, wrapper at
`crates/diar-core`, server at `crates/diar-server`. `transcribe-app` is *not* touched by this task.
**Corpora, file paths and every baseline number:** `docs/TEST_CORPORA_AND_BASELINES.md`.
**Read first:** `validation/RESULTS.md` §7.24 (the measurement), §4.11 (benchmark hygiene),
§4.26 (build traps), `docs/DETAILED_SPECS.md` S-T9a, `PLAN.md` decision #4.

---

## 1. The problem, measured

`crates/diar-server/src/main.rs` holds `engine: Mutex<DiarEngine>` behind a `Semaphore`. So
`DIAR_MAX_INFLIGHT` bounds *queueing*, not execution: every `/diarize` request serialises.

Running the two largest corpus files concurrently, read from the `diarize_request_sent` /
`diarize_joined` columns:

| file | diarize span solo | diarize span, 2 concurrent | GPU stage |
|---|---|---|---|
| Karpathy 66.5 min | 37.5 s | **76.1 s (2.0×)** | 52.9 s → 80.0 s |
| seed 2.1 h | ~58 s | **113.8 s (2.0×)** | → 119.4 s |

Both spans **exactly doubled** — each job waiting out the other's diarization. The pair still beat
sequential (131 s vs 174.7 s) because transcription overlaps, but ~11 s of that window is pure lock
wait, and it degrades with concurrency: diarization is 37-58 s of work per large file, so from
three concurrent large files the sidecar — not transcription — is the binding constraint.

**Do not "fix" this by running N engines.** A second `DiarEngine` costs another ~4.1 GB (§7.14),
and the 12 GB target already holds a 7 575 MiB floor. PLAN decision #4 makes shared weights a
requirement, not a preference.

## 2. Why the spec's option 1 does not work

`docs/DETAILED_SPECS.md` S-T9a says to try mutex-per-session first. **That does not deliver
concurrency on this code shape**, and the reason is structural:

```rust
// vendor/speakrs/src/pipeline.rs:197
pub struct DiarizationPipeline<'a> {
    seg_model: &'a mut SegmentationModel,
    emb_model: &'a mut EmbeddingModel,
    ...
}
```

The pipeline borrows **both** models mutably for its entire lifetime. Wrapping each model in its
own `Mutex` just moves the lock — it is still held for the whole job, so two jobs serialise exactly
as they do now. The contention profile S-T9a predicts ("seg and emb lock independently, so phases
interleave") only materialises if locks are acquired *per inference call*, which they are not.

Correct the spec when you land this.

## 3. What actually has to change

Split immutable weights from per-request state, as decision #4 describes.

**Sessions to share** (13 total — these are the VRAM, and ORT's `Run` is thread-safe at the C API
level even though the Rust binding takes `&mut self`):

| where | count |
|---|---|
| `SegmentationModel` (`src/inference/segmentation.rs:56`) — `session`, `primary_batched_session`, + CoreML variants | 3 |
| `OrtEmbeddingState` (`src/inference/embedding.rs:73`) — `session`, `primary_batched_session`, `split_fbank_session`, `split_fbank_pool` (Vec), `split_fbank_batched_session`, `split_tail_session`, `split_tail_batched_session`, `split_primary_tail_batched_session`, `multi_mask_session`, `multi_mask_batched_session` | 10 |

**State that must become per-request `Scratch`** (~14 items):

| where | items |
|---|---|
| `SegmentationModel` | `input_buffer`, `primary_batch_input_buffer`, `cached_single_input_shape`, `cached_batch_input_shape` |
| `EmbeddingBuffers` (`src/inference/embedding.rs:125`) | 12 ndarray buffers: `multi_mask_fbank_buffer`, `multi_mask_masks_buffer`, `waveform_buffer`, `weights_buffer`, `primary_batch_waveform_buffer`, `primary_batch_weights_buffer`, `split_waveform_buffer`, `split_fbank_batch_buffer`, `split_feature_batch_buffer`, `split_weights_batch_buffer`, `split_primary_feature_batch_buffer`, `split_primary_weights_batch_buffer` |
| `OrtEmbeddingState:84` | **`primary_batch_run_options: Option<RunOptions<…>>`** — S-T9a is right that this is not shareable; it must live in `Scratch` |

**Blast radius:** 10 `&mut self` methods across the model modules, 24 files referencing
`EmbeddingModel`/`SegmentationModel`. Mechanical but wide.

**Target API** (S-T9a's shape, which is still right even though its option 1 is not):

```rust
let shared = engine.clone_shared();          // cheap Arc handle, one weight copy
let mut scratch = Scratch::new(&shared);     // per request, tens of MB
engine.diarize(&shared, &mut scratch, audio, file_id)?;
```

Then `diar-server` holds `Arc<SharedModels>` and allocates a `Scratch` per request instead of
locking one engine.

**Sequencing suggestion:** do `SegmentationModel` first — 3 sessions, 4 scratch items, far smaller
— and prove the pattern end to end (including the gates) before touching `EmbeddingModel`'s 10
sessions and 12 buffers.

## 4. The dangerous failure mode

A partially-split model **shares a buffer between threads and returns silently wrong numbers**. It
does not crash, and a smoke test passes. This is why the gates below are non-negotiable and why
this was not started as a tail-end change: half-done here looks finished.

## 5. Gates — all must pass before this is "done"

| gate | bar | how |
|---|---|---|
| **Determinism** | 3 identical runs, byte-identical RTTM | same file ×3, diff the RTTMs |
| **DER parity, AMI-16** | full **13.10 %** ± 0.01, exclusive **17.81 %** | `validation/score_der.py`, refs in `refs/ami/`, UEM-cropped, collar 0.25 |
| **DER parity, Karpathy** | full **8.219 %**, exclusive **6.545 %** | same harness, `refs/karpathy/` |
| **Concurrency correctness** | N=4 concurrent jobs produce outputs **identical to serial** | run 4 files serially, then concurrently, diff every RTTM |
| **VRAM** | ≈ one engine + N × scratch, *not* N × engine | `nvidia-smi --query-compute-apps` during the N=4 run |
| **Throughput** | **≥ 2× serial** on 4 short files | wall-clock, quiet machine |
| **speakrs test suite** | 94 tests pass incl. Python-parity fixtures | see §6 |

The concurrency-correctness gate is the one that catches a botched split. Do not skip it because
DER looks fine — DER is an aggregate and will hide a handful of corrupted chunks.

## 6. Build and test traps (all cost real time already)

- **`ort` is pinned `=2.0.0-rc.12`.** rc.13 ships a static core that mismatches the 1.24.2 provider
  libs and fails at session load. Do not bump it (§4.26).
- **Tests:** `cargo test --release --no-default-features --features openblas-system,online`.
  Plain `--features openblas-system` fails — `default` already pulls a BLAS and you get
  "depends on ndarray-linalg multiple times". Dropping `online` breaks 6 doctests that need
  `from_pretrained`.
- **`RUST_MIN_STACK=16777216`** for the test harness (2 MiB default overflows).
- **Build in the container**, not on the host: `target/` is root-owned from container writes.
  `docker run --rm -v /mnt/nvm/repos/diar-native:/build -v /tmp/spk_target:/tmp/target -w /build
  -e CARGO_TARGET_DIR=/tmp/target diar-bench-builder:latest cargo test ...`
- **After any vendored change**, regenerate `patches/0001-cuda-performance-patch-set.patch`
  (`cd vendor/speakrs && git diff > ../../patches/0001-...patch`) — T10 ships from it.
- **Benchmark hygiene:** one timed leg at a time; `validation/run_e2e_baseline.sh` takes an
  `flock` for this reason (§7.13). Sample VRAM *during* a run, never after (§7.14).

## 7. Definition of done

1. All gates in §5 pass, numbers appended to `validation/RESULTS.md` as a numbered section.
2. `docs/DETAILED_SPECS.md` S-T9a corrected — option 1 does not work, and why.
3. Patch set regenerated; `PLAN.md` M1 remainder updated.
4. Re-run the two-concurrent-file measurement from §1 and show the spans no longer double.
