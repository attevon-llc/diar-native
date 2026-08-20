# diar-native — agent guide

Native (Rust/ONNX) speaker diarization for OpenTranscribe: a vendored, patched `speakrs`
wrapped by our crates and served as the `diar-native` sidecar. **Shipped**: the live stack
runs `diar-server:0.2.0`. Read `PLAN.md` for roadmap/decisions and `validation/RESULTS.md`
(append-only; **never re-run a logged test** — run only to compare a change against it).

## Layout

- `vendor/speakrs/` — upstream clone pinned at `b0756b1` + our patches as the WORKING TREE
  diff. After ANY vendored edit: `cd vendor/speakrs && git diff > ../../patches/0001-cuda-performance-patch-set.patch`.
  Never commit inside the vendored repo.
- `crates/diar-core` — engine wrapper: `DiarEngine::clone_shared()` per-request handles,
  centroids, `embed_window`, exclusive segments, gender, `audio.rs` media decode.
- `crates/diar-server` — the sidecar (axum): `/diarize` `/embed_window` `/healthz`;
  `DIAR_MAX_INFLIGHT` bounds concurrency; requests run on cloned handles (no engine mutex).
- `crates/diar-cli` — bench runner; `RUST_LOG=speakrs=trace` for engine stage timings.
- `upstream-work/` (gitignored) — upstream-tip clone with the 7 prepared PR branches;
  drafts in `docs/pr_drafts.md`. NO pushes without operator approval.
- `models_folded/` = fast model set (default), `models_small/` = laptop set — gitignored,
  gated community-1 derivatives, never commit or attach to public PRs.

## Build & test (traps that cost real time — RESULTS §4.26)

- `ort` is PINNED `=2.0.0-rc.12`; rc.13 fails at session load. Do not bump.
- Build in the container, never on the host (host lacks openblas; `target/` and
  `Cargo.lock` end up root-owned from container writes — chown via a container if hit):
  `docker run --rm -v $PWD:/build -v /tmp/diar_target:/tmp/target -w /build -e CARGO_TARGET_DIR=/tmp/target diar-bench-builder:latest cargo build --release --features cuda -p diar-server -p diar-cli`
- speakrs tests: same container, `-w /build/vendor/speakrs`, `RUST_MIN_STACK=16777216`,
  `cargo test --release --no-default-features --features openblas-system,online`
  (94 tests; plain `--features openblas-system` fails — duplicate BLAS).
  Fixture models live only in `vendor/speakrs/fixtures/models/` — mount them into any clone.
- Image: `docker build -f docker/Dockerfile.server -t diar-server:<ver> .`
- Host `cargo check` works for fast iteration (CARGO_TARGET_DIR to a /tmp dir).

## Benchmarks & gates (docs/BENCHMARK_PROTOCOL.md is law)

- One timed leg at a time, quiet machine (check `uptime` + `docker stats` — the dsva stack
  and sibling sessions routinely load this box). Sample VRAM DURING a run, never after.
- Every speed claim ships with its accuracy check. For pure-perf changes prove
  OUTPUT IDENTITY by diffing raw records, never assert it.
- Numbers to beat + corpus paths: `docs/TEST_CORPORA_AND_BASELINES.md`. DER scoring runs
  inside `opentranscribe-celery-worker` (it has pyannote.metrics): stage `validation/score_der.py`
  + refs via `docker cp`.
- Concurrency gate harness: `validation/t9a_concurrency.sh`.

## Deployment

- Live sidecar = compose service `diar-native` in transcribe-app (5 compose files; see
  `docker-compose.diar-native.yml`). Old image kept as `diar-server:pre-t9a` for rollback.
- Env knobs: `DIAR_MAX_INFLIGHT`, `SPEAKRS_FBANK_POOL`, `SPEAKRS_LAZY_SESSIONS`,
  `SPEAKRS_ARENA_SHRINK` (4 GB-tier VRAM floor, ~20% per-job cost — default off),
  `DIAR_GENDER_MAX_SECONDS`.
- Decisions on record: TensorRT rolled back (§7.26); native fbank superseded by the
  fbank∥GPU pipeline (§7.28); sinc resampler rejected — keep `FftFixedIn` (§7.29).

## Ground rules

- Gated model artifacts: regenerate locally, never commit/redistribute.
- `pyannote-audio-fork` is read-only. transcribe-app changes go through its own branch/PR flow.
- Update `PLAN.md` status markers and append to `RESULTS.md` (with controls) when work lands;
  retract numbers explicitly, never edit them silently.
