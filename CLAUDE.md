# diar-native — agent guide

Native (Rust/ONNX) speaker diarization for OpenTranscribe: a vendored, patched `speakrs`
wrapped by our crates and served as the `diar-native` sidecar. **Shipped**: the live stack
runs `diar-server:0.2.0`. Read `PLAN.md` for roadmap/decisions and `validation/RESULTS.md`
(append-only; **never re-run a logged test** — run only to compare a change against it).

## Layout

- `vendor/speakrs/` — upstream clone pinned at `b0756b1` + our patches as the WORKING TREE
  diff. After ANY vendored edit: `cd vendor/speakrs && git diff HEAD > ../../patches/0001-cuda-performance-patch-set.patch`
  (use `git diff HEAD`, not bare `git diff` — staged changes are otherwise silently dropped).
  Never commit inside the vendored repo (`vendor/` is fully gitignored; not a submodule).
  Reproduce elsewhere: `scripts/bootstrap_vendor_speakrs.sh` clones our fork
  (`attevon-llc/speakrs`, Apache-2.0 unchanged) pinned at the commit on `master` matching
  live `diar-server:0.2.0` (verified byte-identical to the vendored tree; 94/94 speakrs tests
  pass from a clean clone — 2026-08-20). `avencera/speakrs` remains the canonical upstream for
  PRs; fork's `master` is now the mergeable "what we run" branch, kept separate from the 7
  PR-prep branches (which are trimmed subsets tailored for clean upstream review, not meant to
  represent production).
- `crates/diar-core` — engine wrapper: `DiarEngine::clone_shared()` per-request handles,
  centroids, `embed_window`, exclusive segments, gender, `audio.rs` media decode.
  `clone_shared` is `#[cfg(not(feature = "coreml"))]` — speakrs cfgs its own equivalent out
  for CoreML (not ORT sessions, single-thread-at-a-time). `Mode` also has `CoreMl`/`CoreMlFast`.
- `crates/diar-server` — the sidecar (axum): `/diarize` `/embed_window` `/healthz`;
  `DIAR_MAX_INFLIGHT` bounds concurrency; requests run on cloned handles (no engine mutex) —
  except under `coreml`, where `AppState::with_engine` holds the mutex for the whole request
  instead (RESULTS §7.31; `DIAR_MAX_INFLIGHT` has no effect in that mode).
- `crates/diar-cli` — bench runner; `RUST_LOG=speakrs=trace` for engine stage timings.
- `upstream-work/` (gitignored) — upstream-tip clone with the 7 prepared PR branches;
  drafts in `docs/pr_drafts.md`. `origin` = `attevon-llc/speakrs` (our fork, branches pushed
  2026-08-20), `upstream` = `avencera/speakrs`. Opening PRs/issues against avencera/speakrs
  still needs explicit operator approval — nothing has been opened there.
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
- Image: `docker build -f docker/Dockerfile.server -t diar-server:<ver> .` (CUDA);
  `docker/Dockerfile.server-cpu` for the multi-arch CPU-only variant (linux/amd64+arm64).
- Host `cargo check` works for fast iteration (CARGO_TARGET_DIR to a /tmp dir).
- `coreml` feature (Apple Silicon, native GPU accel via Metal/CoreML — NOT reachable through
  Docker, which has no Metal access on macOS regardless of image arch) builds only on macOS
  (`objc2`/`objc2-core-ml` deps). Dev machine: an Apple Silicon Mac on the operator's private
  network, already has cargo/uv/Homebrew/openblas set up; `~/repos/diar-native` there is
  a manual `git archive | ssh ... tar -x` snapshot, not a git clone — re-sync by hand, no
  remote configured. `.mlmodelc` model artifacts are gated (same policy as ONNX), local-only
  on that machine at `vendor/speakrs/fixtures/models/`, produced via
  `scripts/native_coreml/convert_coreml.py` + `export_b64_seg.py` + `export_fbank_30s.py` (all
  three needed — RESULTS §7.31 has the gaps found in the first two).

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
  `DIAR_GENDER_MAX_SECONDS`, `DIAR_DEVICES` (comma list, first = default; wins over
  `DIAR_MODE`), `DIAR_MAX_INFLIGHT_CPU` (optional inner sub-gate, default off).
- The CUDA image is a SUPERSET of the CPU image on amd64 — one process serves `cuda` and
  `cpu`, picked per request via the `device` field (RESULTS §7.34). The ORT CPU EP is
  statically linked in every build, so this costs no extra bytes. `Dockerfile.server-cpu`
  stays as the arm64 / 189 MB artifact. `/healthz` returns JSON (`devices` = loaded,
  `supported_devices` = compiled in). **Old servers silently IGNORE `device`** (no
  `deny_unknown_fields`) — consumers must gate on `/healthz` `supported_devices`.
- Decisions on record: TensorRT rolled back (§7.26); native fbank superseded by the
  fbank∥GPU pipeline (§7.28); sinc resampler rejected — keep `FftFixedIn` (§7.29).

## Ground rules

- Gated model artifacts: regenerate locally, never commit/redistribute.
- `pyannote-audio-fork` is read-only. transcribe-app changes go through its own branch/PR flow.
- Update `PLAN.md` status markers and append to `RESULTS.md` (with controls) when work lands;
  retract numbers explicitly, never edit them silently.
