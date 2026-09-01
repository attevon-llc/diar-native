# diar-native — agent guide

Native (Rust/ONNX) speaker diarization for OpenTranscribe: a vendored, patched `speakrs`
wrapped by our crates and served as the `diar-native` sidecar. **Released: 0.3.0** (crates and
image tag unified at that number; see `CHANGELOG.md`).

**What actually runs in production is the BINARY, not this image.** The live sidecar is compose
service `diar-native` running the *shared backend* image
(`davidamacey/opentranscribe-backend:...`) with `command: ["diar-server"]`; the binary and three
ORT `.so`s get there via a build stage in `transcribe-app/backend/Dockerfile.prod` pinned to
`davidamacey/diar-native@sha256:...`. So a release here ships to production only when that digest
is repointed — which is a change made in transcribe-app, not here. As of 0.3.0 the pinned digest
is still a 0.2.0 build.

Read `PLAN.md` for roadmap/decisions and `validation/RESULTS.md`
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
  centroids, `embed_window`, exclusive segments, gender, `audio.rs` media decode, and
  `logging.rs` (the `RUST_LOG`/`DIAR_LOG_FORMAT` policy both binaries share; the sink is a
  parameter, and the library never installs a subscriber itself), `ort_compat.rs` (per-platform
  session workarounds — currently the aarch64 fp16 GELU cap), and `provision/` (preflight,
  exporter, marker, `files.rs` = the authoritative required-file lists, `verify.rs` = the
  five-stage smoke test). Provisioning lives in diar-core, NOT diar-server: diar-server is a
  binary crate with no `tests/`, so nothing in it is integration-testable.
  `clone_shared` is `#[cfg(not(feature = "coreml"))]` — speakrs cfgs its own equivalent out
  for CoreML (not ORT sessions, single-thread-at-a-time). `Mode` also has `CoreMl`/`CoreMlFast`.
- `crates/diar-server` — the sidecar (axum): `/diarize` `/embed_window` `/healthz` `/readyz`
  (four routes). `/healthz` returns JSON and is **always 200 while serving, in every model
  state** — a hard compatibility constraint, since the compose healthcheck and
  `diarizer_native.py` both gate on the status alone; `/readyz` is the one allowed to 503.
  `cli.rs` also carries the `provision-models` / `verify-models` / `check-token` subcommands
  (no subcommand = serve, which the deployment relies on).
  `DIAR_MAX_INFLIGHT` bounds concurrency; requests run on cloned handles (no engine mutex) —
  except under `coreml`, where `AppState::with_engine` holds the mutex for the whole request
  instead (RESULTS §7.31; `DIAR_MAX_INFLIGHT` has no effect in that mode).
- `crates/diar-cli` — bench runner; `RUST_LOG=speakrs=trace` for engine stage timings (on
  stderr — stdout is the harness's JSONL). **This works in `diar-server` too as of §7.37**;
  before that the server installed no subscriber at all, so `RUST_LOG` was dead in the
  DEPLOYED artifact and only ever did anything in the CLI.
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
  **`diar-bench-builder:latest` is a machine-local image; no Dockerfile in this repo produces
  it.** On any other machine, build the equivalent from `ubuntu:24.04` + apt `build-essential
  pkg-config cmake ca-certificates curl git libssl-dev libclang-dev libopenblas-dev` + rustup
  stable. That is exactly what CI installs — `.github/actions/setup-build-env` is the single
  definition, and `CONTRIBUTING.md` documents reproducing it locally. openblas is needed because
  speakrs is built with `openblas-system`; libclang is for bindgen (`ort-sys`).
- speakrs tests: same container, `-w /build/vendor/speakrs`, `RUST_MIN_STACK=16777216`,
  `cargo test --release --no-default-features --features openblas-system,online`
  (94 tests; plain `--features openblas-system` fails — duplicate BLAS).
  Fixture models live only in `vendor/speakrs/fixtures/models/` — mount them into any clone.
- Image: `docker build -f docker/Dockerfile.server -t diar-server:<ver> .` (CUDA);
  `docker/Dockerfile.server-cpu` for the multi-arch CPU-only variant (linux/amd64+arm64).
- Host `cargo check` works for fast iteration (CARGO_TARGET_DIR to a /tmp dir).
- `coreml` feature (Apple Silicon, native GPU accel via Metal/CoreML — NOT reachable through
  Docker, which has no Metal access on macOS regardless of image arch) builds only on macOS
  (`objc2`/`objc2-core-ml` deps). Dev machine: an Apple Silicon Mac (M2 Max, macOS 15.7.9) on
  the operator's private network, with cargo/uv/Homebrew/openblas set up. `~/repos/diar-native`
  there is now a REAL CLONE with `origin` set to `attevon-llc/diar-native` (`git pull` works);
  the old `git archive | tar -x` snapshot is parked at `~/repos/diar-native-OLD-snapshot` and
  is the only copy holding the gated model artifacts — do not delete it without moving those.
  Native builds there need `LIBRARY_PATH=/opt/homebrew/opt/openblas/lib` for BOTH the default
  and the `coreml` build; both compile clean (RESULTS §7.40).
  `.mlmodelc` model artifacts are gated (same policy as ONNX), local-only
  on that machine at `vendor/speakrs/fixtures/models/`, produced via
  `scripts/native_coreml/convert_coreml.py` + `export_b64_seg.py` + `export_fbank_30s.py` (all
  three needed — RESULTS §7.31 has the gaps found in the first two).
- **ORT rewrites graphs AT LOAD and the aarch64 builds disagree about it** — read
  `docs/ORT_FUSION_FP16_AARCH64.md` before touching precision, session options, or anything
  fp16. Short version: ORT fuses `Erf`-GELU into `com.microsoft.Gelu` during session load, and
  on **linux/arm64** it then has no fp16 kernel for the node it just made, so
  `gender-wav2vec2.onnx` (fp16) FAILS TO LOAD there — silently disabling gender, HTTP 200 and
  all. macOS arm64 is fine: same missing kernel, but its ORT declines to fuse fp16 at all.
  11 of 15 diarization graphs get the same treatment (`com.microsoft::FusedConv`, fp32-only
  kernel) and are safe ONLY because they are fp32 — so **any future fp16 export needs a LOAD
  gate on aarch64, not just an accuracy gate** (an accuracy gate cannot see this; the session
  never opens). Traps if you go near the fix: the optimizer is `GeluFusionL2` NOT `GeluFusion`,
  an unrecognized optimizer name is SILENTLY IGNORED, and the separator is `;` not `,`.
- Reproduce the FAILING linux/arm64 platform from the Mac — Docker Desktop runs arm64 natively.
  Base image must be `rust:1-trixie` (this ORT needs glibc >= 2.38; bookworm's 2.36 fails at
  link) with `RUSTFLAGS="-C link-arg=-lstdc++"`. One command for both platforms:
  `validation/ort_fusion_probe/run_probe.sh <models-dir>` (see its README; it is NOT a
  workspace member, so a root `cargo build` never touches it).

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
- **Env knobs: `README.md` §6e is the ONE authoritative table** (name, effect, default, which
  subcommand reads it) and is checked in both directions — anything not in it has no read site.
  Do not maintain a second list here; the highlights only:
  - serve: `DIAR_MODELS_DIR` (`/models`), `DIAR_BIND` (`0.0.0.0:8701`), `DIAR_MAX_INFLIGHT` (2),
    `DIAR_MAX_INFLIGHT_CPU` (off), `DIAR_DEVICES` (comma list, first = default; wins over
    `DIAR_MODE`), `DIAR_MODE` (unset *or unrecognized* ⇒ `cuda`), `DIAR_MODEL_SET`,
    `DIAR_ALLOW_UNVERIFIED_MODELS`, `DIAR_GENDER_MAX_SECONDS` (5),
    `RUST_LOG` (default `info,ort::logging=warn` — NOT unset-means-silent; ORT's native bridge
    is held at warn because it emits 5812 INFO lines per CUDA startup), `DIAR_LOG_FORMAT`
    (`text`|`json`).
  - provisioning: `HF_TOKEN` (+ `HUGGINGFACE_TOKEN`, `HUGGING_FACE_HUB_TOKEN`), `HF_ENDPOINT`
    (the only knob that makes provisioning work against an HF mirror / air-gapped proxy),
    `HF_HOME`, `DIAR_EXPORT_PYTHON`. Device defaults to **cpu** here, not cuda.
  - `DIAR_ORT_OPT_LEVEL` / `DIAR_ORT_DISABLED_OPTIMIZERS` cover only sessions built through
    `diar_core::ort_compat` — the gender model and the smoke test, NOT speakrs' 15 diarization
    graphs.
  - engine: `SPEAKRS_LAZY_SESSIONS`, `SPEAKRS_ARENA_SHRINK` (4 GB-tier VRAM floor, ~20% per-job
    cost — default off), `SPEAKRS_INTRA_THREADS`, `SPEAKRS_FBANK_THREADS`, `SPEAKRS_AHC_THREADS`.
  - **`SPEAKRS_FBANK_POOL` is NOT operator-settable through `diar-server`** — `DiarEngine::load`
    unconditionally `set_var`s it before speakrs reads it back in the same call. (That `set_var`
    is also why engine loads must stay serial in `run()` before the server binds: glibc `setenv`
    is not thread-safe.) `SPEAKRS_TRT`/`SPEAKRS_TRT_CACHE` are dead — no read sites.
- Exit codes are a stable contract (`crates/diar-core/src/provision/mod.rs::exit`), tabulated in
  README §6d and `docs/INSTALL_NATIVE.md`. Note **8** = serve-time "models unusable" (was 6, which
  now means only "no usable python export env"); 9 = device unavailable, no marker written;
  10 = `verify-models` found nothing to verify against.
- Server logs go to **stdout** (so `docker logs`/compose capture them); fatal startup errors
  stay on stderr. Only the serve path installs a subscriber — the `provision-models` /
  `verify-models` subcommands write machine-readable JSON to stdout and must not be
  interleaved with log records.
- The CUDA image is a SUPERSET of the CPU image on amd64 — one process serves `cuda` and
  `cpu`, picked per request via the `device` field (RESULTS §7.34). The ORT CPU EP is
  statically linked in every build, so this costs no extra bytes. `Dockerfile.server-cpu`
  stays as the arm64 / 194 MB artifact. `/healthz` returns JSON (`devices` = loaded,
  `supported_devices` = compiled in). **Old servers silently IGNORE `device`** (no
  `deny_unknown_fields`) — consumers must gate on `/healthz` `supported_devices`.
- Decisions on record: TensorRT rolled back (§7.26); native fbank superseded by the
  fbank∥GPU pipeline (§7.28); sinc resampler rejected — keep `FftFixedIn` (§7.29); fp16 gender
  on linux/arm64 root-caused to an ORT load-time fusion, fix chosen = cap the GENDER session
  at `GraphOptimizationLevel::Level1` (bitwise identical to unoptimized; `GeluFusionL2` is the
  validated alternative) — §7.40, NOT YET IMPLEMENTED.

## Ground rules

- Gated model artifacts: regenerate locally, never commit/redistribute.
- `pyannote-audio-fork` is read-only. transcribe-app changes go through its own branch/PR flow.
- Update `PLAN.md` status markers and append to `RESULTS.md` (with controls) when work lands;
  retract numbers explicitly, never edit them silently.
