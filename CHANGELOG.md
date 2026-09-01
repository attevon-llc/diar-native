# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**A note on versions.** This project ships as a container image, and its version history lived in
image tags rather than in git — there are **no git tags** before 0.3.0. `0.2.0` below is the tag
of the image that ran live in the OpenTranscribe stack (`diar-server:0.2.0`, published
2026-08-20), reconstructed from the commit history; the workspace crates were still at `0.1.0`
while that image said `0.2.0`. **As of 0.3.0 the three crates and the image tag are one number**,
and it is the number `diar-server` reports in its startup record and stamps into the provisioning
marker.

Measurements referenced here are recorded in
[`validation/RESULTS.md`](validation/RESULTS.md), which is append-only.

---

## [Unreleased]

Nothing yet.

---

## [0.3.0] — 2026-09-01

**Self-hosting, and one image for every device.** Two things stood between this engine and
anyone outside the OpenTranscribe stack running it: the weights could not be obtained (they are
derivatives of gated pyannote weights, and the export recipe was undocumented — even to us), and
the GPU image could not be asked to run on a CPU. Both are closed. Along the way the sidecar
learned to speak: it had shipped with no log subscriber at all.

### Breaking / upgrade notes

Read this before upgrading. Nothing here breaks a *default* deployment, but each item below
needs a consumer-side decision.

- **`/healthz` body shape changed** — it was the bare string `ok`, it is now a JSON object. The
  **status code is unchanged and guaranteed**: `/healthz` returns **200 in every model state**,
  including unprovisioned and known-bad. That is a deliberate compatibility promise, not an
  oversight — `docker-compose.diar-native.yml` runs `curl -sf .../healthz` and
  `diarizer_native.py` checks `resp.status == 200`, and every models directory deployed today has
  no marker, so a 503 for "unverified" would fail every existing healthcheck on the day it
  shipped and silently fall OpenTranscribe back to in-process PyAnnote. Use **`/readyz`** if you
  want a readiness signal that can fail. Anything parsing the *body* of `/healthz` as a string
  must be updated.
- **Startup model-gate exit code moved 6 → 8.** "The models directory is too broken to serve" is
  now `8` (`MODELS_UNUSABLE`); `6` now means only "no usable python export environment". The two
  used to share a code, and a supervisor could not tell "install torch into the exporter" from
  "provision the models" — which have nothing to do with each other, since serving needs no
  python at all. Exit codes `9` and `10` are also new. Full table in the README.
- **An OLD diar-server silently ignores the new `device` field.** Neither request struct uses
  `deny_unknown_fields`, so `{"device": "cpu"}` sent to a 0.2.0 server is *ignored* and the job
  runs on CUDA, returning 200. Serde cannot help here. Consumers MUST negotiate on `/healthz`
  `supported_devices` (or on the presence of the `x-diar-device` response header) before relying
  on the field. On a 0.3.0 server an unknown device name is a 400 naming the devices the build
  serves.
- **Consumers must bump their pinned digest or none of this ships.** OpenTranscribe consumes the
  **binary**, not this image: `transcribe-app/backend/Dockerfile.prod` pins
  `davidamacey/diar-native@sha256:…` purely to `COPY --from` the `diar-server` binary and three
  ORT `.so`s into the shared backend image, which the `diar-native` compose service then runs
  with `command: ["diar-server"]`. Until that digest is repointed at 0.3.0, the live sidecar is
  still 0.2.0 regardless of what is published here.
- **`DIAR_ORT_OPT_LEVEL` is now a floor, not an override.** The effective level is
  `min(requested, per-model cap)`, so the variable can still *lower* any session's ONNX Runtime
  optimization level but can no longer *raise* one past a cap the code applies. Exactly one cap
  exists today: the gender model on **aarch64**, held at `Level1` by the issue #14 fix below. So
  on linux/arm64, `DIAR_ORT_OPT_LEVEL=all|extended` no longer reaches the gender session — it
  stays at `Level1`. That is a real behaviour change to a shipped knob, and it is the point of
  the change rather than a side effect: the old override *silently un-did the workaround*, so an
  operator raising the level to tune the 15 diarization graphs lost speaker gender with no error
  anywhere, on arm64 only, with `/healthz` still 200. **On x86_64 nothing changes** — there is no
  cap there, and the variable behaves exactly as before at every value. To explore above a cap,
  use `validation/ort_fusion_probe`, which reports load success per configuration instead of
  half-starting a server.
  Two adjacent silent no-ops in the same pair of knobs are now hard errors at startup, which can
  turn a container that previously came up (with the setting quietly ignored) into one that
  refuses to start until the value is corrected: an unrecognized `DIAR_ORT_OPT_LEVEL` value, and
  any `DIAR_ORT_DISABLED_OPTIMIZERS` value containing `,` — ORT's separator is `;`, and it reads
  `A,B` as one optimizer name that matches nothing and disables nothing.
- **Directories provisioned by export recipe 1 now report `stale`.** `EXPORT_RECIPE_VERSION` is
  `2`. Stale is **non-fatal** — they still serve, and `/healthz` stays 200 — but they fail
  `/readyz`, log one warning, and carry the 378.5 MB fp32 gender model instead of the 189.5 MB
  fp16 one. `provision-models --force` brings them current.

### Added

- **Running diar-native no longer requires this repository.** The published images existed but
  nothing shipped here used them: `docker-compose.yml` defaulted to a local tag and `start.sh`
  *built*, so the fastest documented route to a working sidecar was a 20-minute Rust
  compilation in place of a 195 MB pull. The deployment is now two files and no checkout.
  - **`docker-compose.prod.yml`** — pulls published images, with no `build:` key anywhere and
    no reference to any path inside the source tree. A single `docker compose up` provisions
    *and then* serves: the export runs as a service that must exit 0, and the sidecar waits on
    it with `depends_on: { condition: service_completed_successfully }`, so there is no
    two-command dance and no profile to remember. Provisioning is idempotent, so every
    subsequent `up` costs about a second and needs neither token nor network. The healthcheck
    is on `/healthz`, never `/readyz` — an unprovisioned container must not be marked unhealthy
    — and the GPU stays opt-in through the existing `docker-compose.gpu.yml` overlay so the
    base file runs unchanged on a GPU-less host. Pinned by tag, with the three release digests
    in a comment for anyone who wants to pin harder.
  - Models and the HuggingFace cache default to **named volumes** rather than bind mounts,
    because compose creates a missing bind-mount source as **root** (measured): a first run in
    an empty directory would otherwise hand the non-root export a directory it cannot write,
    fail with exit 7 on a path the operator never touched, and leave files needing `sudo` to
    remove. A host directory still works — set `DIAR_MODELS_HOST_DIR` together with
    `DIAR_UID`/`DIAR_GID`, which is exactly what `start.sh` does.
  - **`docker/Dockerfile.provision` now creates `/models` owned by uid 10001.** Docker seeds a
    fresh named volume's ownership from the image directory it covers, and creates it
    `root:root` when that directory does not exist — so without this line the zero-configuration
    volume above is not writable by the very image that has to write it.
- **`start.sh` pulls by default; `--build` is the contributor path.** It selects the image for
  the machine it is on — amd64 + GPU → `:0.3.0`, amd64 → `:0.3.0-cpu`, arm64 →
  `:0.3.0-cpu-arm64` — and the release version appears in exactly one place in the script.
  Because every published tag is single-platform and `:latest` is the amd64 CUDA image, the
  script also **verifies the architecture of every image it is about to run** and refuses a
  mismatch with an actionable message: Docker Desktop emulates a wrong-architecture image
  rather than rejecting it, which turns "you pulled the wrong tag" into "diarization is
  inexplicably slow". `--cli` still builds, because `diar-cli` is the one binary not published.
- **A "Run it / Use it / Develop it" front door on `README.md`**, above the badges, with literal
  commands for the three host types rather than a pointer to another file. The Apple Silicon
  case states plainly that Docker uses **CPU cores and neither the GPU nor the Neural Engine** —
  an `arm64` tag invites the opposite assumption — and mentions the native `coreml` build as
  something that exists, works (§7.31) and is *not* published. `QUICKSTART.md` leads with the
  same no-clone path and keeps the build path as the contributor route.
- **A standalone one-command quickstart.** The repository could be built and deployed but not
  *started* by anyone who had not read `docs/INSTALL_NATIVE.md` end to end: there was no
  `.env.example`, no compose file, no start script and no quickstart, so a newcomer had to
  hand-assemble every command. Now:
  - **`start.sh`** — creates `.env`, prompts for the HuggingFace token (hidden input, stored
    chmod 600, never placed on a command line where `ps` would expose it), runs `check-token`
    and surfaces its own actionable message instead of a traceback, provisions only when the
    marker is absent, starts the sidecar, waits on `/readyz`, and prints a paste-ready
    `curl`. Re-running is a fast no-op that needs neither token nor network. `--cli` diarizes
    a single file with no server at all; `--cpu`, `--gpu`, `--provision`, `--rebuild`,
    `--stop`, `--logs` and `--help` are also there.
  - **`docker-compose.yml`** — a `diar-native` service mounting models `:ro`, plus a
    `provision` service behind `profiles: [provision]` mounting them read-write, so the export
    never runs on a plain `up`. The healthcheck deliberately targets `/healthz`, not
    `/readyz`: an unprovisioned container must not be marked unhealthy and fail `up --wait`.
  - **`docker-compose.gpu.yml`** — GPU is opt-in via an overlay, because a device reservation
    in the base file is a hard startup failure on every machine without one. `start.sh` adds
    it only when `nvidia-smi` **and** the NVIDIA container runtime are both present; checking
    only the former selects the CUDA image on hosts that cannot actually pass a GPU through.
  - **`.env.example`** and **`QUICKSTART.md`** (60-second path, both the server and the CLI,
    how to read the response, and the four failures operators actually hit).
- **A `cli` build target** in both server Dockerfiles, producing a `diar-cli` image that shares
  every layer with the serving image. `diar-cli` was previously unavailable in any published
  artifact. It is a separate target rather than a second binary in the sidecar because it
  statically links its own ~37 MB copy of ORT and speakrs; the serving image is unchanged at
  195 MB (CPU).

- **Model provisioning built into the `diar-server` binary**, closing the last blocker to
  self-hosted OpenTranscribe running the native diarizer. `provision-models` turns a Hugging Face
  token into a verified models directory; `verify-models` runs a five-stage smoke test against
  the same ONNX Runtime build the server uses; `check-token` reports in ~200 ms and two HTTPS
  calls whether the token is valid and the community-1 gate has been accepted. All three write
  machine-readable JSON to stdout under `--json`. (RESULTS §7.35, §7.36)
  - The export recipe turned out to be **5 steps, not 1**, and step 2b — onnxsim constant-folding
    the three segmentation graphs and writing them under the *plain* filenames — was mandatory
    and undocumented anywhere. Skipping it costs ~2× on segmentation and silently reintroduces
    the ORT-CUDA `Sin`/`Cos` CPU-fallback tax, with no error.
  - **A cold run reproduces the shipped set.** All 15 diarization ONNX graphs come back with
    identical op-type histograms and **bit-identical initializer tensors**; all six `plda_*.npy`
    and `min_num_samples.txt` byte-identical; and the diarization **RTTM sha256 is identical** to
    the one the shipped `models_folded/` produces. Measured at **119.5 s** for the cold export.
  - A provisioned recipe-2 directory is **~484 MB** (`fast` set, with gender). The §7.36
    acceptance run measured **673 MB** because it hit the fp32 gender fallback described below.
- Provisioning provenance recorded in `diar-provision.json` — export-recipe version, exporter
  version, upstream pipeline revision, toolchain versions, and every file's size and sha256, plus
  the smoke-test result. Checked `stat`-only at startup and by full sha256 in `verify-models`.
- **A startup model gate.** A missing or zero-length required model file is now fatal (exit 8)
  with a message naming the provisioning command and the Hugging Face gate URL, instead of
  surfacing as one "CUDA session load failed" per configured device inside a
  `restart: unless-stopped` crash loop. A missing *marker* is deliberately only a warning.
  `DIAR_ALLOW_UNVERIFIED_MODELS=1` downgrades the fatal cases.
- **`/readyz`**, returning 503 with an actionable reason until the models are verified, and 200
  after. This is where "still provisioning" is distinguished from "broken", with zero blast
  radius on existing `/healthz` callers.
- Model and device state on `/healthz`: loaded `devices` versus compiled-in `supported_devices`,
  plus flat `models_verified`, `models_state`, `models_dir`, `models_set`,
  `models_exporter_version`, `models_pipeline_revision`, `models_smoke_at`, `models_gender` and
  `models_reason` fields. Flat rather than nested so appending stayed additive.
- Serve **CUDA and CPU from a single image and process**, selected per request via a `device`
  field on `/diarize` and `/embed_window`, with the device that actually ran the job reported in
  an **`x-diar-device`** response header. On amd64 the CUDA image is a strict superset of the CPU
  image and always was — the ORT CPU execution provider is *statically linked* into every build
  by `ort-sys`, so `ldd diar-server` shows no ONNX Runtime `NEEDED` entry at all. Verified
  empirically against the already-shipped `davidamacey/diar-native:0.2.0` with `DIAR_MODE=cpu`,
  no `--gpus` and no `/dev/nvidia*`. **Image size unchanged at 3.46 GB**; the second engine costs
  **+620 MB host RSS and 0 MiB VRAM**. **Speaker centroids are bit-identical between devices**
  (max delta 0.0, on every clip tested). Segment *boundaries* can differ by up to one
  segmentation frame (0.016875 s) where a posterior sits on the binarisation threshold and
  lands on opposite sides under CPU vs CUDA float arithmetic — speaker count, segment count,
  exclusive count and gender verdicts are unaffected. An earlier draft of this entry said
  "output is bit-identical between devices"; that held on the 26 s smoke fixture it was
  measured on but does not generalise. (RESULTS §7.34, corrected by §7.49)
- `DIAR_DEVICES` (comma-separated; first entry is the default device, wins over `DIAR_MODE`) and
  an optional `DIAR_MAX_INFLIGHT_CPU` sub-gate beneath the global `DIAR_MAX_INFLIGHT`.
- **Structured logging.** `diar-server` now installs a `tracing` subscriber — it previously
  installed **none**, so `RUST_LOG` was inert in the deployed artifact and speakrs' 40 events and
  diar-core's 2 warnings were silently discarded; an operator saw two `eprintln!` lines and
  crashes. Records go to **stdout** (so `docker logs` and compose capture them) while fatal
  startup errors stay on stderr; `RUST_LOG` filters (default `info,ort::logging=warn`) and
  `DIAR_LOG_FORMAT=text|json` selects rendering. The provisioning subcommands install no
  subscriber, so their JSON is never interleaved with log records. (RESULTS §7.37)
- One log span per request carrying `request_id`, endpoint, device, audio **basename** and the
  gender flag, closed by a record with `duration_ms`, outcome, and either speaker/segment counts
  or an `error_class` (`bad_device`, `admission`, `invalid_input`, `audio_decode`, `inference`,
  `panic`) — plus one startup record derived from the `/healthz` body, so the two cannot drift.
- Propagation of an inbound **`x-request-id`** (sanitized before logging — control characters
  stripped, 64 chars max, so a caller cannot forge a log record), echoed on both success and
  failure responses so a single id spans caller and sidecar.
- **`audio_path` and `media_path` as aliases for `wav_path`** on `/diarize` and `/embed_window`.
  The field accepts any symphonia-decodable media; the old name was pushing third parties to
  transcode to WAV first for no reason. `wav_path` is still accepted and is what the live caller
  sends.
- A `HEALTHCHECK` on both serving images, deliberately targeting `/healthz` (liveness) rather
  than `/readyz` — pointing it at readiness would mark every not-yet-provisioned container
  unhealthy and fail `compose up --wait` for the whole stack.
- The smoke-test clip (832 KB) baked into both serving images so `verify-models` runs in-image,
  plus a fallback provisioning image (`docker/Dockerfile.provision`) carrying a pinned CPU-only
  torch environment for operators on the plain image.
- `scripts/bootstrap_vendor_speakrs.sh`, which reproduces the vendored, patched speakrs tree from
  the public `attevon-llc/speakrs` fork at a pinned commit — CI depends on it, since `vendor/` is
  gitignored and is not a submodule.
- CoreML support for Apple Silicon behind a `coreml` feature, wired through `diar-core`,
  `diar-server` and `diar-cli`. Not reachable through Docker — macOS grants containers no Metal
  access regardless of image architecture. (RESULTS §7.31)
- A CPU-only multi-architecture image variant (`linux/amd64` + `linux/arm64`).
- **A reproducible build environment.** `docker/Dockerfile.builder` builds the container the
  project is meant to be built in — previously the documented image was machine-local and no
  Dockerfile produced it, so a fresh clone could not reproduce the documented build at all. The
  apt package list now lives once in `scripts/build-deps.txt`, shared with
  `.github/actions/setup-build-env`; `rust-toolchain.toml` pins the compiler (1.97.1) for the
  container, CI and a bare workstation `cargo` alike; `scripts/setup_dev_env.sh` sets up a host;
  and a `dev-container-parity` CI job fails if the environments drift apart.
- `docs/ORT_FUSION_FP16_AARCH64.md`, explaining the fp16 load failure above, plus
  `validation/ort_fusion_probe/` — a standalone harness that dumps ORT's *optimized* graph so
  load-time fusions can be observed directly rather than inferred from an error message.
- CI (rustfmt, clippy, CPU build + tests, `.dockerignore` guard, CPU image build, ruff), a
  release workflow, Dependabot, `.pre-commit-config.yaml`, `CONTRIBUTING.md`, `SECURITY.md`,
  `CODEOWNERS`, `rustfmt.toml`, `clippy.toml`, `.editorconfig`, `.gitattributes`, a PR template,
  and this changelog. CI builds only the default (CPU) feature set, downloads no model weights,
  and requires no secrets.
- **Three more CI gates: supply chain, rustdoc and coverage.**
  - `cargo deny` (`deny.toml`) over advisories, licences, duplicate/banned crates and dependency
    sources. It matters here because the repo vendors Apache-2.0 speakrs, pins `ort` to a
    release candidate that must not move, and ships beside terms-gated weights — keeping the
    crate graph permissive-only keeps the licence question about the weights alone. Every
    exception is per-crate, dated and reasoned in the file; MPL-2.0 (symphonia + `option-ext`) is
    granted crate by crate rather than added to the allow list, and the six duplicate crates each
    name what would retire them. `unmaintained = "all"`, `yanked = "deny"`. Not also cargo-audit:
    it reads the same RustSec database and adds nothing this does not already cover.
  - `cargo doc --no-deps --workspace` with `RUSTDOCFLAGS=-D warnings`, so broken intra-doc links
    fail the build instead of silently rendering as plain text. `missing_docs` is deliberately
    left off.
  - Coverage via `cargo-llvm-cov`, **reporting only — no threshold**. It measures ~49% of lines,
    and the job prints why that is a floor rather than a verdict: the ten `#[ignore]`d
    integration tests that cover `provision/verify.rs`, `gender.rs` and `audio.rs` need
    terms-gated weights and can never run in CI.
- **A declared MSRV of 1.88.0** (`rust-version` in `[workspace.package]`), distinct from the
  1.97.1 build pin in `rust-toolchain.toml`. Measured, not assumed: 1.87.0 is refused by `ort`,
  `speakrs`, `time` and the `icu_*` crates, and 1.88.0 builds and tests green. CI still builds at
  1.97.1 only, which `CONTRIBUTING.md` says plainly.
- `publish = false` on all three crates. They cannot go to crates.io — `diar-core` depends on the
  vendored speakrs by path — and saying so stops `cargo publish` failing in a confusing way.
- Apache-2.0 `LICENSE`.

### Changed

- **Crate versions unified at 0.3.0.** All three were `0.1.0` while the image tag said `0.2.0`.
  The drift was not cosmetic: `env!("CARGO_PKG_VERSION")` feeds the startup log record and the
  provisioning marker's `generated_by`, so markers were being stamped with a version that
  matched no artifact anyone could name.
- **`EXPORT_RECIPE_VERSION` bumped to 2** for the fp16 gender export. Directories from recipe 1
  are reported `stale` — non-fatal, they still serve.
- **Gender model fp16 restored: 378.5 MB → 189.5 MB (−50.0%), and −252 MiB VRAM measured
  per-process on the 10-minute reference clip (−506 MiB was measured on a whole-container AMI
  run; the peak arena is workload dependent, so quote the basis with the number).** The
  root cause was two **no-op `Cast` nodes** that torch 2.13 emits and torch 2.11 did not, which
  made `onnxconverter_common.float16` produce a graph ORT rejected with "Type parameter (T) of
  Optype (Add) bound to different types". The exporter now elides them. The regenerated graph
  matches the shipped torch-2.11 artifact on every load-bearing property — 213/213 FLOAT16
  initializers, fp32 in and out, opset 17, same two boundary casts by role. (RESULTS §7.39)
- Cut the Docker **build context from 8.3 GB to 302 MB** with an allowlist `.dockerignore`. There
  was none before, so every build was also shipping ~1.1 GB of gated model weights into the
  context.
- The smoke test now **numerically verifies the graphs production actually runs**. It previously
  checked the b64 family; live compose sets `SPEAKRS_LAZY_SESSIONS=1`, under which speakrs gates
  the batched sessions but **not** the multimask ones — so a `wespeaker-multimask-tail-b32.onnx`
  with its largest initializer zeroed passed all five stages green, earned a `verified` marker
  and a 200 from `/readyz`. New stage 3e catches it at 9.222e-1 against a 1e-4 bar. (RESULTS
  §7.38)
- Compare model sets by content multiset of `(dtype, shape, sha256)` rather than by initializer
  name. `torch.onnx.export(dynamo=True)` assigns those names at trace time, so the comparison
  paired different tensors and reported "13/15 initializers differ, max |Δ| 2.000e+00" for graphs
  that were in fact bit-identical. The false positive was plausible enough that a tolerance would
  have been "fixed" instead of the tool.
- `verify-models` **re-attests by default**: on a fully verified directory it refreshes only the
  marker's `smoke` record, leaving all provenance untouched. That is the recovery path for a
  directory carrying a stale `fail` record without a full re-export. `--no-attest` opts out, and
  a read-only mount is not an error.
- Scale embedding-session intra-op thread count with core count in CPU mode. (RESULTS §7.32)
- `DIAR_DEVICES` takes precedence over `DIAR_MODE` when both are set. With neither set, behaviour
  is unchanged — including the long-standing "unset or unrecognized means `cuda`" default.
- CPU image grew 189 MB → 194 MB (+2.6%) from the provisioning code, `clap`, `ureq`/rustls,
  `sha2`/`time` and the baked smoke clip. The alternative it replaces — bundling torch and
  pyannote into the runtime image so provisioning could run there — measured ~13× on this image,
  for a step that runs once.
- `nvidia/cuda` pinned to the 12.8.x line in Dependabot, so a base-image bump cannot silently
  move the CUDA minor the provider libraries were built against.

### Fixed

- **Provisioning no longer defaults to CUDA.** It defaulted to a *device*, so on a GPU-less host
  the smoke test failed for want of a GPU and provisioning wrote a marker declaring
  known-**good** models known-**bad** — after which the startup gate refused to serve them,
  permanently, with no path back that did not involve deleting the marker by hand. Provisioning
  now defaults to CPU, and a genuinely unavailable device exits 9 (`DEVICE_UNAVAILABLE`) writing
  **no marker at all**, because "I could not test this" must never be recorded as "this is
  broken".
- **`verify-models` no longer reports success on a directory with no marker.** It hashed zero
  bytes, compared nothing, and exited 0 — a clean bill of health from the one command whose job
  is detecting a silent rewrite. It now exits 10 (`UNVERIFIABLE`) and says so in as many words.
  A marker with an empty `files` array is likewise treated as drift.
- **Fixed the ORT log flood: 5835 → 38 lines** on a CUDA startup. ONNX Runtime's native bridge
  emitted 5812 `ort::logging` INFO lines ("Removing NodeArg…", "GraphTransformer… modified: 0")
  against 3 from diar-server, burying the startup record ~2000:1. The default filter holds that
  one target at `warn` while keeping its genuine perf diagnostics and `ort::ep`'s
  EP-registration lines; `RUST_LOG=ort=info` brings the firehose back.
- Fixed a **privacy leak in logged error text**. Keeping the span *field* to a basename was not
  sufficient: the underlying I/O errors interpolate the path they were handed, so a failed decode
  logged the full media path through the `error` field while the `audio` field dutifully said
  `smoke.wav`. Error text is now redacted before it is logged. The HTTP *response* still carries
  the full path — the caller supplied it, so it is not a disclosure to them.
- **fp16 gender would not load at all on linux/arm64** (issue #14), silently disabling speaker
  gender on that platform while still answering 200. The graph is plain opset-17 `ai.onnx` with
  no contrib domain, but it has 20 `Erf` nodes, and one of ORT's *extended* (level-2)
  optimizations rewrites that GELU pattern into `com.microsoft.Gelu`, for which there is no fp16
  kernel — so the optimizer synthesized a node the very same runtime then refused to execute.
  The node named in the error does not exist in the file on disk, which is what made it hard to
  read. Fixed by capping optimization at `Level1` for that one model on that one architecture;
  the 15 diarization graphs keep full optimization. See `docs/ORT_FUSION_FP16_AARCH64.md`.
  - **It is a fusion-gate difference, not an aarch64 kernel gap** (RESULTS §7.40). *Every*
    aarch64 ORT build checked lacks the fp16 kernel, including macOS arm64 — where the model
    loads fine, because that build declines to apply the fusion to fp16 at all. Only
    linux/arm64 has the losing combination. It is a build-configuration divergence between two
    targets of the same ORT release.
  - A disable-list **does** work, under the name `GeluFusionL2` — the pass is registered twice
    and `GeluFusion`/`GeluFusionL1` both still fail. The level cap shipped anyway because it is
    bitwise identical to the unoptimized graph (0.000e+00, vs 9.58e-04 for the disable-list) and
    does not depend on an undocumented ORT-internal name that is **silently ignored when
    misspelled** and has already been renamed once.
- Scope `TARGETARCH` to the builder stage in the CPU Dockerfile. ARG scope is per-stage, so it
  expanded to empty and both platforms of a multi-platform buildx run shared one cargo registry
  cache id and contended on its lock.
- Copy `scripts/` into the Docker builder stages. Both image builds had begun failing to compile
  `diar-core`, whose provisioning module `include_str!`s the export scripts.
- Stop provisioning from aborting when the Hugging Face API is unreachable. The upstream revision
  is provenance, not a requirement, so warm-cache and `HF_HUB_OFFLINE=1` re-exports now succeed.
- Fix a fail-open in the PLDA exporter, which hardcoded `~/.cache/huggingface`, blind-scanned
  blobs and swallowed every error — so a cache miss produced a PLDA-less models directory while
  still exiting 0. It now resolves through `hf_hub_download` (honouring `HF_HOME`) and asserts
  all six arrays.
- `provision-models --set small` is no longer self-defeating. The startup gate defaulted to
  judging every directory as `fast`, so provisioning exited 0 having deliberately deleted the
  four batch-64 graphs and the server then refused to start over four "missing" files — with
  remediation text telling a laptop operator to build the tier they had just declined. The gate
  now reads the tier from the directory's own marker.
- Resolve `--smoke-clip` late, after the writability, idempotency, token and python checks. The
  documented "provision from OpenTranscribe's backend image" route died at exit 2 for want of a
  clip before it had so much as looked at the token — that image copies only the binary and three
  `.so`s out of this one, not the clip.
- Serialize requests under the `coreml` feature; speakrs' CoreML sessions are
  single-thread-at-a-time, so `DIAR_MAX_INFLIGHT` has no effect in that mode.
- Fall back to the 10 s fbank graph when the 30 s model is absent, instead of failing.
- Export the missing tail-b64 model artifact and re-enable the split-primary test.
- Make clippy pass with no exemptions.
- Make the provisioning read-only-directory test run on macOS, where the previous approach did
  not actually produce an unwritable directory.

### Security

- **Both serving images now run as non-root** (issue #7), as a `diar` account with a fixed,
  documented **uid/gid of `10001:10001`** — outside the 1000-1999 range `useradd` allocates on
  a normal host, so the ownership is never ambiguous. `no-new-privileges:true` is set on both
  compose services. Size is unchanged (CPU image: 195 MB, identical to 0.3.0).
  - Serving needs no write access anywhere: `/models` is `:ro`, the startup gate only `stat`s
    the marker, and the sole writable path is `/tmp/diar-native` (mode 1777).
  - **Provisioning is the path this would have broken.** A container user cannot write a host
    bind-mount it does not own, so a fixed uid would have made "export the models" require a
    `chown` — a quickstart that fails on step one. Instead the export runs as the *invoking*
    user (`--user "$(id -u):$(id -g)"`, wired through `DIAR_UID`/`DIAR_GID`), so the files land
    owned by the operator and serving reads them as 10001. **No `chown` in the normal flow.**
  - `Dockerfile.provision` reclaims root for its `apt`/`pip` layers and drops back at the end.
    It also pins `HF_HOME=/hf` (mode 1777), because under a `--user` override `~` is not
    writable and the export child would otherwise resolve `~/.cache/huggingface`.
  - `.env` is now gitignored. It was not, and `start.sh` writes a token into it.
- **Never expose the Hugging Face token**: passed by environment rather than argv, scrubbed from
  the exporter's stdout *and* stderr, marked `hide_env_values` so clap cannot render it in an
  error, and never logged.
- **Full media paths are logged as basenames only**, in span fields *and* in error text.
- Keep terms-gated model weights out of the Docker build context. The `.dockerignore` is an
  allowlist **by design** — a denylist fails open the moment someone adds a directory, and
  failing open here means redistributing gated weights. A CI job guards it.
- Document `SPEAKRS_TRT` and `SPEAKRS_TRT_CACHE` as dead configuration: they have no read sites
  and do nothing, following the TensorRT rollback (RESULTS §7.26).

### Dependencies

Six dependabot bumps were reviewed for 0.3.0; five landed. The two that change a base image were
built and run against real model weights rather than merged on green CI, because CI builds
neither the CUDA image nor anything with weights in it — the gap that would have let the cuda13
bump through. Evidence in RESULTS §7.41.

- **Base image: `ubuntu` 24.04 → 26.04** (`docker/Dockerfile.server-cpu`, `docker/Dockerfile.builder`).
  26.04 is a released LTS ("Resolute Raccoon"). glibc moves 2.39 → 2.43 and `libopenblas0`
  0.3.26 → 0.3.32; the package name is unchanged and the pinned rustc 1.97.1 installs cleanly on
  it. **The CPU image grows 195 MB → 246 MB (+26%)** — worth knowing, since small size is half of
  why that image exists. The CUDA image is untouched by this and stays on Ubuntu 24.04, which is
  what `nvidia/cuda:*-ubuntu24.04` pins.
- **Base image: `nvidia/cuda` 12.8.1 → 12.8.2** (`docker/Dockerfile.server`, `docker/Dockerfile.bench`).
  A patch bump *within the 12.8.x line*, which is what the dependabot ignore rule deliberately
  still allows; 13.x remains blocked and unusable. The hand-installed cuBLAS/cuFFT/cuRAND/cuDNN
  set and the ONNX Runtime 1.24.2 GPU tarball are unchanged.
- **`huggingface_hub` 1.28.0 → 1.29.0** (`scripts/provision/requirements.txt`). Provisioning-only.
  The download path we use (`hf_hub_download`, `model_info`, `HF_HOME`) is unchanged in 1.29.0;
  its one download-side change is an internal Xet connection-cache fix that removes a per-file API
  call. **Not exercised end to end** — a real export needs a gated HF token.
- **CI: `actions/checkout` 4 → 7, `actions/setup-python` 5 → 7, `docker/setup-qemu-action` 3 → 4.**
  All three are node20 → node24 moves needing Actions Runner ≥ v2.327.1, which every
  `ubuntu-latest` runner already exceeds. checkout v7 refuses to check out fork PR code under
  `pull_request_target` / `workflow_run`; neither trigger appears in this repo, and all seven
  checkout steps pass no inputs at all.
- **`ort` was not bumped and must not be.** No PR in this batch touched `Cargo.toml` or
  `Cargo.lock`. The `=2.0.0-rc.12` pin stands (RESULTS §4.26).

---


## [0.2.0] — 2026-08-20

The image running live in the OpenTranscribe stack. Published as `diar-server:0.2.0` /
`davidamacey/diar-native:0.2.0`; there is no corresponding git tag. Content per RESULTS §7.30:
**T9a shared sessions + pipelined fbank∥GPU + the arena knob + native media ingest.**

> The `0.2.0` tag was rebuilt in place on 2026-08-20 to pick up the two image-level entries below;
> the Rust source did not change.

### Added

- **T9a shared sessions** — concurrent requests run on cloned per-request handles against one
  engine's ONNX Runtime sessions, so VRAM no longer doubles under load and there is no engine
  mutex on the request path.
- **Pipelined fbank against the GPU**, superseding the native-fbank approach. (RESULTS §7.28)
- `SPEAKRS_ARENA_SHRINK`, a VRAM floor for 4 GB-tier cards at roughly a 20 % per-job cost.
  Default off.
- **Native media ingest** — the sidecar decodes mp3/m4a/flac/any-rate WAV in process via
  symphonia, instead of requiring a pre-decoded 16 kHz WAV from the caller.

### Changed

- Shrank the CUDA runtime image from 5.28 GB to 3.46 GB by basing it on `cuda:12.8.1-base` and
  installing only the six CUDA libraries the CUDA execution provider actually links. Diarization
  output was verified byte-identical to the prior image.

### Removed

- `libonnxruntime_providers_tensorrt.so` from the runtime image. It required `libnvinfer` and
  `libnvonnxparser`, which were never installed.

### Security

- Patched base-image CVEs by upgrading preinstalled packages in the runtime stage: Trivy went
  from 12 HIGH to 0.
- Untracked ~110 MB of gated Triton model derivatives from the repository and extended
  `.gitignore` to cover every `models*/` set. An earlier leak of 254 MB of gated weights had to be
  removed with `git filter-repo` (RESULTS §7.8) — hence the deliberately broad glob.
- Redacted private network addresses and filesystem paths from docs and validation scripts ahead
  of the repository going public.

### Accuracy and performance at 0.2.0

Recorded gates, all held: AMI-16 full **13.101 %** DER / exclusive **17.813 %**, Karpathy
**8.219 %**, VoxConverse **4.847 %**. Warm-engine speed: Karpathy 66.5 min diarized in **21.6 s**
(184× real time); ES2004a 36 min in **6.6 s**. End-to-end upload-to-transcript on the reference
file went from 108.4 s (Python) to **54.4 s**. Full detail and controls in
`validation/RESULTS.md`; the numbers to beat are in `docs/TEST_CORPORA_AND_BASELINES.md`.

---

## Earlier

Development before 0.2.0 is not itemized here. `PLAN.md` carries the roadmap and the locked
decisions, and `validation/RESULTS.md` is the append-only record of every measurement, including
the three decisions taken and reversed along the way: TensorRT rolled back (§7.26), native fbank
superseded by the fbank∥GPU pipeline (§7.28), and the sinc resampler rejected in favour of
`FftFixedIn` (§7.29).

[Unreleased]: https://github.com/attevon-llc/diar-native/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/attevon-llc/diar-native/releases/tag/v0.3.0
