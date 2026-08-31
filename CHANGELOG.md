# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**A note on versions.** This project ships as a container image, and its version history lives in
image tags rather than in git — there are currently **no git tags**, and the workspace crates are
all still at `0.1.0`. `0.2.0` below is the tag of the image running live in the OpenTranscribe
stack (`diar-server:0.2.0`, published 2026-08-20), reconstructed from the commit history. The
work since then has **not** been assigned a release number yet; it sits under *Unreleased* until
someone decides what that number is.

Measurements referenced here are recorded in
[`validation/RESULTS.md`](validation/RESULTS.md), which is append-only.

---

## [Unreleased]

### Added

- Serve **CUDA and CPU from a single image and process**, selected per request via a `device`
  field on `/diarize` and `/embed_window`. On amd64 the CUDA image is now a strict superset of
  the CPU image — the ONNX Runtime CPU execution provider is statically linked into every build,
  so this costs no extra bytes. The device actually used is reported in an `x-diar-device`
  response header. (RESULTS §7.34)
- `DIAR_DEVICES` (comma-separated; first entry is the default) to load several engines at
  startup, and an optional `DIAR_MAX_INFLIGHT_CPU` sub-gate beneath the global
  `DIAR_MAX_INFLIGHT`.
- **Model provisioning built into the `diar-server` binary.** `provision-models` turns a Hugging
  Face token into a verified models directory; `verify-models` runs a five-stage smoke test
  against the same ONNX Runtime the server uses; `check-token` reports whether the token is valid
  and the community-1 gate has been accepted. All three write machine-readable JSON to stdout.
- Provisioning provenance recorded in `diar-provision.json` (export-recipe version, exporter
  version, file inventory) — checked cheaply at startup and thoroughly by `verify-models`.
- **A startup model gate.** A missing required model file is now fatal, with a message naming the
  provisioning command and the Hugging Face gate URL. `DIAR_ALLOW_UNVERIFIED_MODELS` is the
  escape hatch.
- **`/readyz`**, which returns 503 with an actionable reason until the models are provisioned and
  verified. `/healthz` remains liveness-only and stays 200 whenever the process is serving.
- Model and device state on `/healthz`: loaded `devices` versus compiled-in `supported_devices`,
  the models directory, state and set, and whether the gender classifier was provisioned.
- **Structured logging.** `diar-server` now installs a tracing subscriber — it previously
  installed none, so `RUST_LOG` was inert in the deployed artifact and every engine log was
  discarded. Records go to **stdout** (so `docker logs` and compose capture them) while fatal
  startup errors stay on stderr; `RUST_LOG` filters and `DIAR_LOG_FORMAT=text|json` selects
  rendering. The `provision-models` / `verify-models` subcommands install no subscriber, so their
  JSON output is never interleaved with log records. (RESULTS §7.37)
- One log span per request carrying request id, endpoint, device, audio basename and gender flag,
  closed by a record with `duration_ms`, outcome, and either speaker/segment counts or an error
  class — plus one startup record derived from the `/healthz` body, so the two cannot drift.
- Propagation of an inbound `x-request-id` (sanitized before logging), echoed on both success and
  failure responses so a single id spans caller and sidecar.
- **`audio_path` as an alias for `wav_path`** on `/diarize` and `/embed_window`. The field accepts
  any symphonia-decodable media; the old name was pushing callers to transcode first. `wav_path`
  and `media_path` are still accepted.
- A `HEALTHCHECK` on both serving images, deliberately targeting `/healthz` (liveness) rather than
  `/readyz` — pointing it at readiness would mark every not-yet-provisioned container unhealthy
  and fail `compose up --wait` for the whole stack.
- The smoke-test clip (~832 KB) baked into both serving images, so `verify-models` runs in-image;
  plus a fallback provisioning image carrying CPU-only torch for operators on the plain image.
- `scripts/bootstrap_vendor_speakrs.sh`, which reproduces the vendored, patched speakrs tree from
  the public `attevon-llc/speakrs` fork at a pinned commit.
- CoreML support for Apple Silicon behind a `coreml` feature, wired through `diar-core`,
  `diar-server` and `diar-cli`. Not reachable through Docker — macOS grants containers no Metal
  access regardless of image architecture. (RESULTS §7.31)
- A CPU-only multi-architecture image variant (`linux/amd64` + `linux/arm64`), 189 MB.
- Apache-2.0 `LICENSE`.

### Changed

- Cut the Docker **build context** from 8.3 GB to 302 MB with an allowlist `.dockerignore`.
- Scale embedding-session intra-op thread count with core count in CPU mode. (RESULTS §7.32)
- Compare model sets by content multiset of `(dtype, shape, sha256)` rather than by initializer
  name. `torch.onnx.export(dynamo=True)` assigns those names at trace time, which produced false
  "13/15 initializers differ" reports on bit-identical graphs.
- `DIAR_DEVICES` takes precedence over `DIAR_MODE` when both are set. With neither set, behaviour
  is unchanged — including the long-standing "unset or unrecognized means `cuda`" default.

### Fixed

- Serialize requests under the `coreml` feature; speakrs' CoreML sessions are
  single-thread-at-a-time, so `DIAR_MAX_INFLIGHT` has no effect in that mode.
- Fall back to the 10 s fbank graph when the 30 s model is absent, instead of failing.
- Stop provisioning from aborting when the Hugging Face API is unreachable. The upstream revision
  is provenance, not a requirement, so warm-cache and `HF_HUB_OFFLINE=1` re-exports now succeed.
- Fix a fail-open in the PLDA exporter, which hardcoded `~/.cache/huggingface`, blind-scanned
  blobs and swallowed every error — so a cache miss produced a PLDA-less models directory while
  still exiting 0. It now resolves through `hf_hub_download` (honouring `HF_HOME`) and asserts
  all six arrays.
- Copy `scripts/` into the Docker builder stages. Both image builds had begun failing to compile
  `diar-core`, whose provisioning module `include_str!`s the export scripts.
- Scope `TARGETARCH` to the builder stage in the CPU Dockerfile. It expanded to empty, so both
  platforms of a multi-platform buildx run shared one cargo registry cache id and contended on
  its lock.
- Export the missing tail-b64 model artifact and re-enable the split-primary test.
- Document `SPEAKRS_TRT` and `SPEAKRS_TRT_CACHE` as dead configuration — they have no read sites
  and do nothing, following the TensorRT rollback (RESULTS §7.26).

### Security

- Never expose the Hugging Face token: it is passed by environment rather than argv, scrubbed
  from exporter stdout and stderr, hidden from CLI error rendering, and never logged. Full media
  paths are logged as basenames only.
- Keep terms-gated model weights out of the Docker build context. The `.dockerignore` is an
  allowlist by design — a denylist fails open the moment someone adds a directory, and failing
  open here means redistributing gated weights.

### Changed — repository practices

- Added CI (format, clippy, CPU build, tests, `.dockerignore` guard, CPU image build),
  `.pre-commit-config.yaml`, `CONTRIBUTING.md`, `SECURITY.md`, `CODEOWNERS`, `rustfmt.toml`,
  `clippy.toml`, `.editorconfig`, `.gitattributes`, and this changelog. CI builds only the
  default (CPU) feature set, downloads no model weights, and requires no secrets.

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

[Unreleased]: https://github.com/attevon-llc/diar-native/compare/main...HEAD
