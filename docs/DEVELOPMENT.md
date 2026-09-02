# Development

Only needed if you are **changing** diar-native. Running it needs no clone at all — see the
[README](../README.md).

[`CONTRIBUTING.md`](../CONTRIBUTING.md) is the companion to this page: it holds the PR flow, the
style rules and the exact command blocks. This page is the orientation — what to build, what is
pinned and why, and how a release is cut.

- [Get a working tree that builds](#get-a-working-tree-that-builds)
- [Building](#building)
- [What is pinned, and why](#what-is-pinned-and-why)
- [The vendored speakrs patch set](#the-vendored-speakrs-patch-set)
- [Testing](#testing)
- [Cutting a release](#cutting-a-release)

---

## Get a working tree that builds

```bash
git clone https://github.com/attevon-llc/diar-native && cd diar-native
./scripts/bootstrap_vendor_speakrs.sh    # REQUIRED — a clean clone will not build without it
./start.sh --build                       # compile from this checkout, provision, serve, wait on /readyz
./start.sh --cli meeting.wav             # one-shot diarization, no server
```

`vendor/speakrs/` is gitignored and **not** a submodule, so a fresh clone has nothing there.
`bootstrap_vendor_speakrs.sh` clones our fork (`attevon-llc/speakrs`) at a pinned commit and
applies `patches/0001-cuda-performance-patch-set.patch`. It is idempotent — re-run it to refresh.

`diar-cli` is the one binary that is not published, so `--cli` always builds. Without `--build`,
`./start.sh` pulls the published image matching your architecture and GPU.

Match your workstation to the build container and CI in one step:

```bash
./scripts/setup_dev_env.sh           # install/update the pinned toolchain + system packages
./scripts/setup_dev_env.sh --check   # verify only, change nothing (exit 1 on mismatch)
```

It installs the packages listed in `scripts/build-deps.txt` — the same list the Dockerfile and
the `setup-build-env` CI action parse, with the `dev-container-parity` CI job failing if they
diverge — and ensures rustup has the channel named in `rust-toolchain.toml`.

## Building

**Build in the container.** The canonical build, and the only one that can produce a CUDA
binary:

```bash
docker build -f docker/Dockerfile.builder -t diar-native-builder:latest .

docker run --rm -v "$PWD":/build -v /tmp/diar_target:/tmp/target \
  -w /build -e CARGO_TARGET_DIR=/tmp/target \
  diar-native-builder:latest \
  cargo build --release --features cuda -p diar-server -p diar-cli
```

`openblas` is needed because speakrs is built with `openblas-system`; `libclang` is for bindgen
(`ort-sys`). A **host** build works for fast iteration on **default (CPU) features only**, once
`setup_dev_env.sh` has installed those packages — point `CARGO_TARGET_DIR` at a `/tmp` directory.
`--features cuda` additionally needs the CUDA toolkit and the ONNX Runtime GPU libraries that
`docker/Dockerfile.server` installs, so that build stays in the container.

> **Two traps that cost real time.** `target/` and `Cargo.lock` end up **root-owned** after a
> container build; fix it from a container rather than with host `sudo`. And always scope
> formatting with `-p`: a bare `cargo fmt --all` reformats `vendor/speakrs` too, because cargo
> resolves path dependencies — which silently invalidates the patch file.

Images:

```bash
docker build -f docker/Dockerfile.server     -t diar-server:dev .   # CUDA
docker build -f docker/Dockerfile.server-cpu -t diar-server:dev .   # multi-arch CPU-only
```

To test against a local `transcribe-app`, set `DIAR_NATIVE_IMAGE=diar-server:dev` in **that**
repo's `.env`. Note this only exercises the sidecar; production consumes the **binary**, so
testing that path also means rebuilding the backend image against your local diar-native image.
See [ARCHITECTURE.md](ARCHITECTURE.md#production-consumes-the-binary-not-the-image).

## What is pinned, and why

| pin | value | why |
|---|---|---|
| Rust toolchain | **1.97.1** (`rust-toolchain.toml`) | Pinned, not `stable`, on purpose. `stable` is a moving target and its failure mode is confusing and always badly timed: a new stable ships new clippy lints, CI turns red on code that passed locally an hour ago, and nothing in the repo changed. Worse, the dev container is a frozen image, so it would keep passing while CI kept failing. Honoured automatically by rustup everywhere. |
| MSRV | **1.88.0** (`[workspace.package] rust-version`) | The *minimum* supported version. Deliberately distinct from the toolchain pin, which is the one version everything is **built** with. |
| `ort` | **`=2.0.0-rc.12`** | **Do not bump.** rc.13 ships a static core that mismatches the 1.24.2 provider libs and **fails at session load** (RESULTS §4.26). |
| Base image | **ubuntu 24.04** | Do not re-bump. 26.04 is a severe arm64 accuracy regression — see [DEPLOYMENT.md](DEPLOYMENT.md#the-base-image-is-pinned-to-ubuntu-2404) and issue #18. |
| `vendor/speakrs` | **`b0756b1`** + our patch set | See below. |

Bumping the toolchain is a deliberate act: change the version, rebuild the builder image, and run
fmt + clippy + tests before pushing. Expect new lints.

## The vendored speakrs patch set

`vendor/speakrs/` is an upstream clone whose **working-tree diff is the patch set**. After **any**
edit under `vendor/speakrs`:

```bash
cd vendor/speakrs
git diff HEAD > ../../patches/0001-cuda-performance-patch-set.patch
```

Use `git diff HEAD`, **not** a bare `git diff` — staged changes are otherwise silently dropped.
Never commit inside the vendored repo; `vendor/` is fully gitignored.

> **This is the check a developer cannot perform by eye.** The patch set lives as an uncommitted
> diff, so the machine that makes a vendored change is the one machine that cannot see a broken
> bootstrap. `main` has shipped unbuildable exactly this way. `scripts/release.sh` refuses to
> release if `vendor/speakrs` has drifted from the patch file.

Upstream contribution queue: [`UPSTREAM_PRS.md`](UPSTREAM_PRS.md). Opening PRs or issues against
`avencera/speakrs` needs explicit operator approval.

## Testing

```bash
# workspace tests
cargo test --release -p diar-core -p diar-server -p diar-cli

# speakrs' own suite (94 tests) — in the container, from the vendored tree
docker run --rm -v "$PWD":/build -v /tmp/diar_target:/tmp/target \
  -w /build/vendor/speakrs -e CARGO_TARGET_DIR=/tmp/target -e RUST_MIN_STACK=16777216 \
  diar-native-builder:latest \
  cargo test --release --no-default-features --features openblas-system,online
```

`--no-default-features` is **required**; plain `--features openblas-system` fails with a
duplicate BLAS. `RUST_MIN_STACK=16777216` is required because speakrs pipeline and ORT worker
threads overflow the 2 MiB default. Fixture models live only in `vendor/speakrs/fixtures/models/`
— mount them into any other clone.

Tests needing the gated model artifacts are `#[ignore]`d and gated on environment variables, so
they never fail for someone without the weights:

```bash
DIAR_TEST_MODELS_DIR=/build/models_folded \
  cargo test --release -p diar-core --test provision_smoke -- --ignored --nocapture
```

Before every commit:

```bash
.venv/bin/pre-commit run --all-files
```

Call the venv binary directly rather than activating the environment first. Never bypass hooks
(`--no-verify`) or signing.

**Benchmarks are governed by [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md)**, which is law —
one timed leg at a time on a quiet machine, VRAM sampled *during* the run, every speed claim
shipped with its accuracy check, and `validation/RESULTS.md` is append-only.

## Cutting a release

Releases are built, scanned and pushed **locally**, by `scripts/release.sh`. Not by CI.

```bash
./scripts/release.sh 0.3.1                 # build + scan, push NOTHING
./scripts/release.sh 0.3.1 --push          # ... and publish
./scripts/release.sh 0.3.1 --push --latest # ... and move :latest to this version
```

`--skip-scan` exists as an escape hatch. Publishing takes an explicit `--push` because a tag
someone has already pulled cannot be un-pulled.

**Why not GitHub Actions.** Hosted runners give ~14 GB of disk; the CUDA image alone is ~3 GB
before build artifacts, and a release is five images across two architectures. A publish workflow
existed and was removed — the one time it ran, on the `v0.3.0` tag push, it failed at its
credentials gate before it could even discover the disk problem, and every 0.3.0 image was built
and pushed by hand anyway.

The script does what that workflow tried to, plus the checks a workflow could not:

- Refuses a **dirty tree**, so the images match a commit someone can check out.
- Refuses if **`vendor/speakrs` has drifted from the patch file** — the check that matters most,
  and the one a developer cannot do by eye.
- Asserts each image's **architecture matches its tag name**, that the **binary reports the
  version being released** (catching an un-bumped crate version), and that the image **does not
  run as root**.
- Fails the release on any **HIGH/CRITICAL** trivy finding, across all five images.
- Verifies the published digests **against the registry**, not the local daemon — a local tag
  proves nothing about what a stranger will pull.

Then update [`CHANGELOG.md`](../CHANGELOG.md) (Keep a Changelog format), tag annotated
(`git tag -a`), and record the new digests in
[DEPLOYMENT.md](DEPLOYMENT.md#published-images).

---

See also: [`CONTRIBUTING.md`](../CONTRIBUTING.md) · [ARCHITECTURE.md](ARCHITECTURE.md) ·
[PERFORMANCE.md](PERFORMANCE.md) · [README](../README.md)
