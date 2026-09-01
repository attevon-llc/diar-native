# syntax=docker/dockerfile:1.7
# The build environment, as a reproducible artifact.
#
# WHY THIS EXISTS: CLAUDE.md named `diar-bench-builder:latest` as *the* way to build this
# project, but no Dockerfile in the repo produced it — it was a machine-local image. A fresh
# clone therefore could not reproduce the documented build at all, and there was nothing
# keeping it aligned with what CI does. Three environments (host, dev container, CI), one of
# them undefined, and no mechanism tying them together.
#
# This file and `.github/actions/setup-build-env/action.yml` install the SAME packages, and
# both defer to `rust-toolchain.toml` for the compiler. Keep them in step: the CI job
# `dev-container-parity` fails if the package lists diverge.
#
# The host cannot build this project — it has no openblas, and speakrs links against it. That
# is not a preference; a host `cargo build` fails outright. Container writes also leave
# `target/` root-owned, which is why CARGO_TARGET_DIR points outside the mounted tree.
#
#   docker build -f docker/Dockerfile.builder -t diar-native-builder:latest .
#   docker run --rm -v $PWD:/build -v /tmp/diar_target:/tmp/target -w /build \
#     -e CARGO_TARGET_DIR=/tmp/target diar-native-builder:latest \
#     cargo build --release -p diar-server -p diar-cli
#
# Build context: repo root. Only rust-toolchain.toml is copied in, so this layer caches until
# the toolchain actually moves.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Read from scripts/build-deps.txt rather than duplicated here. Three hand-maintained copies
# of this list (container, CI, workstation) is exactly how a host stops matching CI.
COPY scripts/build-deps.txt /tmp/build-deps.txt
RUN apt-get update \
 && grep -vE '^\s*(#|$)' /tmp/build-deps.txt | xargs apt-get install -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/* /tmp/build-deps.txt

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

# rustup reads rust-toolchain.toml and installs exactly the pinned channel + components, so the
# image cannot drift from CI or from a developer's workstation.
COPY rust-toolchain.toml /tmp/rust-toolchain.toml
# Install the pinned channel AND set it as the image default. Installing it without a default
# would work when building this repo (rustup honours the mounted rust-toolchain.toml) but leave
# `docker run <image> rustc --version` failing with "no default is configured" — a confusing
# way for the build environment to introduce itself. If rust-toolchain.toml later moves, rustup
# still prefers the file and fetches the newer channel at build time; the default is a fallback,
# not an override.
RUN CHANNEL="$(sed -n 's/^channel[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' /tmp/rust-toolchain.toml)" \
 && test -n "$CHANNEL" \
 && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal \
        --default-toolchain "$CHANNEL" -c rustfmt -c clippy \
 && rustc --version && cargo --version && cargo clippy --version && cargo fmt --version \
 && rm /tmp/rust-toolchain.toml

# speakrs' tests recurse deeply enough to blow the default 2 MB stack in release mode.
ENV RUST_MIN_STACK=16777216

WORKDIR /build
