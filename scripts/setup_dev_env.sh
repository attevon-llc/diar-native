#!/usr/bin/env bash
# Make a workstation match the build container and CI.
#
# Why this exists: the host, the dev container and CI are three separate environments, and
# nothing used to keep them aligned. The failure mode is not dramatic, just expensive — code
# passes locally, CI goes red, and nothing in the repo changed. This script installs the SAME
# system packages the container and CI install (scripts/build-deps.txt) and the SAME compiler
# (rust-toolchain.toml).
#
#   ./scripts/setup_dev_env.sh          install/update, then verify
#   ./scripts/setup_dev_env.sh --check  verify only, change nothing (exit 1 on mismatch)
#
# Installing needs sudo for apt. Everything else is per-user.
#
# NOTE: a host build covers default (CPU) features only. `--features cuda` additionally needs
# the CUDA toolkit and the ONNX Runtime GPU libraries that docker/Dockerfile.server installs;
# that build stays in the container.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_FILE="${REPO_ROOT}/scripts/build-deps.txt"
TOOLCHAIN_FILE="${REPO_ROOT}/rust-toolchain.toml"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mMISS\033[0m  %s\n' "$*"; }

[ -f "$DEPS_FILE" ]      || die "missing $DEPS_FILE"
[ -f "$TOOLCHAIN_FILE" ] || die "missing $TOOLCHAIN_FILE"

# Strip comments and blanks — the same parse the Dockerfile and CI use.
mapfile -t PACKAGES < <(grep -vE '^\s*(#|$)' "$DEPS_FILE")
WANT_CHANNEL="$(sed -n 's/^channel[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' "$TOOLCHAIN_FILE")"
[ -n "$WANT_CHANNEL" ] || die "could not read [toolchain] channel from $TOOLCHAIN_FILE"

echo "==> System packages (${#PACKAGES[@]} from scripts/build-deps.txt)"
MISSING=()
for pkg in "${PACKAGES[@]}"; do
  if dpkg -s "$pkg" >/dev/null 2>&1; then ok "$pkg"; else bad "$pkg"; MISSING+=("$pkg"); fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  if [ "$CHECK_ONLY" -eq 1 ]; then
    echo
    die "${#MISSING[@]} package(s) missing. Install with:
  sudo apt-get update && sudo apt-get install -y --no-install-recommends ${MISSING[*]}"
  fi
  echo
  echo "==> Installing ${#MISSING[@]} missing package(s) (needs sudo)"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${MISSING[@]}"
fi

echo
echo "==> Rust toolchain (pinned to ${WANT_CHANNEL} by rust-toolchain.toml)"
if ! command -v rustup >/dev/null 2>&1; then
  if [ "$CHECK_ONLY" -eq 1 ]; then
    die "rustup is not installed. Without it nothing honours rust-toolchain.toml and this host
will silently build with a different compiler than CI. Install:
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
  fi
  echo "  installing rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
  # shellcheck disable=SC1091
  . "${CARGO_HOME:-$HOME/.cargo}/env"
fi

# rustup reads rust-toolchain.toml automatically inside the repo; `rustup show` installs the
# pinned channel if it is absent. Deliberately does NOT touch the global default.
( cd "$REPO_ROOT" && rustup show active-toolchain >/dev/null )

HAVE_RUSTC="$(cd "$REPO_ROOT" && rustc --version | awk '{print $2}')"
if [ "$HAVE_RUSTC" = "$WANT_CHANNEL" ]; then
  ok "rustc ${HAVE_RUSTC}"
else
  bad "rustc ${HAVE_RUSTC} (want ${WANT_CHANNEL})"
  [ "$CHECK_ONLY" -eq 1 ] && die "toolchain mismatch inside the repo"
fi
# The component name and the cargo subcommand differ for rustfmt: the component is `rustfmt`,
# the subcommand is `cargo fmt`. Checking `cargo rustfmt` reports a missing component that is
# in fact installed.
check_component() { # <component-name> <cargo-subcommand>
  if ( cd "$REPO_ROOT" && cargo "$2" --version >/dev/null 2>&1 ); then
    ok "$1 (cargo $2)"
  else
    bad "$1 (cargo $2)"
    [ "$CHECK_ONLY" -eq 1 ] && die "missing component $1 — install with: rustup component add $1"
  fi
}
check_component rustfmt fmt
check_component clippy clippy

echo
echo "==> Parity summary"
printf '  host    rustc %s\n' "$(cd "$REPO_ROOT" && rustc --version | awk '{print $2}')"
if command -v docker >/dev/null 2>&1 && docker image inspect diar-native-builder:latest >/dev/null 2>&1; then
  printf '  builder rustc %s\n' "$(docker run --rm diar-native-builder:latest rustc --version 2>/dev/null | awk '{print $2}')"
else
  printf '  builder rustc (image not built — docker build -f docker/Dockerfile.builder -t diar-native-builder:latest .)\n'
fi
printf '  pinned  rustc %s\n' "$WANT_CHANNEL"

cat <<EOF

Host builds (default/CPU features):
  CARGO_TARGET_DIR=/tmp/diar_target_host cargo build --release -p diar-server -p diar-cli

Keep CARGO_TARGET_DIR off the repo's target/ — the container writes there as root, and mixing
the two leaves root-owned files you then cannot delete.
EOF
