#!/usr/bin/env bash
# Populates vendor/speakrs from our fork at a pinned commit, so the container
# build (docker/Dockerfile.server, docker/Dockerfile.bench) has a source tree
# to COPY. Idempotent: safe to re-run.
#
# Fork:     https://github.com/attevon-llc/speakrs (Apache-2.0, avencera/speakrs upstream)
# Pin:      master @ 725cc4d — "Merge production patch set (0.2.0) into master"
#           (verified byte-identical to the diar-server:0.2.0 build; 94/94 tests
#           pass from a clean clone at this commit — see docs/UPSTREAM_PRS.md)
#
# To bump the pin: update SPEAKRS_FORK_COMMIT below after re-validating
# (cargo test --release --no-default-features --features openblas-system,online
# inside diar-bench-builder, fixtures mounted from a prior vendor/speakrs checkout).

set -euo pipefail

SPEAKRS_FORK_URL="https://github.com/attevon-llc/speakrs.git"
SPEAKRS_FORK_COMMIT="725cc4dacde9e9bad2a0698ed566dfb6680ce9fd"
VENDOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/vendor/speakrs"

if [ -d "$VENDOR_DIR/.git" ]; then
  echo "vendor/speakrs already present at $VENDOR_DIR — fetching pin only" >&2
  git -C "$VENDOR_DIR" remote set-url origin "$SPEAKRS_FORK_URL"
  git -C "$VENDOR_DIR" fetch origin "$SPEAKRS_FORK_COMMIT"
  git -C "$VENDOR_DIR" checkout --detach "$SPEAKRS_FORK_COMMIT"
else
  git clone "$SPEAKRS_FORK_URL" "$VENDOR_DIR"
  git -C "$VENDOR_DIR" checkout --detach "$SPEAKRS_FORK_COMMIT"
fi

echo "vendor/speakrs checked out at $(git -C "$VENDOR_DIR" rev-parse --short HEAD) (attevon-llc/speakrs fork)" >&2
echo "Fixture models are gitignored upstream — mount/copy them into vendor/speakrs/fixtures/models/ separately." >&2
