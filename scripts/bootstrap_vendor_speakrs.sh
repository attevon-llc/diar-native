#!/usr/bin/env bash
# Populates vendor/speakrs from our fork at a pinned commit AND applies our patch set, so a
# clean clone can build. Idempotent: safe to re-run.
#
# Fork:     https://github.com/attevon-llc/speakrs, branch attevon/production-0.3.1
# Pin:      5517abc — the production patch set as ONE COMMIT on top of upstream b0756b1.
#
# WHY A FORK COMMIT AND NOT base+patch. This used to check out upstream b0756b1 and apply
# patches/0001-*.patch on top. That has one failure mode, and we hit it: the pin and the patch
# can drift apart, and NOBODY CAN SEE IT LOCALLY, because a developer's working tree already
# carries the change as an uncommitted diff. The machine that makes a vendored edit is the one
# machine that cannot observe a broken bootstrap. diar-native shipped unbuildable that way, and
# CI was red for hours before anyone looked.
#
# One commit removes the class. There is no second artifact to keep in step.
#
# patches/0001-cuda-performance-patch-set.patch still exists, but it is now DERIVED, not an
# input. It is what the upstream PR series is cut from, and scripts/release.sh refuses to
# publish if it disagrees with the vendored tree. Regenerate it after any vendored change:
#
#   cd vendor/speakrs && git diff b0756b1..HEAD > ../../patches/0001-cuda-performance-patch-set.patch
#
# To bump the pin: push a new commit to the fork branch, update SPEAKRS_FORK_COMMIT, regenerate
# the patch, and re-validate (cargo test --release --no-default-features
# --features openblas-system,online inside the builder image, fixtures mounted).

set -euo pipefail

SPEAKRS_FORK_URL="https://github.com/attevon-llc/speakrs.git"
SPEAKRS_FORK_COMMIT="5517abce1274fdaa03bdb9c0a53defc48cf2b03a"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${REPO_ROOT}/vendor/speakrs"
PATCH_FILE="${REPO_ROOT}/patches/0001-cuda-performance-patch-set.patch"

if [[ -d "$VENDOR_DIR/.git" ]]; then
  echo "vendor/speakrs already present at $VENDOR_DIR — fetching pin only" >&2
  git -C "$VENDOR_DIR" remote set-url origin "$SPEAKRS_FORK_URL"
  git -C "$VENDOR_DIR" fetch origin "$SPEAKRS_FORK_COMMIT"
  # --force: the working tree carries the patch set as an uncommitted diff by design, so a
  # plain checkout refuses. Discarding it is correct — the patch file is reapplied below.
  git -C "$VENDOR_DIR" checkout --force --detach "$SPEAKRS_FORK_COMMIT"
else
  git clone "$SPEAKRS_FORK_URL" "$VENDOR_DIR"
  git -C "$VENDOR_DIR" checkout --detach "$SPEAKRS_FORK_COMMIT"
fi

# The pin already contains the patch set, so there is nothing to apply. Verify instead: if the
# committed patch file does not match what the pinned commit actually changed, one of them is
# stale and the tree we build is not the tree the patch claims to describe.
if [[ -f "$PATCH_FILE" ]]; then
  if ! diff -q <(git -C "$VENDOR_DIR" diff b0756b1f39e63f3cb4d49ceca68ba0265d603848..HEAD) \
                "$PATCH_FILE" >/dev/null 2>&1; then
    echo "warning: patches/$(basename "$PATCH_FILE") does not match the pinned commit's diff." >&2
    echo "         The build is unaffected — the pin is the source of truth — but the patch" >&2
    echo "         file feeds the upstream PR series and should be regenerated:" >&2
    echo "           cd vendor/speakrs && git diff b0756b1..HEAD > ../../$(basename "$PATCH_FILE")" >&2
  fi
fi

echo "vendor/speakrs at $(git -C "$VENDOR_DIR" rev-parse --short HEAD) (attevon-llc/speakrs, patch set included in the pin)" >&2
echo "Fixture models are gitignored upstream — mount/copy them into vendor/speakrs/fixtures/models/ separately." >&2
