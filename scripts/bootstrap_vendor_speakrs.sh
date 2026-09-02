#!/usr/bin/env bash
# Populates vendor/speakrs from our fork at a pinned commit AND applies our patch set, so a
# clean clone can build. Idempotent: safe to re-run.
#
# Fork:     https://github.com/attevon-llc/speakrs (Apache-2.0, avencera/speakrs upstream)
# Pin:      b0756b1 — the UPSTREAM base the patch set is generated against.
#
# NOT the fork's master @ 725cc4d. That commit already contains the 0.2.0 patch set, so the
# repo carried two contradictory reproduction stories: CLAUDE.md said "upstream b0756b1 + our
# patches as the working-tree diff" while this script said "fork master 725cc4d". The patch is
# produced by `git diff HEAD` from a tree at b0756b1, so it applies to b0756b1 and NOT to
# 725cc4d — verified both ways. Checking out 725cc4d and applying the patch fails; checking out
# b0756b1 and applying it succeeds. One base, one patch, one story.
#
# WHY THE PATCH STEP EXISTS. Until issue #3 the pin alone was enough: the fork's master already
# contained everything, our extra patches were performance-only, and diar-core compiled against
# an unpatched speakrs either way. `be11492` changed that — `DiarEngine::load` now sets
# `RuntimeConfig { fbank_pool, .. }`, a field that exists ONLY in patches/0001-*.patch. A tree
# built from the bare pin therefore fails with:
#
#     error[E0560]: struct `RuntimeConfig` has no field named `fbank_pool`
#
# and it fails ONLY on a clean clone or in CI. A developer's working tree already carries the
# change as an uncommitted vendored diff, so the machine that made the change is the one machine
# that cannot see the breakage. That is exactly how `main` shipped broken.
#
# So the patch is applied here, and `patches/0001-*.patch` — not a developer's working tree — is
# the authoritative statement of what we run.
#
# To bump the pin: update SPEAKRS_FORK_COMMIT below after re-validating
# (cargo test --release --no-default-features --features openblas-system,online
# inside the builder image, fixtures mounted from a prior vendor/speakrs checkout).
#
# After ANY vendored edit, regenerate the patch — `git diff HEAD`, never bare `git diff`, or
# staged changes are silently dropped:
#   cd vendor/speakrs && git diff HEAD > ../../patches/0001-cuda-performance-patch-set.patch

set -euo pipefail

SPEAKRS_FORK_URL="https://github.com/attevon-llc/speakrs.git"
SPEAKRS_FORK_COMMIT="b0756b1f39e63f3cb4d49ceca68ba0265d603848"
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

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "error: missing $PATCH_FILE — cannot reproduce the tree we build" >&2
  exit 1
fi

# `git apply` rather than `patch`: it fails atomically, so a partial application cannot leave a
# tree that compiles but is not what the patch describes. --check first for a clear message.
if git -C "$VENDOR_DIR" apply --check "$PATCH_FILE" 2>/dev/null; then
  git -C "$VENDOR_DIR" apply "$PATCH_FILE"
  echo "applied $(basename "$PATCH_FILE") ($(grep -c '^diff --git' "$PATCH_FILE") files)" >&2
elif git -C "$VENDOR_DIR" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "patch already applied — leaving the tree as-is" >&2
else
  echo "error: $(basename "$PATCH_FILE") does not apply cleanly to $SPEAKRS_FORK_COMMIT." >&2
  echo "       The pin and the patch set have diverged. Regenerate the patch against the pin," >&2
  echo "       or bump SPEAKRS_FORK_COMMIT to a commit the patch applies to." >&2
  exit 1
fi

echo "vendor/speakrs at $(git -C "$VENDOR_DIR" rev-parse --short HEAD) + patch set (attevon-llc/speakrs fork)" >&2
echo "Fixture models are gitignored upstream — mount/copy them into vendor/speakrs/fixtures/models/ separately." >&2
