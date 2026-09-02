#!/usr/bin/env bash
# Build, scan and publish the release images. RUN THIS LOCALLY — not in CI.
#
# WHY LOCAL. GitHub-hosted runners give ~14 GB of disk. The CUDA image alone is ~3 GB before
# any build artifacts, and a release needs five images across two architectures. The publish
# half of .github/workflows/release.yml was never viable on hosted runners, and the one time it
# ran (the v0.3.0 tag push) it failed anyway — on a missing-credentials gate, before it could
# discover the disk problem. This script is the supported path.
#
#   ./scripts/release.sh 0.3.1                 # build + scan, push NOTHING
#   ./scripts/release.sh 0.3.1 --push          # build + scan + push
#   ./scripts/release.sh 0.3.1 --push --latest # ... and move :latest to this version
#
# Default is build-and-scan only: publishing is irreversible in practice (a tag someone has
# already pulled cannot be un-pulled), so it takes an explicit flag.
#
# WHAT IT PRODUCES — five images, six tags. Every tag is SINGLE-PLATFORM by design; see
# .github/workflows/release.yml's header and README "Published images" for why `:latest` and
# `:<ver>` stay amd64 (they are the CUDA superset, and no aarch64 ONNX Runtime GPU build
# exists — issue #4).
#
#   <ver>                  linux/amd64  CUDA + CPU superset, from Dockerfile.server
#   <ver>-cpu              linux/amd64  CPU only, from Dockerfile.server-cpu
#   <ver>-cpu-arm64        linux/arm64  CPU only, same Dockerfile
#   <ver>-provision        linux/amd64  adds Python for the ONNX export
#   <ver>-provision-arm64  linux/arm64  same
#   latest                 linux/amd64  alias of <ver>, only with --latest
#
# REQUIREMENTS
#   * `docker login` already done (this script never touches credentials)
#   * a buildx builder with a native linux/arm64 node — qemu emulation is not used here
#     because an emulated Rust build is slow enough to be a different kind of risk
#   * trivy on PATH (scanning is not optional; use --skip-scan only with a reason)
set -euo pipefail

REGISTRY="davidamacey/diar-native"
ARM64_BUILDER="${ARM64_BUILDER:-opentranscribe-multiarch}"
# A docker CONTEXT pointing at a native arm64 daemon. Needed separately from the buildx
# builder because the provisioning image is FROM a locally-built base, which the
# container-driver builder cannot see.
ARM64_CONTEXT="${ARM64_CONTEXT:-remote-arm64}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VERSION="${1:-}"
PUSH=0
MOVE_LATEST=0
SKIP_SCAN=0
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push) PUSH=1 ;;
    --latest) MOVE_LATEST=1 ;;
    --skip-scan) SKIP_SCAN=1 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
  shift
done

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok() { printf '  \033[32mok\033[0m   %s\n' "$*"; }

[[ -n "$VERSION" ]] || die "usage: $0 <version> [--push] [--latest] [--skip-scan]"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]] || die "version '$VERSION' is not semver-ish"

# ---------------------------------------------------------------------------------------
# Preflight. Every check here has cost someone real time at least once.
# ---------------------------------------------------------------------------------------
step "Preflight"

[[ -z "$(git status --porcelain)" ]] || die "working tree is dirty — release from a clean tree
so the images match a commit someone can check out"

HEAD_SHA="$(git rev-parse --short HEAD)"
ok "clean tree at $HEAD_SHA"

# The vendored tree is the single most likely thing to be wrong, and it is invisible locally:
# a developer's working tree carries the patch set as an uncommitted diff, so the machine that
# made a vendored change is the one machine that cannot see a broken bootstrap. This is exactly
# how a release once shipped from a tree CI could not build.
[[ -d vendor/speakrs ]] || die "vendor/speakrs missing — run scripts/bootstrap_vendor_speakrs.sh"
if ! git -C vendor/speakrs diff --quiet HEAD 2>/dev/null; then
  if ! diff -q <(git -C vendor/speakrs diff HEAD) patches/0001-cuda-performance-patch-set.patch >/dev/null 2>&1; then
    die "vendor/speakrs differs from patches/0001-cuda-performance-patch-set.patch.
Regenerate it (git diff HEAD, NOT bare git diff) or the published binary will not match the
patch file that claims to describe it."
  fi
fi
ok "vendor/speakrs matches the committed patch set"

command -v docker >/dev/null || die "docker not on PATH"
if [[ $SKIP_SCAN -eq 0 ]]; then
  command -v trivy >/dev/null || die "trivy not on PATH (or pass --skip-scan, with a reason)"
fi
docker buildx inspect "$ARM64_BUILDER" >/dev/null 2>&1 \
  || die "buildx builder '$ARM64_BUILDER' not found — set ARM64_BUILDER, or create one with a
native linux/arm64 node (this script does not use qemu emulation)"
docker buildx inspect "$ARM64_BUILDER" 2>/dev/null | grep -q 'linux/arm64' \
  || die "builder '$ARM64_BUILDER' has no linux/arm64 platform"
ok "buildx builder '$ARM64_BUILDER' has a linux/arm64 node"

docker --context "$ARM64_CONTEXT" version --format '{{.Server.Arch}}' 2>/dev/null | grep -q arm64 \
  || die "docker context '$ARM64_CONTEXT' is not reachable or is not arm64 — set ARM64_CONTEXT.
The provisioning image is FROM a locally-built base, which the buildx container driver cannot
resolve, so that leg needs a real arm64 daemon."
ok "docker context '$ARM64_CONTEXT' is a reachable arm64 daemon"

if [[ $PUSH -eq 1 ]]; then
  grep -q 'index.docker.io' "${DOCKER_CONFIG:-$HOME/.docker}/config.json" 2>/dev/null \
    || die "not logged in to Docker Hub — run 'docker login' first"
  ok "docker hub credentials present"
fi

# ---------------------------------------------------------------------------------------
# Build. Local tags first; registry tags only after everything succeeds, so a failure
# halfway through cannot leave half a release named as if it were real.
# ---------------------------------------------------------------------------------------
step "Building ${VERSION} (this takes a while — the CUDA image is ~3 GB)"

docker build -f docker/Dockerfile.server     -t "rel:${VERSION}"      . >/dev/null
ok "amd64 CUDA superset  $(docker image inspect "rel:${VERSION}" --format '{{.Size}}' | numfmt --to=iec)"

docker build -f docker/Dockerfile.server-cpu -t "rel:${VERSION}-cpu"  . >/dev/null
ok "amd64 CPU            $(docker image inspect "rel:${VERSION}-cpu" --format '{{.Size}}' | numfmt --to=iec)"

# --load puts it in the LOCAL daemon; the provisioning build below needs it in the REMOTE
# arm64 daemon too, where `FROM` can resolve it without a registry round-trip.
docker buildx build --builder "$ARM64_BUILDER" --platform linux/arm64 \
  -f docker/Dockerfile.server-cpu -t "rel:${VERSION}-cpu-arm64" --load . >/dev/null
docker save "rel:${VERSION}-cpu-arm64" | docker --context "$ARM64_CONTEXT" load >/dev/null
ok "arm64 CPU            $(docker image inspect "rel:${VERSION}-cpu-arm64" --format '{{.Size}}' | numfmt --to=iec)"

# The serving images contain NO Python, so `provision-models` against them exits 6. The
# provisioning images are what the documented quickstart actually runs.
docker build -f docker/Dockerfile.provision --build-arg "BASE=rel:${VERSION}-cpu" \
  -t "rel:${VERSION}-provision" . >/dev/null
ok "amd64 provisioning   $(docker image inspect "rel:${VERSION}-provision" --format '{{.Size}}' | numfmt --to=iec)"

# NOT `docker buildx --builder`: that driver runs in its own container and cannot see images
# in the local daemon, so `FROM ${BASE}` resolves against Docker Hub and fails with
# "pull access denied" on a purely local tag. The arm64 CPU image was just --load'ed into the
# remote arm64 daemon, so build there with a plain `docker build`, where FROM resolves locally.
docker --context "$ARM64_CONTEXT" build -f docker/Dockerfile.provision \
  --build-arg "BASE=rel:${VERSION}-cpu-arm64" -t "rel:${VERSION}-provision-arm64" . >/dev/null
# Bring it back so the scan and tagging steps below treat every image the same way.
docker --context "$ARM64_CONTEXT" save "rel:${VERSION}-provision-arm64" | docker load >/dev/null
ok "arm64 provisioning   $(docker image inspect "rel:${VERSION}-provision-arm64" --format '{{.Size}}' | numfmt --to=iec)"

# ---------------------------------------------------------------------------------------
# Sanity. Cheap, and catches the class of mistake that is embarrassing to publish.
# ---------------------------------------------------------------------------------------
step "Sanity checks"
for t in "${VERSION}" "${VERSION}-cpu" "${VERSION}-cpu-arm64"; do
  arch="$(docker image inspect "rel:${t}" --format '{{.Architecture}}')"
  case "$t" in
    *-arm64) [[ "$arch" == "arm64" ]] || die "rel:${t} is ${arch}, expected arm64" ;;
    *)       [[ "$arch" == "amd64" ]] || die "rel:${t} is ${arch}, expected amd64" ;;
  esac
done
ok "architectures match their tag names"

reported="$(docker run --rm "rel:${VERSION}-cpu" --version 2>/dev/null | awk '{print $2}')"
[[ "$reported" == "$VERSION" ]] \
  || die "binary reports '$reported' but you are releasing '$VERSION' — bump the crate versions"
ok "binary reports $reported"

uid="$(docker run --rm --entrypoint id "rel:${VERSION}-cpu" -u)"
[[ "$uid" != "0" ]] || die "image runs as root — expected the non-root user (issue #7)"
ok "runs as uid $uid, not root"

# ---------------------------------------------------------------------------------------
step "Vulnerability scan"
if [[ $SKIP_SCAN -eq 1 ]]; then
  printf '  \033[33mSKIPPED\033[0m — record why in the release notes\n'
else
  for t in "${VERSION}" "${VERSION}-cpu" "${VERSION}-cpu-arm64" "${VERSION}-provision" "${VERSION}-provision-arm64"; do
    n="$(trivy image --quiet --severity HIGH,CRITICAL --format json "rel:${t}" 2>/dev/null \
      | python3 -c 'import json,sys; print(sum(len(r.get("Vulnerabilities") or []) for r in (json.load(sys.stdin).get("Results") or [])))')"
    [[ "$n" == "0" ]] || die "rel:${t} has ${n} HIGH/CRITICAL findings — fix or accept explicitly"
    ok "rel:${t}: 0 HIGH/CRITICAL"
  done
fi

# ---------------------------------------------------------------------------------------
step "Tagging ${REGISTRY}"
TAGS=("${VERSION}" "${VERSION}-cpu" "${VERSION}-cpu-arm64" "${VERSION}-provision" "${VERSION}-provision-arm64")
for t in "${TAGS[@]}"; do docker tag "rel:${t}" "${REGISTRY}:${t}"; ok "${REGISTRY}:${t}"; done
if [[ $MOVE_LATEST -eq 1 ]]; then
  docker tag "rel:${VERSION}" "${REGISTRY}:latest"
  TAGS+=("latest")
  ok "${REGISTRY}:latest -> ${VERSION} (amd64 CUDA)"
fi

if [[ $PUSH -eq 0 ]]; then
  step "Built and scanned, NOTHING PUSHED"
  echo "  Re-run with --push (and --latest to move the floating tag) when you are ready."
  exit 0
fi

step "Pushing"
for t in "${TAGS[@]}"; do
  docker push "${REGISTRY}:${t}" >/dev/null
  ok "pushed ${REGISTRY}:${t}"
done

# ---------------------------------------------------------------------------------------
# Verify against the REGISTRY, not the local daemon. A local tag proves nothing about what
# a stranger will pull.
# ---------------------------------------------------------------------------------------
step "Verifying the published tags"
for t in "${TAGS[@]}"; do
  digest="$(docker buildx imagetools inspect "${REGISTRY}:${t}" --format '{{.Manifest.Digest}}' 2>/dev/null || echo '?')"
  ok "${t}  ${digest}"
done

cat <<EOF

Published ${#TAGS[@]} tags of ${VERSION} from ${HEAD_SHA}.

Next:
  * git tag -a v${VERSION} && git push origin v${VERSION}
  * gh release create v${VERSION} --notes-file <(extract the CHANGELOG section)
  * put the digests above in README "Published images"
  * bump the pinned digest in transcribe-app backend/Dockerfile.prod, or none of this ships
EOF
