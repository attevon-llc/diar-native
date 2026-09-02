#!/usr/bin/env bash
# diar-native — one-command start, for people who have a clone.
#
# PULLS the published images by default, exports the models once (with YOUR HuggingFace
# token), starts the sidecar, waits for it to be ready, and tells you how to diarize a file.
# `--build` compiles everything from this checkout instead, which is the contributor path.
#
# You do NOT need this script, or a clone, to run diar-native. Two files — the published
# docker-compose.prod.yml and a .env — are the whole deployment; see the top of README.md.
# This script exists to detect your platform, prompt for the token and wait on /readyz.
#
# Safe to re-run: with the models already on disk it is a fast no-op that needs no token and
# no network. See docs/DEPLOYMENT.md.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

REPO_DIR="$PWD"
ENV_FILE="$REPO_DIR/.env"
EXAMPLE_FILE="$REPO_DIR/.env.example"
MARKER_NAME="diar-provision.json"

# ── the published release ────────────────────────────────────────────────────────────────
# THE one place the version lives in this script. Pinned rather than tracking `:latest` on
# purpose: `:latest` is the amd64 CUDA image, so on any other host it is not merely stale,
# it is the wrong architecture. Every tag below is derived from these two lines.
DIAR_VERSION=0.3.1
REGISTRY=davidamacey/diar-native

# The image's built-in non-root user. Must match docker/Dockerfile.server,
# docker/Dockerfile.server-cpu, docker-compose.yml and docs/PROVISIONING.md.
IMAGE_UID=10001
IMAGE_GID=10001

# ── output helpers ───────────────────────────────────────────────────────────────────────
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
  B=$'\033[1m'; R=$'\033[31m'; Y=$'\033[33m'; G=$'\033[32m'; N=$'\033[0m'
else
  B=''; R=''; Y=''; G=''; N=''
fi
say()  { printf '%s\n' "$*"; }
step() { printf '\n%s==>%s %s%s%s\n' "$G" "$N" "$B" "$*" "$N"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '%swarning:%s %s\n' "$Y" "$N" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
diar-native — fast, self-hosted speaker diarization.

USAGE
  ./start.sh [options]              pull if needed, provision if needed, serve
  ./start.sh --cli <audio-file>     diarize one file directly, no server at all

OPTIONS
  --build               Build the images from THIS checkout instead of pulling them.
                        Compiles the Rust workspace; minutes, not seconds. Use it when
                        you are changing the code, or running an unpublished commit.
  --cpu                 Force the CPU image even if a GPU is present.
  --gpu                 Require the GPU; fail instead of falling back to CPU.
  --check-token         Only check the HuggingFace token and gate, then exit.
  --provision           Force a re-export of the models even if they look fine.
  --rebuild             Force image work even if the image is already present. With
                        --build that is a rebuild; without it, a re-pull.
  --stop                Stop the sidecar and exit.
  --logs                Follow the sidecar's logs and exit.
  --models-dir <path>   Host directory for the models (default: ./models).
  --port <n>            Host port to publish (default: 8701).
  -h, --help            This text.

WHAT IT DOES ON A FIRST RUN
  1. Creates .env from .env.example and asks for your HuggingFace token.
  2. PULLS the published image that matches this machine — architecture and GPU are
     detected, because the CUDA image is amd64-only and the right tag has to be named.
     ~195 MB on a CPU host.
     (--build compiles it from source instead: several minutes.)
  3. Exports the diarization models into ./models — about 484 MB, roughly 3 minutes.
     They cannot be shipped for you: they derive from the gated pyannote
     speaker-diarization-community-1 weights, so every operator exports their own.
  4. Starts the sidecar and waits for GET /readyz to return 200.

A SECOND RUN skips 1-3 entirely and just starts the server.

YOU DO NOT NEED THIS SCRIPT
  The deployment is two files and no clone: fetch docker-compose.prod.yml and .env.example,
  set your token, `docker compose up`. See the top of README.md. This script is the
  convenience wrapper that picks your image, prompts for the token and waits on /readyz.

CONTAINER USER
  The images run as non-root uid:gid 10001:10001 and mount ./models READ-ONLY to serve.
  The one-time export runs as YOUR uid instead, so the models land owned by you and no
  chown or sudo is ever needed.
EOF
}

# ── argument parsing ─────────────────────────────────────────────────────────────────────
FORCE_CPU=false
FORCE_GPU=false
ONLY_CHECK_TOKEN=false
FORCE_PROVISION=false
REBUILD=false
DO_BUILD=false
DO_STOP=false
DO_LOGS=false
CLI_FILE=""
OPT_MODELS_DIR=""
OPT_PORT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=true ;;
    --cpu) FORCE_CPU=true ;;
    --gpu) FORCE_GPU=true ;;
    --check-token) ONLY_CHECK_TOKEN=true ;;
    --provision|--reprovision) FORCE_PROVISION=true ;;
    --rebuild) REBUILD=true ;;
    --stop) DO_STOP=true ;;
    --logs) DO_LOGS=true ;;
    --cli) shift; [[ $# -gt 0 ]] || die "--cli needs an audio file"; CLI_FILE="$1" ;;
    --models-dir) shift; [[ $# -gt 0 ]] || die "--models-dir needs a path"; OPT_MODELS_DIR="$1" ;;
    --port) shift; [[ $# -gt 0 ]] || die "--port needs a number"; OPT_PORT="$1" ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (try --help)" ;;
  esac
  shift
done

$FORCE_CPU && $FORCE_GPU && die "--cpu and --gpu are mutually exclusive"

# ── prerequisites ────────────────────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH."
docker info >/dev/null 2>&1 || die "cannot talk to the Docker daemon. Is it running, and is your user in the 'docker' group?"
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
else
  die "'docker compose' (v2) is required. The legacy 'docker-compose' binary is not supported."
fi

# ── .env ─────────────────────────────────────────────────────────────────────────────────
# Read one key out of .env WITHOUT sourcing it. Sourcing would execute whatever is in there,
# and this script is often the first thing a new operator runs.
env_get() {
  local key="$1" default="${2:-}" val
  [[ -f "$ENV_FILE" ]] || { printf '%s' "$default"; return; }
  val="$(grep -E "^[[:space:]]*${key}=" "$ENV_FILE" 2>/dev/null | tail -n1 | cut -d= -f2- || true)"
  val="${val%$'\r'}"
  [[ -n "$val" ]] || val="$default"
  printf '%s' "$val"
}

ensure_env_file() {
  if [[ -f "$ENV_FILE" ]]; then return; fi
  [[ -f "$EXAMPLE_FILE" ]] || die "neither .env nor .env.example exists — is this a complete checkout?"
  cp "$EXAMPLE_FILE" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  step "Created .env from .env.example"
  note "Edit it later to change ports, devices or logging. All of it is optional except the token."
}

# Ask for the token and store it. The value is never echoed, never logged, and never passed
# on a command line (which would expose it to `ps` for every user on the box) — it goes into
# .env, which is chmod 600 and gitignored, and reaches containers via --env-file.
prompt_for_token() {
  local token=""
  cat <<EOF

${B}A HuggingFace token is needed — once — to export the models.${N}

  The diarization models are derivatives of pyannote/speaker-diarization-community-1,
  which is a GATED repository. Nobody can legally redistribute the converted weights, so
  every operator exports their own copy locally. Nothing is uploaded, and the token is used
  only to download ~32 MB from HuggingFace.

  1. Create a READ token:  https://huggingface.co/settings/tokens
  2. Accept the terms, signed in as that same account, at:
     ${B}https://huggingface.co/pyannote/speaker-diarization-community-1${N}
     It is a short email-capture form, ${B}auto-approved${N} — there is no waiting list and no
     human review. The pipeline is CC-BY-4.0 and free.

  The token is stored in ./.env (chmod 600, gitignored) and nowhere else.

EOF
  if [[ ! -t 0 ]]; then
    die "no HUGGINGFACE_TOKEN in .env and stdin is not a terminal, so I cannot prompt.
       Set it non-interactively instead:
           printf 'HUGGINGFACE_TOKEN=%s\\n' \"\$YOUR_TOKEN\" >> .env
       or export HUGGINGFACE_TOKEN in the environment before running this script."
  fi
  printf 'Paste your HuggingFace token (input hidden, Enter to abort): '
  read -rs token || true
  printf '\n'
  [[ -n "$token" ]] || die "no token entered. Re-run ./start.sh when you have one."

  # Rewrite the placeholder line in place; never print the value.
  local tmp
  tmp="$(mktemp "${ENV_FILE}.XXXXXX")"
  chmod 600 "$tmp"
  if grep -qE '^[[:space:]]*HUGGINGFACE_TOKEN=' "$ENV_FILE"; then
    TOKEN_VALUE="$token" awk '
      /^[[:space:]]*HUGGINGFACE_TOKEN=/ && !done { print "HUGGINGFACE_TOKEN=" ENVIRON["TOKEN_VALUE"]; done=1; next }
      { print }
    ' "$ENV_FILE" > "$tmp"
  else
    cat "$ENV_FILE" > "$tmp"
    TOKEN_VALUE="$token" awk 'BEGIN { print "HUGGINGFACE_TOKEN=" ENVIRON["TOKEN_VALUE"] }' >> "$tmp"
  fi
  mv "$tmp" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  unset token
  note "Token saved to .env"
}

have_token() {
  local t
  t="$(env_get HUGGINGFACE_TOKEN "")"
  [[ -n "$t" ]] && [[ "$t" != "your_token_here" ]]
}

ensure_env_file

# ── settings ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR="${OPT_MODELS_DIR:-$(env_get DIAR_MODELS_HOST_DIR ./models)}"
AUDIO_DIR="$(env_get DIAR_AUDIO_HOST_DIR ./audio)"
HF_CACHE_DIR="$(env_get DIAR_HF_CACHE_HOST_DIR ./.hf-cache)"
PORT="${OPT_PORT:-$(env_get DIAR_PORT 8701)}"
GPU_DEVICE_ID="$(env_get DIAR_GPU_DEVICE_ID 0)"

# Bind-mount sources must exist BEFORE compose runs. If Docker auto-creates a missing bind
# source it makes it root-owned, and then the non-root export cannot write to it — the exact
# failure this whole design is meant to avoid.
mkdir -p "$MODELS_DIR" "$AUDIO_DIR" "$HF_CACHE_DIR"
MODELS_DIR_ABS="$(cd "$MODELS_DIR" && pwd)"
MARKER="$MODELS_DIR_ABS/$MARKER_NAME"

# ── architecture ─────────────────────────────────────────────────────────────────────────
# Not cosmetic. `:latest` and `:$DIAR_VERSION` are the amd64 CUDA image and there is no arm64
# entry under them — permanently, since no aarch64 ONNX Runtime GPU build exists — so an arm64
# host that pulls one does not get a clean "wrong architecture" error, it gets emulation, and
# the symptom is "diarization is mysteriously slow" rather than a failure anyone can act on.
# DIAR_ARCH overrides the detection, which is also how this selection is testable without an
# arm64 machine.
HOST_ARCH="${DIAR_ARCH:-$(uname -m)}"
case "$HOST_ARCH" in
  x86_64|amd64)  ARCH=amd64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) die "unsupported CPU architecture '$HOST_ARCH'.
       Images are published for x86_64/amd64 and aarch64/arm64. On anything else, build
       from this checkout instead: ./start.sh --build" ;;
esac

# ── GPU detection ────────────────────────────────────────────────────────────────────────
# Both halves are required. A host can have a perfectly good driver and still not be able to
# pass a GPU into a container; if we only checked nvidia-smi we would select the CUDA image
# and the container would simply fail to start, which reads as "this software is broken".
gpu_reason=""
detect_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    gpu_reason="no nvidia-smi on PATH"; return 1
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    gpu_reason="nvidia-smi present but reports no usable GPU"; return 1
  fi
  if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    gpu_reason="NVIDIA container runtime is not registered with Docker (install nvidia-container-toolkit)"
    return 1
  fi
  return 0
}

USE_GPU=false
if [[ "$ARCH" != amd64 ]]; then
  # The CUDA image is built and published for linux/amd64 only. Saying so here, rather than
  # letting the pull fail later, keeps the message about the real constraint.
  gpu_reason="no CUDA image is published for linux/$ARCH"
  $FORCE_GPU && die "--gpu requested but $gpu_reason"
elif $FORCE_CPU; then
  gpu_reason="--cpu requested"
elif detect_gpu; then
  USE_GPU=true
elif $FORCE_GPU; then
  die "--gpu requested but no usable GPU: $gpu_reason"
fi

# The Dockerfile is selected even when pulling: --cli always builds (diar-cli is not
# published), and the provisioning image is built locally if its published tag is missing.
if $USE_GPU; then
  DOCKERFILE=docker/Dockerfile.server
  DEVICES="cuda,cpu"
  CLI_MODE=cuda
  CLI_IMAGE=diar-native-cli:local-cuda
else
  DOCKERFILE=docker/Dockerfile.server-cpu
  DEVICES="cpu"
  CLI_MODE=cpu
  CLI_IMAGE=diar-native-cli:local-cpu
fi

# ── which images ─────────────────────────────────────────────────────────────────────────
# Pulling is the default because it is what almost everyone wants: ~195 MB and seconds,
# against several minutes of Rust compilation for a byte-identical result. Building is the
# specialist path — you changed the code, or you are on a commit that was never published.
#
# The provisioning image is deliberately the CPU-based one on every host, GPU included: the
# export runs `pipeline.to(torch.device("cpu"))` and never touches an accelerator, so a
# CUDA-based provisioning image would be 3 GB larger for no reason at all.
if $DO_BUILD; then
  if $USE_GPU; then
    SERVER_IMAGE="$(env_get DIAR_IMAGE diar-native:local-cuda)"
    PROVISION_IMAGE="$(env_get DIAR_PROVISION_IMAGE diar-native-provision:local-cuda)"
  else
    SERVER_IMAGE="$(env_get DIAR_IMAGE diar-native:local-cpu)"
    PROVISION_IMAGE="$(env_get DIAR_PROVISION_IMAGE diar-native-provision:local-cpu)"
  fi
  PROVISION_BASE="$SERVER_IMAGE"
else
  case "$ARCH" in
    arm64) PUBLISHED_SERVER="$REGISTRY:$DIAR_VERSION-cpu-arm64"
           PUBLISHED_PROVISION="$REGISTRY:$DIAR_VERSION-provision-arm64"
           PROVISION_BASE="$REGISTRY:$DIAR_VERSION-cpu-arm64" ;;
    *)     $USE_GPU && PUBLISHED_SERVER="$REGISTRY:$DIAR_VERSION" \
                    || PUBLISHED_SERVER="$REGISTRY:$DIAR_VERSION-cpu"
           PUBLISHED_PROVISION="$REGISTRY:$DIAR_VERSION-provision"
           PROVISION_BASE="$REGISTRY:$DIAR_VERSION-cpu" ;;
  esac
  SERVER_IMAGE="$(env_get DIAR_IMAGE "$PUBLISHED_SERVER")"
  PROVISION_IMAGE="$(env_get DIAR_PROVISION_IMAGE "$PUBLISHED_PROVISION")"
fi

# The published-image compose file has no `build:` key anywhere; the source one is paired
# with locally-tagged images. Same services, same volumes, different provenance.
$DO_BUILD && COMPOSE_FILES=(-f docker-compose.yml) || COMPOSE_FILES=(-f docker-compose.prod.yml)
$USE_GPU && COMPOSE_FILES+=(-f docker-compose.gpu.yml)

# Everything compose needs to interpolate. Exported here rather than written into .env so a
# --cpu run does not permanently rewrite the operator's file.
export DIAR_IMAGE="$SERVER_IMAGE"
export DIAR_PROVISION_IMAGE="$PROVISION_IMAGE"
DIAR_DEVICES="$(env_get DIAR_DEVICES "$DEVICES")"; export DIAR_DEVICES
export DIAR_MODELS_HOST_DIR="$MODELS_DIR_ABS"
export DIAR_AUDIO_HOST_DIR="$AUDIO_DIR"
export DIAR_HF_CACHE_HOST_DIR="$HF_CACHE_DIR"
export DIAR_PORT="$PORT"
export DIAR_GPU_DEVICE_ID="$GPU_DEVICE_ID"
DIAR_UID="$(id -u)"; export DIAR_UID
DIAR_GID="$(id -g)"; export DIAR_GID

compose() { "${COMPOSE[@]}" "${COMPOSE_FILES[@]}" "$@"; }

# ── simple actions ───────────────────────────────────────────────────────────────────────
if $DO_STOP; then
  step "Stopping"
  compose down
  exit 0
fi
if $DO_LOGS; then
  compose logs -f diar-native
  exit 0
fi

image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

# Refuse an image built for another architecture. Docker does NOT reliably refuse this for
# itself: every tag this script names is single-platform — `:latest` and `:$DIAR_VERSION` are
# amd64 CUDA images, and `-cpu-arm64` / `-provision-arm64` are arm64 BY NAME — so there is no
# manifest list to select from, and on Docker Desktop a mismatched image is emulated rather
# than rejected. That turns "you pulled the wrong tag" into "diarization is inexplicably slow",
# which is a much worse bug to be handed. Checked after every pull, including one named via
# DIAR_IMAGE.
#
# This stays correct if a tag DOES become a manifest list. From the first release published by
# scripts/release.sh, `:<ver>-cpu` and `:<ver>-provision` are multi-arch (issue #20)
# — `docker pull` then selects the matching architecture and the check below simply passes.
# The explicit `-arm64` aliases keep being published, which is why the selection above still
# names them: an explicit tag turns a mistake into this error message instead of a slow run.
assert_image_arch() {
  local ref="$1" got
  got="$(docker image inspect -f '{{.Architecture}}' "$ref" 2>/dev/null || true)"
  [[ -n "$got" ]] || return 0
  [[ "$got" == "$ARCH" ]] && return 0
  die "image $ref is linux/$got, but this machine is linux/$ARCH.
       These tags are single-platform, so the right one has to be named explicitly:
         linux/amd64 + NVIDIA GPU   $REGISTRY:$DIAR_VERSION
         linux/amd64, CPU           $REGISTRY:$DIAR_VERSION-cpu
         linux/arm64                $REGISTRY:$DIAR_VERSION-cpu-arm64
       Unset DIAR_IMAGE in .env to let this script pick, or build from source: --build"
}

pull_image() {
  local ref="$1"
  if image_exists "$ref" && ! $REBUILD; then
    note "Image $ref already present (use --rebuild to re-pull)."
  else
    step "Pulling $ref"
    docker pull "$ref" >/dev/null || die "could not pull $ref.
       Check your network, or build from this checkout instead: ./start.sh --build"
    note "Pulled."
  fi
  assert_image_arch "$ref"
}

build_server_image() {
  if image_exists "$SERVER_IMAGE" && ! $REBUILD; then
    note "Image $SERVER_IMAGE already present (use --rebuild to force)."
    return
  fi
  step "Building $SERVER_IMAGE from $DOCKERFILE"
  note "This compiles the Rust workspace and takes several minutes the first time."
  $USE_GPU && note "GPU detected, so this is the CUDA image (~3 GB). Use --cpu for the ~200 MB one."
  docker build -f "$DOCKERFILE" -t "$SERVER_IMAGE" "$REPO_DIR"
}

# Pull by default, build when asked — or when the published provisioning tag does not exist.
ensure_server_image() {
  if $DO_BUILD; then
    build_server_image
  else
    pull_image "$SERVER_IMAGE"
  fi
}

build_provision_image() {
  step "Building $PROVISION_IMAGE on $PROVISION_BASE"
  note "Adds a pinned CPU-only torch + pyannote.audio environment (~2 GB) on top of the"
  note "server image. NOTHING is compiled here — it is a pip install, not a Rust build."
  note "It is only needed for the one-time export and can be deleted afterwards."
  docker build -f docker/Dockerfile.provision --build-arg BASE="$PROVISION_BASE" \
    -t "$PROVISION_IMAGE" "$REPO_DIR"
}

ensure_provision_image() {
  if image_exists "$PROVISION_IMAGE" && ! $REBUILD; then
    note "Image $PROVISION_IMAGE already present."
    return
  fi
  if $DO_BUILD; then
    build_provision_image
    return
  fi
  # The serving images carry no Python at all — that is why the CPU one is 195 MB — so the
  # export needs this separate image. Try the published tag first; fall back to building it
  # here, which costs a pip install and no compiler, and always works from a checkout.
  step "Pulling $PROVISION_IMAGE"
  if docker pull "$PROVISION_IMAGE" >/dev/null 2>&1; then
    note "Pulled."
    assert_image_arch "$PROVISION_IMAGE"
  else
    note "Not available from the registry — building it locally instead."
    pull_image "$PROVISION_BASE"
    build_provision_image
  fi
}

# ── --cli: no server at all ──────────────────────────────────────────────────────────────
if [[ -n "$CLI_FILE" ]]; then
  [[ -f "$CLI_FILE" ]] || die "no such file: $CLI_FILE"
  [[ -f "$MARKER" ]] || die "the models are not provisioned yet ($MARKER is missing).
       Run ./start.sh once to export them, then retry --cli."

  CLI_ABS="$(readlink -f "$CLI_FILE")"
  CLI_DIR="$(dirname "$CLI_ABS")"
  CLI_BASE="$(basename "$CLI_ABS")"
  OUT_DIR="$REPO_DIR/cli-out"
  mkdir -p "$OUT_DIR"

  step "Building $CLI_IMAGE"
  note "This one is ALWAYS built: diar-cli is not among the published images — only the"
  note "server is. It compiles the workspace, so expect several minutes the first time."
  docker build -f "$DOCKERFILE" --target cli -t "$CLI_IMAGE" "$REPO_DIR"

  step "Diarizing $CLI_BASE (mode: $CLI_MODE)"
  gpu_args=()
  $USE_GPU && gpu_args=(--gpus "device=$GPU_DEVICE_ID")
  # diar-cli's own default is `info`, which buries the one line of actual output under
  # hundreds of ONNX Runtime provider-registration records. Overridable.
  cli_rust_log="$(env_get RUST_LOG "warn,ort::logging=error")"
  # Runs as YOU so the RTTM in ./cli-out is yours, not uid 10001's.
  cli_rc=0
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    "${gpu_args[@]}" \
    -e RUST_LOG="$cli_rust_log" \
    -v "$MODELS_DIR_ABS":/models:ro \
    -v "$CLI_DIR":/in:ro \
    -v "$OUT_DIR":/out \
    "$CLI_IMAGE" \
    --models-dir /models --out-dir /out --mode "$CLI_MODE" "/in/$CLI_BASE" || cli_rc=$?

  # diar-cli intermittently faults while tearing down (`corrupted double-linked list`,
  # SIGSEGV -> 139) AFTER it has printed its result line and written the RTTM. Measured on a
  # clean build of the committed tree: 2 of 3 runs at the default log level, 0 of 2 at
  # RUST_LOG=error, so it tracks shutdown work rather than anything about the diarization.
  # The output is complete and correct when it happens. Report it plainly instead of either
  # swallowing a segfault or failing a run whose results are sitting right there.
  if [[ $cli_rc -ne 0 ]]; then
    if compgen -G "$OUT_DIR/*.rttm" >/dev/null; then
      warn "diar-cli exited $cli_rc while shutting down, AFTER writing its output.
       This is a known intermittent teardown fault in diar-cli, not a diarization failure —
       the RTTM below is complete. Re-run with RUST_LOG=error in .env to make it rarer."
    else
      die "diar-cli failed with exit $cli_rc and produced no output. See the messages above."
    fi
  fi

  step "Done"
  note "RTTM written to: $OUT_DIR"
  ls -1 "$OUT_DIR" | sed 's/^/    /'
  note ""
  note "Add --json to diar-cli for segments/centroids as JSON:"
  note "  docker run --rm --user \$(id -u):\$(id -g) -v $MODELS_DIR_ABS:/models:ro \\"
  note "    -v $CLI_DIR:/in:ro -v $OUT_DIR:/out $CLI_IMAGE \\"
  note "    --models-dir /models --out-dir /out --mode $CLI_MODE --json /in/$CLI_BASE"
  exit 0
fi

# ── token check ──────────────────────────────────────────────────────────────────────────
# Runs the real `check-token` subcommand in the real image: two HTTPS calls, ~200 ms, no
# download. Its own message names the gate URL and is more actionable than anything this
# script could invent, so it is shown verbatim.
run_check_token() {
  step "Checking your HuggingFace token and the repo gate"
  local out rc=0
  out="$(docker run --rm --env-file "$ENV_FILE" "$SERVER_IMAGE" check-token 2>&1)" || rc=$?
  printf '%s\n' "$out" | sed 's/^/    /'
  if [[ "$rc" -ne 0 ]]; then
    case "$rc" in
      5) die "token check failed (exit 5). The message above is the actionable one.
       Most often this means one of:
         - the token is wrong, expired, or not a READ token
         - you have not accepted the terms at
           https://huggingface.co/pyannote/speaker-diarization-community-1
           while signed in as THAT token's account
       Fix it, then re-run: ./start.sh" ;;
      *) die "token check failed with exit $rc. See the message above." ;;
    esac
  fi
  note "${G}Token OK.${N}"
}

if $ONLY_CHECK_TOKEN; then
  ensure_server_image
  have_token || prompt_for_token
  run_check_token
  exit 0
fi

# ── get the image ────────────────────────────────────────────────────────────────────────
ensure_server_image

# ── provision (only if needed) ───────────────────────────────────────────────────────────
# The marker is the cheap host-side signal. Its absence means "definitely not provisioned";
# its presence is confirmed properly by the server's own startup gate and /readyz below,
# which is the authoritative check. This ordering is what keeps a re-run from needing a
# token, a network connection, or the 2 GB provisioning image.
if [[ -f "$MARKER" ]] && ! $FORCE_PROVISION; then
  step "Models already provisioned"
  note "Found $MARKER — skipping the export."
  note "Re-run with --provision to force a fresh one."
else
  if $FORCE_PROVISION && [[ -f "$MARKER" ]]; then
    step "Re-provisioning models (--provision)"
  else
    step "Provisioning models — this is the one-time slow step"
  fi
  cat <<EOF
    About ${B}484 MB${N} will be written to $MODELS_DIR_ABS, in roughly ${B}3 minutes${N}.
    Only ~32 MB of that is downloaded; the rest is converted locally on your machine.

    Why you have to do this: the models are derivatives of the gated
    pyannote/speaker-diarization-community-1 weights and cannot legally be shipped
    pre-built. There is no .onnx published anywhere — the conversion is mandatory.
EOF
  have_token || prompt_for_token
  ensure_provision_image
  run_check_token

  step "Exporting (writing as uid $(id -u):$(id -g), so the files end up owned by you)"
  provision_args=()
  $FORCE_PROVISION && provision_args=(--force)
  # Capture the status OUT of band. `if ! cmd; then rc=$?` looks right and is not: `!` has
  # already inverted the status by the time the body runs, so rc is always 0 and every
  # provisioning failure reports "exit 0". The exit codes are a documented contract here
  # (README 6d), so getting them wrong throws away the actual diagnosis.
  rc=0
  compose --profile provision run --rm provision \
      provision-models --models-dir /models --set "$(env_get DIAR_MODEL_SET fast)" \
      "${provision_args[@]}" || rc=$?
  if [[ $rc -ne 0 ]]; then
    case "$rc" in
      3) die "the models were exported but failed the smoke test (exit 3). They are not usable.
       Re-run with --provision to try again; if it persists this is a bug worth reporting." ;;
      4) die "the export subprocess failed (exit 4). Scroll up for the Python traceback." ;;
      5) die "HuggingFace rejected the token or the gate (exit 5). See the message above.
       Accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1" ;;
      7) die "the models directory is not writable (exit 7): $MODELS_DIR_ABS
       The export runs as uid $(id -u):$(id -g). Fix the ownership with:
           sudo chown -R $(id -u):$(id -g) '$MODELS_DIR_ABS'" ;;
      *) die "provisioning failed with exit $rc. See the output above." ;;
    esac
  fi
  note "${G}Models provisioned.${N}"
fi

# ── serve ────────────────────────────────────────────────────────────────────────────────
step "Starting the sidecar"
$USE_GPU && note "GPU: on (device $GPU_DEVICE_ID), devices=$DIAR_DEVICES" || note "GPU: off ($gpu_reason), devices=$DIAR_DEVICES"
note "Container user: $IMAGE_UID:$IMAGE_GID (non-root), models mounted read-only."
compose up -d diar-native

# ── wait for readiness ───────────────────────────────────────────────────────────────────
# /readyz, not /healthz. /healthz is 200 the moment the process is listening, even with no
# models at all, so waiting on it would declare success before the engine has loaded.
step "Waiting for GET /readyz"
BASE_URL="http://localhost:$PORT"
deadline=$(( $(date +%s) + 300 ))
ready=false
while [[ "$(date +%s)" -lt "$deadline" ]]; do
  if ! compose ps --status running --services 2>/dev/null | grep -q '^diar-native$'; then
    printf '\n'
    compose logs --tail 40 diar-native || true
    die "the container exited. Its logs are above."
  fi
  code="$(curl -s -o /dev/null -w '%{http_code}' "$BASE_URL/readyz" 2>/dev/null || true)"
  if [[ "$code" = "200" ]]; then ready=true; printf '\n'; break; fi
  printf '.'
  sleep 2
done

if ! $ready; then
  printf '\n'
  warn "the server did not become ready within 300s."
  say ""
  say "  /healthz says (this endpoint is 200 even when the models are not usable):"
  curl -s "$BASE_URL/healthz" | sed 's/^/    /' || true
  say ""
  say "  Look at models_state and models_reason above — they carry the remediation."
  say "  Full logs:  ./start.sh --logs"
  exit 1
fi

# ── done ─────────────────────────────────────────────────────────────────────────────────
step "Ready"
health="$(curl -s "$BASE_URL/healthz" || true)"
printf '    %s\n' "$health"

cat <<EOF

${B}Diarize a file${N}

  /diarize takes a PATH inside the container, not an upload. Drop the audio into
  ${B}$AUDIO_DIR${N} (mounted read-only at /audio) and then:

    cp /path/to/meeting.wav $AUDIO_DIR/

    curl -s -X POST $BASE_URL/diarize \\
      -H 'content-type: application/json' \\
      -d '{"wav_path": "/audio/meeting.wav", "gender": true}'

  wav/flac/mp3/m4a/ogg all work. The reply has:
    segments[]            {start, end, speaker} — may overlap
    exclusive_segments[]  the same, with overlaps resolved; use these for transcripts
    num_speakers          how many speakers were found
    rttm                  standard RTTM as one string
    speaker_gender        only when "gender": true was sent

${B}Other things${N}
  ./start.sh --logs        follow the logs
  ./start.sh --stop        stop it
  ./start.sh --cli FILE    diarize one file with no server running
  curl $BASE_URL/readyz    200 only when the models are verified

EOF
