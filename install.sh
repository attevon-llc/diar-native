#!/usr/bin/env bash
# diar-native installer — takes a machine from nothing to a running, verified diarization
# service in one command. No clone, no Rust toolchain, no manual steps.
#
#   curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/install.sh -o install.sh
#   bash install.sh
#
# Downloading first and running second, rather than `curl | bash`, is deliberate: you should be
# able to read a script before it touches your machine. It is short enough to read.
#
# WHAT IT DOES
#   1. Works out your platform (GPU? amd64 or arm64?) and picks the right published image.
#   2. Fetches docker-compose.prod.yml and .env.example into ./diar-native.
#   3. Asks for your HuggingFace token, unless one is already available.
#   4. Downloads and exports the models with YOUR token (~484 MB, a few minutes, once).
#   5. Starts the service and waits until it reports ready.
#   6. Tells you how to diarize a file.
#
# Re-running is safe and fast: provisioning is idempotent, so a second run skips it entirely
# and needs no token and no network.
#
# OPTIONS
#   --dir <path>       where to install                     (default: ./diar-native)
#   --port <n>         host port for the API                (default: 8701)
#   --cpu              force the CPU image even if a GPU is present
#   --gpu              require GPU; fail rather than falling back to CPU
#   --token <t>        HuggingFace token (else $HUGGINGFACE_TOKEN / $HF_TOKEN, else prompt)
#   --models <path>    host directory for the models        (default: a docker volume)
#   --diarize <file>   after starting, diarize this file and print the result
#   --no-start         set everything up but do not start the service
#   -h, --help         this text
set -euo pipefail

VERSION="0.3.1"
RAW="https://raw.githubusercontent.com/attevon-llc/diar-native/v${VERSION}"

DIR="./diar-native"; PORT=8701; FORCE=""; TOKEN=""; MODELS=""; DIARIZE=""; START=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) DIR="${2:?}"; shift ;;
    --port) PORT="${2:?}"; shift ;;
    --cpu) FORCE=cpu ;;
    --gpu) FORCE=gpu ;;
    --token) TOKEN="${2:?}"; shift ;;
    --models) MODELS="${2:?}"; shift ;;
    --diarize) DIARIZE="${2:?}"; shift ;;
    --no-start) START=0 ;;
    -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
  shift
done

die() { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --diarize is resolved to an ABSOLUTE path here, while we are still in the directory the user
# ran us from. Everything below runs after `cd "$DIR"`, so a relative path like `clip30.wav`
# silently stops resolving — which is exactly what happened the first time this script was run
# end to end from a clean directory.
if [[ -n "$DIARIZE" ]]; then
  [[ -f "$DIARIZE" ]] || { printf '\n\033[31merror:\033[0m no such file: %s\n' "$DIARIZE" >&2; exit 1; }
  DIARIZE="$(cd "$(dirname "$DIARIZE")" && pwd)/$(basename "$DIARIZE")"
fi
say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
ok()  { printf '  \033[32mok\033[0m  %s\n' "$*"; }

command -v docker >/dev/null || die "docker is not installed. See https://docs.docker.com/get-docker/"
docker compose version >/dev/null 2>&1 \
  || die "this needs Docker Compose v2 ('docker compose', not 'docker-compose')"
docker info >/dev/null 2>&1 || die "the docker daemon is not reachable — is it running?"
command -v curl >/dev/null || die "curl is not installed"

# ── platform ────────────────────────────────────────────────────────────────────────────
# Every published tag is single-platform, so picking the wrong one fails at run time with a
# confusing exec-format error rather than at pull time. Choose it here instead.
say "Working out what to run"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH=amd64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) die "unsupported architecture '$ARCH' — diar-native publishes amd64 and arm64 images" ;;
esac

HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  # A driver is not enough: the container runtime has to be able to hand the GPU through.
  if docker run --rm --gpus all ubuntu:24.04 true >/dev/null 2>&1; then HAS_GPU=1; fi
fi

[[ "$FORCE" == gpu && ( "$HAS_GPU" == 0 || "$ARCH" != amd64 ) ]] && \
  die "--gpu was requested but no usable NVIDIA GPU was found on an amd64 host.
A driver alone is not enough — the NVIDIA container runtime must also be installed."
[[ "$FORCE" == cpu ]] && HAS_GPU=0

if [[ "$ARCH" == amd64 && "$HAS_GPU" == 1 ]]; then
  IMAGE="davidamacey/diar-native:${VERSION}";              PROV="davidamacey/diar-native:${VERSION}-provision";       MODE="amd64 + NVIDIA GPU"
elif [[ "$ARCH" == amd64 ]]; then
  IMAGE="davidamacey/diar-native:${VERSION}-cpu";          PROV="davidamacey/diar-native:${VERSION}-provision";       MODE="amd64, CPU only"
else
  IMAGE="davidamacey/diar-native:${VERSION}-cpu-arm64";    PROV="davidamacey/diar-native:${VERSION}-provision-arm64"; MODE="arm64, CPU only"
  # An arm64 tag invites the assumption that Apple Silicon acceleration is in play. It is not.
  [[ "$(uname -s)" == Darwin ]] && \
    printf '  \033[33mnote\033[0m Docker on macOS has no Metal access, so this runs on CPU cores,\n       not the GPU or Neural Engine.\n'
fi
ok "$MODE"
ok "serving image     $IMAGE"
ok "provisioning      $PROV"

# ── files ───────────────────────────────────────────────────────────────────────────────
say "Setting up $DIR"
mkdir -p "$DIR/audio" && cd "$DIR"
# Fetched explicitly rather than in a loop with name-mangling: the compose file is renamed on
# the way in, and getting that wrong silently leaves you with a directory that looks set up.
curl -fsSL "${RAW}/docker-compose.prod.yml" -o docker-compose.yml \
  || die "could not fetch docker-compose.prod.yml from ${RAW}
Is the v${VERSION} tag published? Check https://github.com/attevon-llc/diar-native/tags"
curl -fsSL "${RAW}/.env.example" -o .env.example \
  || die "could not fetch .env.example from ${RAW}"
ok "docker-compose.yml + .env.example"

# ── token ───────────────────────────────────────────────────────────────────────────────
# Only needed for the first provisioning run. A valid marker makes every later start a no-op.
if [[ -f .env ]] && grep -qE '^HUGGINGFACE_TOKEN=.+' .env; then
  ok ".env already has a token"
else
  [[ -n "$TOKEN" ]] || TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
  if [[ -z "$TOKEN" ]]; then
    cat <<'EOF'

  diar-native downloads pyannote's models with YOUR HuggingFace token. Nothing is
  redistributed — the weights are CC-BY-4.0 and the gate is auto-approved.

    1. Create a READ token:  https://huggingface.co/settings/tokens
    2. Accept the terms:     https://huggingface.co/pyannote/speaker-diarization-community-1

EOF
    [[ -t 0 ]] || die "no token given and this is not an interactive terminal.
Pass --token, or set HUGGINGFACE_TOKEN."
    read -rsp "  HuggingFace token: " TOKEN; echo
    [[ -n "$TOKEN" ]] || die "no token entered"
  fi
  cp -f .env.example .env
  # Written with awk rather than sed -i so a token containing / or & cannot corrupt the file.
  awk -v t="$TOKEN" '/^HUGGINGFACE_TOKEN=/{print "HUGGINGFACE_TOKEN=" t; next} {print}' .env > .env.tmp
  mv -f .env.tmp .env && chmod 600 .env
  ok ".env written (mode 600)"
fi

{ echo "DIAR_IMAGE=${IMAGE}"; echo "DIAR_PROVISION_IMAGE=${PROV}"; echo "DIAR_PORT=${PORT}"
  [[ -n "$MODELS" ]] && echo "DIAR_MODELS_HOST_DIR=${MODELS}"
} >> .env
ok "pinned to ${VERSION} for this install"

if [[ $START -eq 0 ]]; then
  say "Set up, not started (--no-start). Run: cd $DIR && docker compose up -d"
  exit 0
fi

# ── run ─────────────────────────────────────────────────────────────────────────────────
say "Starting (first run downloads ~484 MB of models and exports them — a few minutes)"
COMPOSE=(docker compose)
[[ "$HAS_GPU" == 1 ]] && { curl -fsSL "${RAW}/docker-compose.gpu.yml" -o docker-compose.gpu.yml 2>/dev/null \
  && COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.gpu.yml); }
"${COMPOSE[@]}" up -d

say "Waiting for the service to report ready"
for i in $(seq 1 180); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/readyz" 2>/dev/null || true)"
  [[ "$code" == 200 ]] && { ok "ready"; break; }
  [[ $i -eq 180 ]] && die "not ready after 15 minutes. Logs: cd $DIR && docker compose logs"
  sleep 5
done

STATE="$(curl -s "http://localhost:${PORT}/healthz" 2>/dev/null \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["models_state"], "/", ",".join(d["devices"]))' 2>/dev/null || echo '?')"
ok "models: $STATE"

if [[ -n "$DIARIZE" ]]; then
  cp -f "$DIARIZE" audio/
  say "Diarizing $(basename "$DIARIZE")"
  curl -s "http://localhost:${PORT}/diarize" -H 'Content-Type: application/json' \
    -d "{\"audio_path\":\"/audio/$(basename "$DIARIZE")\"}" \
    | python3 -m json.tool 2>/dev/null || echo "(request failed — see: cd $DIR && docker compose logs)"
fi

cat <<EOF

$(printf '\033[1mdiar-native %s is running on port %s\033[0m' "$VERSION" "$PORT")

  Diarize a file — put it in $DIR/audio/ first:

    curl -s localhost:${PORT}/diarize -H 'Content-Type: application/json' \\
      -d '{"audio_path":"/audio/your-file.wav"}'

  Any format symphonia reads works: wav, flac, mp3, m4a, ogg, aac, mp4.

  Health:  curl -s localhost:${PORT}/healthz     (200 whenever it is serving)
  Ready:   curl -s localhost:${PORT}/readyz      (200 only when models are verified)
  Logs:    cd $DIR && docker compose logs -f
  Stop:    cd $DIR && docker compose down
EOF
