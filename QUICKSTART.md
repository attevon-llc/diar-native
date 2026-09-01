# Quickstart

Speaker diarization — "who spoke when" — as a small self-hosted HTTP service, or as a
one-shot command-line tool. Rust + ONNX Runtime, no Python at serving time.

**Deploying it needs no clone of this repository.** Two files and published images are the
whole deployment; the repo is for *contributing to* diar-native, not for *running* it.

```bash
mkdir -p diar-native/audio && cd diar-native
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.0/docker-compose.prod.yml -o docker-compose.yml
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.0/.env.example -o .env
$EDITOR .env      # set HUGGINGFACE_TOKEN=hf_...
docker compose up
```

That one command exports the models with your token (~484 MB, ~3 minutes, once) and then
serves on port 8701. Re-running it is a fast no-op that needs no token and no network.

### Pick the right image

Every published tag is **single-platform** — there is no multi-arch manifest, so nothing will
stop you pulling an amd64 image onto an arm64 machine. The default is the amd64 CPU image.

| your machine | add to `.env` | notes |
|---|---|---|
| linux/amd64, CPU | *(nothing)* | 195 MB, the default |
| linux/amd64 + NVIDIA GPU | `DIAR_IMAGE=davidamacey/diar-native:0.3.0` | 3.04 GB, serves `cuda` **and** `cpu` per request. Also add the GPU overlay (below) |
| linux/arm64 · Apple Silicon | `DIAR_IMAGE=davidamacey/diar-native:0.3.0-cpu-arm64`<br>`DIAR_PROVISION_IMAGE=davidamacey/diar-native:0.3.0-provision-arm64` | 223 MB. Runs on **CPU cores** — not the Apple GPU, not the Neural Engine |

For the GPU, fetch the overlay too and name both files. The reservation lives in a separate
file because compose cannot make a device request conditional — present and unsatisfiable, it
is a hard startup failure on every host without an NVIDIA runtime:

```bash
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.0/docker-compose.gpu.yml -o docker-compose.gpu.yml
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

On Apple Silicon, be clear-eyed about what Docker gives you: the correct instruction set and
nothing more. Docker on macOS has no Metal access at any image architecture. A native
**CoreML** build that does use the Apple GPU exists and works (M2 Max, 93 vs 92 segments
against the CUDA reference — `validation/RESULTS.md` §7.31), but it is **not published** and
must be compiled from source on the machine.

---

## What you need

- **Docker** with the Compose v2 plugin (`docker compose version` works).
- **A HuggingFace account and a read token.** Free. See [The token](#the-token) below.
- **~700 MB of disk** on a CPU host (195 MB image + 484 MB of models), or ~3.5 GB for the CUDA
  image. Provisioning temporarily needs a further ~2 GB for the export environment, which can
  be deleted afterwards.
- **A GPU is optional.** Without one everything runs correctly on the CPU; only speed changes.

Nothing else — no Python environment, no CUDA toolkit on the host, no model downloads to
manage by hand, and no clone.

---

## The token

The models are derivatives of
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1),
which is a **gated** repository. Nobody is allowed to redistribute the converted weights, so
this project cannot ship them to you and does not try. Instead every operator exports their
own copy locally, once.

Two one-time steps, both free, both instant:

1. Create a **read** token at <https://huggingface.co/settings/tokens>.
2. Signed in as that same account, accept the terms at
   <https://huggingface.co/pyannote/speaker-diarization-community-1>. It is a short
   email-capture form and it is **auto-approved** — there is no waiting list, no human
   review, and the pipeline is CC-BY-4.0.

`start.sh` prompts for the token (hidden input), stores it in `.env` with mode 600, and never
puts it on a command line. Check it any time without downloading anything:

```bash
./start.sh --check-token
```

**What actually gets downloaded:** about **32 MB** from HuggingFace. The export then converts
that into roughly **484 MB** of ONNX and PLDA files on your machine, in about **3 minutes**.
There is no `.onnx` published anywhere for these models — upstream ships PyTorch weights — so
the conversion is mandatory, not an optimization. The token is used only for that one
download; serving never touches the network.

---

## Path 1 — the server

Either `docker compose up` with the two fetched files above, or — from a clone — the wrapper
script, which adds platform detection, a hidden token prompt and a wait on `/readyz`:

```bash
./start.sh                    # pull, provision, serve, wait for ready
./start.sh --build            # compile from this checkout instead (contributors)
```

On success it prints the health JSON and a ready-to-paste request. To diarize a file, drop it
into `./audio/` (bind-mounted read-only at `/audio`) and POST its **container path** — the API
takes a path, not a file upload:

```bash
cp ~/meeting.wav ./audio/

curl -s -X POST http://localhost:8701/diarize \
  -H 'content-type: application/json' \
  -d '{"wav_path": "/audio/meeting.wav", "gender": true}'
```

`wav` despite the name is not a restriction: wav, flac, mp3, m4a and ogg all decode.

Housekeeping:

```bash
./start.sh --logs      # follow logs
./start.sh --stop      # stop it
./start.sh --cpu       # force the small CPU image even on a GPU box
./start.sh --provision # force a fresh model export
```

Re-running `./start.sh` is safe and fast. With the models already exported it needs no token
and no network — it just starts the server.

### Reading the response

```jsonc
{
  "segments": [
    {"start": 0.51, "end": 4.28, "speaker": "SPEAKER_00"}
  ],
  "exclusive_segments": [
    {"start": 0.51, "end": 4.28, "speaker": "SPEAKER_00"}
  ],
  "num_speakers": 2,
  "rttm": "SPEAKER meeting 1 0.510 3.770 <NA> <NA> SPEAKER_00 <NA> <NA>\n...",
  "speaker_gender": {
    "SPEAKER_00": {"label": "male", "confidence": 0.94, "windows": 12}
  }
}
```

- **`segments`** — raw turns. These **can overlap**, because two people genuinely can talk at
  once. Good for diarization metrics, awkward for anything that needs a single speaker per
  moment.
- **`exclusive_segments`** — the same turns with overlaps resolved so no two segments cover
  the same instant. **This is the one you want** if you are attaching speaker labels to a
  transcript.
- **`num_speakers`** — how many distinct speakers were found. It is discovered, not
  configured; you do not have to know it in advance.
- **`rttm`** — the whole result as a standard RTTM string, ready to write to a `.rttm` file
  and feed to any scoring tool.
- **`speaker_gender`** — present **only** if you sent `"gender": true`. `confidence` is
  0-1 and `windows` is how many audio windows voted, which is your signal for how much to
  trust a verdict: a speaker with 2 windows is a guess, one with 40 is not.

Speaker labels (`SPEAKER_00`, …) are arbitrary and stable only within a single response. The
same person in two different files will not get the same label.

Two more endpoints: `GET /healthz` is 200 whenever the process is alive, in every state, and
carries `models_state` and `models_reason` explaining exactly what is wrong when something is.
`GET /readyz` is 200 **only** when the models are verified and it is the one to gate on.

---

## Path 2 — the CLI, no server

For a one-off, or a batch, with nothing left running:

```bash
./start.sh --cli ~/meeting.wav
```

Writes `meeting_run0.rttm` into `./cli-out/` and prints one line of JSON with the timing and
speaker count. It reuses the models you already exported, so run `./start.sh` once first.

The image shares every layer with the server image and adds only the ~37 MB binary. Under the
hood it is:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/models":/models:ro -v ~/audio:/in:ro -v "$PWD/cli-out":/out \
  diar-native-cli:local-cpu \
  --models-dir /models --out-dir /out --mode cpu --json /in/meeting.wav
```

`--json` additionally writes segments, centroids and exclusive segments next to the RTTM.

---

## The container user

Both images run as **non-root, uid/gid `10001:10001`**. Two consequences worth knowing:

- **Serving** mounts the models **read-only** and writes nothing at all, so it works against a
  models directory owned by anyone, as long as the files are world-readable — which is what a
  normal umask produces.
- **Provisioning** has to write ~484 MB, and a container user cannot write a host directory it
  does not own. So the export runs as **your** uid instead (`--user "$(id -u):$(id -g)"`,
  which `start.sh` and the `provision` compose service both set). The files land owned by you.

**You should never need a `chown`.** If you somehow do — for example the directory was created
by an older root container — this is the exact command:

```bash
sudo chown -R "$(id -u):$(id -g)" ./models
```

Do *not* chown to `10001`. Owning the directory yourself is what keeps re-provisioning working.

---

## Building it from source (contributors)

Only needed when you are changing diar-native, or running a commit that was never published.
`./start.sh --build` does all of this; by hand it is:

```bash
cp .env.example .env
$EDITOR .env                                    # set HUGGINGFACE_TOKEN

# CPU (~195 MB) — or Dockerfile.server for CUDA
docker build -f docker/Dockerfile.server-cpu -t diar-native:local-cpu .
docker build -f docker/Dockerfile.provision \
  --build-arg BASE=diar-native:local-cpu -t diar-native-provision:local-cpu .

mkdir -p models audio .hf-cache
DIAR_UID=$(id -u) DIAR_GID=$(id -g) \
  docker compose --profile provision run --rm provision

docker compose up -d
curl -s localhost:8701/readyz
```

Note `docker-compose.yml` — the source-build file, where `provision` sits behind a profile and
the models are a bind mount you own. `docker-compose.prod.yml` is its published-image sibling:
no `build:` key, provisioning wired into a plain `up`, named volumes by default.

Add the GPU overlay to use a GPU, and point `DIAR_IMAGE` at a CUDA build:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

The provisioning image is the only one that must be built rather than pulled even on the
published path: the serving images carry **no Python at all** (which is why the CPU one is
195 MB and not ~2 GB), and the export needs torch + pyannote.audio. Asking a serving image to
provision fails immediately with exit 6, `No python interpreter at 'python3'`. Building it is a
pip install on top of the image you already have — no Rust, no compiler:

```bash
docker build -f docker/Dockerfile.provision \
  --build-arg BASE=davidamacey/diar-native:0.3.0-cpu \
  -t davidamacey/diar-native:0.3.0-provision .
```

Every knob is documented in `.env.example`; `README.md` §6e is the authoritative table of the
variables the binary itself reads.

---

## The four things that actually go wrong

### 1. The token is wrong → `HTTP 401`, exit 5

```
error: Your HuggingFace token was rejected (HTTP 401).
```

It is not a read token, it was revoked, or it was pasted with a stray character. Make a fresh
read token and re-run `./start.sh --check-token`. Note the token belongs to *you* — this
project has no token of its own to fall back on.

### 2. The terms were never accepted → `HTTP 403`, exit 5

This one is sneaky: the token is perfectly valid, so it *feels* like it should work. The gate
is per-account, so accepting the terms while signed in as a different HuggingFace account than
the one that issued the token fails in exactly the same way.

Fix: visit
<https://huggingface.co/pyannote/speaker-diarization-community-1> **signed in as the token's
own account**, accept, then re-run. Auto-approved, no wait.

### 3. `models directory is not writable` → exit 7

```
error: the models directory is not writable (exit 7)
```

You ran the export against the read-only mount. The serving service mounts `/models:ro` on
purpose; only the `provision` service mounts it read-write. Use
`docker compose --profile provision run --rm provision`, or `./start.sh --provision`, rather
than `docker compose exec diar-native provision-models`.

The check happens **before** the export, not after 484 MB of work.

If the directory is genuinely owned by someone else, see
[The container user](#the-container-user).

### 4. No GPU, or a GPU Docker cannot reach

`start.sh` requires **both** a working `nvidia-smi` **and** the NVIDIA container runtime
registered with Docker, and falls back to the CPU image if either is missing. That second
check is the one people forget: a host can have a perfectly good driver and still be unable to
pass the GPU into a container, in which case the container simply fails to start.

If you expected a GPU and did not get one, `start.sh` prints the reason. Install
`nvidia-container-toolkit`, restart Docker, and re-run with `--gpu` to make the failure loud
instead of falling back.

The CPU path is fully supported and produces the same output — only slower. Forcing
`DIAR_DEVICES=cuda` against the CPU image is fatal at startup, deliberately: a diarizer that
quietly falls back to the CPU is a performance regression nobody notices.

---

## Where to go next

- `.env.example` — every setting, with its default and its sharp edges.
- `README.md` — architecture, benchmarks, and §6e, the authoritative environment table.
- `docs/INSTALL_NATIVE.md` — provisioning in depth: exit codes, what the smoke test checks,
  and what the provenance marker does and does not claim.
