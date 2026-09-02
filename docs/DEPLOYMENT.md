# Deployment

Running diar-native in anger: compose, published images, the platform matrix, ports and volumes,
and exit codes. For the one-command install, start at the [README](../README.md). For the
variables you can set, see [CONFIGURATION.md](CONFIGURATION.md).

- [You do not need this repository](#you-do-not-need-this-repository)
- [Platform matrix](#platform-matrix)
- [Compose files](#compose-files)
- [Ports, volumes and the container user](#ports-volumes-and-the-container-user)
- [Published images](#published-images)
- [Exit codes](#exit-codes)
- [Platform note: the fp16 gender model on linux/arm64](#the-fp16-gender-model-on-linuxarm64)
- [The base image is pinned to ubuntu 24.04](#the-base-image-is-pinned-to-ubuntu-2404)

---

## You do not need this repository

The deployment is two files — a compose file and a `.env` — plus images published on Docker
Hub. Nothing is cloned, built or compiled. `install.sh` does all of this for you; by hand it is:

```bash
mkdir -p diar-native/audio && cd diar-native
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/docker-compose.prod.yml -o docker-compose.yml
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/.env.example -o .env
$EDITOR .env      # set HUGGINGFACE_TOKEN=hf_...   (the only thing you must supply)
docker compose up
```

One thing cannot be shipped for you: the models are derivatives of the **gated**
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
weights, which nobody may redistribute. So you need a free HuggingFace read token, once. That
single `up` exports the models and then serves on port 8701. **Every later `docker compose up`
skips the export** — it finds the provenance marker and starts serving in seconds, with no token
and no network access at all. See [PROVISIONING.md](PROVISIONING.md).

## Platform matrix

| your machine | image to name in `.env` | what you get |
|---|---|---|
| **linux/amd64, CPU** | *(nothing — it is the default)* | The 195 MB CPU image. Works on any amd64 machine, GPU or not, and needs no NVIDIA runtime. |
| **linux/amd64 + NVIDIA GPU** | `DIAR_IMAGE=davidamacey/diar-native:0.3.1` | The CUDA image, which serves **both `cuda` and `cpu`**, chosen per request. Also add the GPU overlay (below). |
| **linux/arm64 · Apple Silicon** | `DIAR_IMAGE=davidamacey/diar-native:0.3.1-cpu-arm64`<br>`DIAR_PROVISION_IMAGE=davidamacey/diar-native:0.3.1-provision-arm64` | The arm64 CPU image. Runs on **CPU cores** — not the Apple GPU, not the Neural Engine. |

### On a GPU

The CUDA image needs the NVIDIA container toolkit installed **and registered with Docker** — a
working `nvidia-smi` on the host is not sufficient by itself. The GPU is opt-in through a
separate overlay file because a compose device reservation cannot be made conditional: present
and unsatisfiable, it is a hard startup failure on every GPU-less host.

```bash
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/docker-compose.gpu.yml -o docker-compose.gpu.yml
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

The CPU path is fully supported and produces the same output — only slower. Forcing
`DIAR_DEVICES=cuda` against the CPU image is fatal at startup, deliberately: a diarizer that
quietly falls back to the CPU is a performance regression nobody notices.

### On macOS and arm64 Linux

Docker runs the **arm64 CPU image**, and it is worth being blunt about what that is: it uses your
**CPU cores**. It does **not** use the GPU and does **not** use the Neural Engine. Docker on
macOS has no Metal access at any image architecture, so an `arm64` tag buys you the right
instruction set and nothing else. It works correctly; it is simply not accelerated.

> **Do not use `:latest` or the bare `:<ver>` tag on an arm64 host.** Those are the CUDA image
> and are published for **linux/amd64 only** — permanently, because no aarch64 ONNX Runtime GPU
> build exists (issue #4, closed as a documented impossibility). Docker Desktop will **emulate**
> them rather than refuse them, so the symptom is "inexplicably slow" rather than an error you
> can act on. Name the `-arm64` tags instead.

A native **CoreML** build that does use the Apple GPU exists and works — verified on an M2 Max at
93 vs 92 segments against the CUDA reference ([RESULTS](../validation/RESULTS.md) §7.31). But it
is **not published**, and Docker cannot reach Metal, so it must be compiled from source on the
machine itself. See [ARCHITECTURE.md](ARCHITECTURE.md#apple-silicon-native-coreml).

## Compose files

| file | what it is |
|---|---|
| `docker-compose.prod.yml` | **The published-image deployment.** No `build:` key, provisioning wired into a plain `up`, named volumes by default. This is what `install.sh` fetches (renamed to `docker-compose.yml`). |
| `docker-compose.yml` | The **source-build** file: `provision` sits behind a compose profile and the models are a bind mount you own. For contributors. |
| `docker-compose.gpu.yml` | The NVIDIA device reservation, and nothing else. An overlay, for the reason above. |

## Ports, volumes and the container user

| | |
|---|---|
| **Port** | `${DIAR_PORT:-8701}` on the host → `8701` in the container (`DIAR_BIND=0.0.0.0:8701`). |
| **Models** | `${DIAR_MODELS_HOST_DIR:-diar-models}` → `/models`, mounted **`:ro` for serving** and read-write only for the `provision` service. |
| **Audio** | `${DIAR_AUDIO_HOST_DIR:-./audio}` → `/audio`, **`:ro`**. This is the one you care about: `/diarize` takes a path *inside the container*, so a file must be here before you can POST it. |
| **Healthcheck** | `curl -fsS http://localhost:8701/healthz`. After provisioning once, you may move this to `/readyz`. |

The models volume defaults to a **named volume, not a bind mount**, and that is a correctness
decision: the provisioning image ships `/models` owned by uid 10001, so a named volume comes out
owned correctly on its own. To use a host directory instead, create it yourself and pass your own
uid:

```bash
mkdir -p ./models
DIAR_MODELS_HOST_DIR=./models DIAR_UID=$(id -u) DIAR_GID=$(id -g) docker compose up
```

### The container user

Every image runs as **non-root, uid/gid `10001:10001`** (issue #7). Two consequences:

- **Serving** mounts the models read-only and writes nothing at all, so it works against a models
  directory owned by anyone, as long as the files are world-readable — which is what a normal
  umask produces.
- **Provisioning** has to write ~484 MB, and a container user cannot write a host directory it
  does not own. So the export runs as **your** uid instead (`--user "$(id -u):$(id -g)"`, which
  `install.sh`, `start.sh` and the `provision` compose service all set). The files land owned by
  you.

**You should never need a `chown`.** If you somehow do — for example the directory was created by
an older root container:

```bash
sudo chown -R "$(id -u):$(id -g)" ./models
```

Do *not* chown to `10001`. Owning the directory yourself is what keeps re-provisioning working.

## Published images

Tag shape follows **capability**, not convenience — a manifest list is only honest when every
architecture under it is the same thing (issue #20):

- **`:latest` and `:<ver>` are `linux/amd64` only, and always will be.** They are the CUDA + CPU
  superset, and there is no aarch64 ONNX Runtime GPU build in existence (issue #4). The only
  arm64 entry those tags could carry is the CPU image — one tag whose capabilities differ by
  architecture, so that `docker pull` hands an arm64 user a diarizer with no GPU support and an
  amd64 user one with it. Being explicit is better, so they stay amd64 and this section says so.
- **`:<ver>-cpu` and `:<ver>-provision` are multi-arch manifest lists** (amd64 + arm64). Both
  architectures are the same build with the same capabilities there, so `docker pull` resolving
  them for you is the truth rather than a convenient lie.
- **`:<ver>-cpu-arm64` and `:<ver>-provision-arm64` remain published** as single-platform arm64
  aliases. `install.sh`, `start.sh` and `docker-compose.prod.yml` name an exact tag per host and
  then assert the architecture of what they pulled; that path wants a tag that is arm64 *by
  name*, so a mismatch is a clean error rather than a slow emulated one. The manifest list is for
  `docker pull` doing the right thing on its own; the alias is for naming it deliberately.

**0.3.0 and 0.3.1 both predate this.** Their images were built and pushed one platform at a time,
so **every published tag so far is single-platform**: `:0.3.1-cpu` is amd64 and arm64 lives only
at `:0.3.1-cpu-arm64`. The manifest lists start at the first release published by
`scripts/release.sh`. The table below is 0.3.1 as actually published, verified by fresh pull.

| tag | platform | size | contents |
|---|---|---|---|
| `davidamacey/diar-native:0.3.1`<br>`davidamacey/diar-native:latest` | linux/amd64 | 3.04 GB | CUDA **and** CPU, selected per request |
| `davidamacey/diar-native:0.3.1-cpu` | linux/amd64 | 195 MB | CPU only — no CUDA libraries, no NVIDIA runtime needed |
| `davidamacey/diar-native:0.3.1-cpu-arm64` | linux/arm64 | 223 MB | CPU only. Runs on ARM Linux and on Apple Silicon under Docker — **on CPU cores**, not the GPU or Neural Engine |
| `davidamacey/diar-native:0.3.1-provision`<br>`davidamacey/diar-native:0.3.1-provision-arm64` | linux/amd64<br>linux/arm64 | 1.94 GB<br>1.86 GB | **Provisioning only.** The serving image plus a pinned CPU-only torch + pyannote.audio environment. Referenced by `docker-compose.prod.yml`; delete it once the models exist |

Digests, for pinning:

```
0.3.1 / latest         sha256:83a709be94d0ca06441fa10aea0680f53b03cc10eb3ce11c4eeb84478400567d
0.3.1-cpu              sha256:b00b4bb5999d0b5cc353dae27c07b17fe61b232517f250aaf3ef03536c610879
0.3.1-cpu-arm64        sha256:63e7a7275aada3da8c4840cbc5e2b4e498605a6658086355c62b84af75232d64
0.3.1-provision        sha256:99eead002b34f7dd6a18c63a11a305ec13edefce4f5e5f75ff29862f881b63d0
0.3.1-provision-arm64  sha256:58998c5e4365e32df450893de7b0b2f0011754a95bc61792bba7f6e16df51cc8
```

These are the **republished** 0.3.1 digests. 0.3.1 was built twice: once at release, and again
after the vendored speakrs pin was corrected to a real fork commit. The source is unchanged
between the two — the patch regenerated byte-identically against the new pin — and the two builds
were run side by side on one GPU against the same models directory, producing a **byte-identical
`/diarize` response**. The binaries differ by sha256 only because Rust release builds are not
bit-reproducible. If you pinned the earlier `sha256:1a8e1491…`, moving to the digest above is a
provenance improvement and not a behaviour change.

All five: **trivy 0.67.1, 0 HIGH / 0 CRITICAL**, running as **uid 10001**. Consumers that copy
the binary out of the image (see
[ARCHITECTURE.md](ARCHITECTURE.md#production-consumes-the-binary-not-the-image)) want the **amd64
CUDA** digest — that is the build the `diar-native-bin` stage extracts from.

### No Python in the serving images

None of the serving images contain Python — that is why the CPU one is 195 MB rather than ~2 GB —
so `provision-models` run against one exits **6** with `No python interpreter at 'python3'`. That
is what the separate provisioning tag is for, and why it is CPU-based even for GPU deployments:
the export does `pipeline.to(torch.device("cpu"))` and never touches an accelerator. It builds
with no compiler at all, as a pip layer on the image you already have:

```bash
docker build -f docker/Dockerfile.provision \
  --build-arg BASE=davidamacey/diar-native:0.3.1-cpu \
  -t davidamacey/diar-native:0.3.1-provision .
```

## Exit codes

Authoritative source: `crates/diar-core/src/provision/mod.rs::exit`. **Stable** — scripts and
supervisors branch on these.

| code | name | meaning | emitted by |
|---|---|---|---|
| 0 | `OK` | Success, including a no-op on an already-valid directory. | all subcommands |
| 1 | *(none)* | Serve path only: any other startup failure (bind failed, engine load failed) surfacing as a non-zero `main`. | serve |
| 2 | `USAGE` | Bad arguments — unknown `--set`/`--mode`, unresolvable `--smoke-clip`. | all |
| 3 | `SMOKE_FAILED` | Files exist but the smoke test rejected them; in `verify-models` also means recorded-hash **drift**. | provision-models, verify-models |
| 4 | `EXPORT_FAILED` | The export subprocess failed. | provision-models |
| 5 | `TOKEN_DENIED` | Token missing/invalid, or the repo terms have not been accepted. | provision-models, check-token |
| 6 | `NO_EXPORTER_ENV` | No usable python export environment — interpreter missing, or it cannot import torch / pyannote.audio / onnx. The fix is `pip install`. | provision-models |
| 7 | `NOT_WRITABLE` | The models directory is not writable. Checked up front, before a multi-hundred-MB export. | provision-models |
| 8 | `MODELS_UNUSABLE` | **Serve only:** the models directory is too broken to start against. | serve |
| 9 | `DEVICE_UNAVAILABLE` | The requested execution device is not usable here. Says nothing about the models, and — unlike a smoke failure — never marks them known-bad. | provision-models, verify-models |
| 10 | `UNVERIFIABLE` | **`verify-models` only:** the files work, but there is no marker to verify them *against*, so nothing was compared to a recorded hash. Not `OK`, not `SMOKE_FAILED`. | verify-models |

> **6 and 8 were one code before 0.3.0.** A supervisor could not tell "install torch into the
> exporter" from "provision the models", which have nothing to do with each other — serving needs
> no python at all.

## The fp16 gender model on linux/arm64

The fp16 gender classifier does not load at all on **linux/arm64** without a workaround — which
silently disables speaker gender there while the server still answers 200. It is not a bug in the
model. The graph is plain opset-17 `ai.onnx` with no contrib domain, but it has 20 `Erf` nodes,
and one of ORT's *extended* (level-2) optimizations rewrites that GELU pattern into
`com.microsoft.Gelu`, for which no fp16 kernel exists. The optimizer synthesizes a node the very
same runtime then refuses to execute. The node named in the error **is not in the file on disk**,
which is what makes the message confusing on first read.

The obvious explanation — "x86_64 has the fp16 kernel, aarch64 does not" — is only half right,
and the wrong half is the half that matters. *Every* aarch64 ORT build checked lacks that kernel,
**including macOS arm64, where the model loads fine**. What differs is whether the fusion fires:

| platform | fp16 kernel | fuses fp16? | result |
|---|---|---|---|
| linux/amd64 | yes | yes | loads |
| **linux/arm64** | **no** | **yes** | **fails** |
| macOS arm64 | no | **no** | loads |

So it is a build-configuration divergence between two targets of the same ORT release, not an
architecture property. **This is fixed:** `crates/diar-core/src/ort_compat.rs` caps optimization
at `Level1` for that one model on aarch64 — a no-op on the platform that declines the fusion
anyway. The 15 diarization graphs keep full optimization.

The lesson generalises: **any future fp16 export needs a LOAD gate on aarch64, not just an
accuracy gate.** An accuracy gate cannot see this, because the session never opens.
`verify-models` stage 1 carries exactly that gate, and it must stay a load check, never a numeric
one.

Full analysis, the measured alternatives, and three traps around the escape hatches (the
optimizer is named `GeluFusionL2`, a wrong name is **silently ignored**, and the separator is `;`
not `,`): [`ORT_FUSION_FP16_AARCH64.md`](ORT_FUSION_FP16_AARCH64.md) and RESULTS §7.40.

## The base image is pinned to ubuntu 24.04

**Do not re-bump it.** The 26.04 bump was merged, reverted, and then scored (RESULTS §7.52): on
**linux/arm64** it is a severe accuracy regression, not the cosmetic 8 → 10 exclusive-segment
difference it looks like on the smoke fixture.

- AMI-16 **exclusive DER 18.7% → 52.4%**, full DER 13.8% → 48.7%.
- Missed detection and false alarm are **identical to the digit** — the entire +33.7 pp is
  speaker confusion. Embeddings survive (centroids match at ~1.0000 cosine); the clustering
  groups them into the wrong number of speakers.
- Cause is **OpenBLAS 0.3.26 → 0.3.32**, isolated to that one shared object: speakrs is built
  `openblas-system`, so overwriting only `openblas-pthread/` inside the unchanged 26.04 image
  with 24.04's 0.3.26 makes the output byte-identical to the 24.04 image. glibc 2.43 is ruled
  out.
- amd64 is unaffected *on the fixture*, matching OpenBLAS 0.3.28/0.3.29's arm64-only GEMM→GEMV
  forwarding — but the fixture is now known to **understate** this failure, so amd64 has not
  really been cleared either.

**`verify-models` cannot catch this.** It passes every stage on the broken build, reporting a
plausible 2 speakers / 7 segments while exclusive DER is ~52%. A base bump needs a real-corpus
DER check on **each published architecture**, run **natively** — OpenBLAS picks kernels by
runtime CPU detection, so a QEMU run proves nothing. Tracked as issue #18.

---

See also: [CONFIGURATION.md](CONFIGURATION.md) · [PROVISIONING.md](PROVISIONING.md) ·
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) · [API.md](API.md) · [README](../README.md)
