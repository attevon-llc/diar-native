# diar-native

Speaker diarization — **"who spoke when"** — as a small, self-hosted HTTP service. Rust and ONNX
Runtime, no Python at serving time, 195 MB on a CPU host.

[![CI](https://github.com/attevon-llc/diar-native/actions/workflows/ci.yml/badge.svg)](https://github.com/attevon-llc/diar-native/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Get started

```bash
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/install.sh -o install.sh
bash install.sh
```

That is the whole install. It works out your platform, picks the right published image, fetches
the compose file, explains and prompts for the free HuggingFace token it needs once, exports the
models on your machine, starts the service and waits until it reports ready. No clone, no Rust
toolchain, no manual steps. Downloading first and running second — rather than piping curl into
bash — is deliberate: you should be able to read a script before it touches your machine.

Useful flags: `--gpu` (require a GPU rather than silently falling back to CPU), `--cpu` (force
the CPU image), `--diarize <file>` (diarize something immediately and print the result),
`--port`, `--dir`, `--models`, `--no-start`. Run `bash install.sh --help` for the full list.

Re-running is safe and fast: provisioning is idempotent, so a second run skips it entirely and
needs no token and no network. Deploying from the compose file by hand instead is documented in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## What it does

It runs the **pyannote `speaker-diarization-community-1`** pipeline natively — segmentation,
speaker-conditioned embeddings, PLDA + VBx clustering — matching that pipeline's accuracy at a
fraction of its wall time and deployment weight. It is built on a vendored, heavily patched
[`speakrs`](https://github.com/avencera/speakrs).

Accuracy, scored with `pyannote.metrics` (collar 0.25, overlap included, official UEMs) — this
beats the production Python fork it replaces:

| AMI-16 (full) | Karpathy | VoxConverse |
|---|---|---|
| **13.101%** DER | **8.219%** DER | **4.847%** DER |

Speed. One number, one unit: **× realtime** — how many seconds of audio are diarized per second
of wall clock. Every multiplier below is relative to the PyAnnote fork, which is the baseline
because it is what runs in production today.

| | × realtime | vs the fork |
|---|---|---|
| **PyAnnote fork** — the customized production engine | 80–83× | 1.0× *(baseline)* |
| **stock `speakrs`** — unpatched, as adopted | 31–49× | **0.4–0.6× — slower** |
| **diar-native** — speakrs + our patch set | **184×** | **2.2× faster** |

So the honest shape of this: adopting `speakrs` alone made diarization **slower** than the fork.
What made it faster is the patch set — folded segmentation graphs, a multimask-batching fix, and
an fbank session pool that removed a CPU bottleneck the GPU was never responsible for. On a
same-session A/B that pool alone took ES2004a from 32.1 s to 12.9 s, **3.1× against stock, with
the RTTM byte-identical at every rung of the ladder**.

In wall-clock terms on the acceptance clip: 66.5 minutes of audio, one RTX A6000, warm engine —
**48.0 s → 21.6 s**, with DER moving 8.194% → 8.219% (+0.025 pp, inside the gate).

<sub>Measured on different corpora and cards, so the rows are comparisons against the fork, not a
single chained multiplier: fork and diar-native on Karpathy 66.5 min / A6000 (RESULTS 2.3, 4.21);
stock speakrs on AMI-16 (4.5); the fbank ladder on ES2004a / 3080 Ti (4.16).</sub>

Concurrent requests share one engine's VRAM. A GPU is optional; the CPU path produces the same
output, only slower. Conditions and the full record: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
and [validation/RESULTS.md](validation/RESULTS.md).

## Use it

Four routes: `POST /diarize`, `POST /embed_window`, `GET /healthz` (liveness — 200 whenever the
process is serving), `GET /readyz` (readiness — 200 only once the models are verified).

`/diarize` takes a **path inside the container**, not an upload, which is why your audio
directory is mounted at `/audio`. Despite the field name, anything symphonia reads decodes —
wav, flac, mp3, m4a, ogg, aac, mp4.

```bash
cp ~/meeting.wav ./audio/

curl -s -X POST localhost:8701/diarize \
  -H 'content-type: application/json' \
  -d '{"wav_path": "/audio/meeting.wav", "gender": true}'
```

```jsonc
{
  "segments":           [{"start": 0.51, "end": 4.28, "speaker": "SPEAKER_00"}],
  "exclusive_segments": [{"start": 0.51, "end": 4.28, "speaker": "SPEAKER_00"}],
  "centroids":          [[0.031, -0.114, "…256 floats…"]],
  "num_speakers": 2,
  "rttm": "SPEAKER meeting 1 0.510 3.770 <NA> <NA> SPEAKER_00 <NA> <NA>\n…",
  "speaker_gender": {"SPEAKER_00": {"label": "male", "confidence": 0.94, "windows": 12}}
}
```

`segments` may overlap, because two people genuinely can talk at once. **Use
`exclusive_segments` for transcripts** — same turns, overlaps resolved. Speaker labels are
arbitrary and stable only within one response. Full schemas, headers and the device field:
[docs/API.md](docs/API.md).

## Documentation

| document | what it answers |
|---|---|
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Compose, published images and digests, the platform matrix, ports and volumes, exit codes |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every environment variable the binary reads — the authoritative list |
| [docs/API.md](docs/API.md) | The four routes: request and response schemas, headers, device selection |
| [docs/PROVISIONING.md](docs/PROVISIONING.md) | How the models are obtained, the token, the marker, and what verification does *not* prove |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | The things that actually go wrong, and what each exit code means |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The speakrs relationship, the patch set, crate layout, and what runs where |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Benchmark numbers and the conditions they were measured under |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Building, the pinned toolchain, testing, and cutting a release |
| [docs/](docs/README.md) | Index of every document, including the deep dives |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md) | How to contribute · reporting vulnerabilities |

Three things worth knowing before you deploy, each of which cost someone real time:

- **Every published tag is single-platform, and `:latest` is amd64 CUDA.** `install.sh` picks
  correctly for you; naming a tag by hand does not. On arm64, Docker *emulates* an amd64 image
  rather than refusing it, so the symptom is "inexplicably slow", not an error.
- **On Apple Silicon under Docker you get CPU cores** — not the GPU, not the Neural Engine.
  Docker on macOS has no Metal access at any image architecture.
- **`/healthz` is 200 in every state while serving; `/readyz` is the readiness signal.** Gate
  container health on the first and your rollout on the second.

## Licence and attribution

diar-native is **Apache-2.0** ([LICENSE](LICENSE)), as is the
[`speakrs`](https://github.com/avencera/speakrs) engine it vendors.

The models are derivatives of
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
(**CC-BY-4.0**, gated). They are **not redistributed here and cannot be** — every operator
obtains and exports their own copy locally with their own HuggingFace token, which is what the
first run of `install.sh` is doing. Nothing gated is committed to this repository.
