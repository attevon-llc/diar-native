# diar-native

Speaker diarization — **"who spoke when"** — as a small, self-hosted HTTP service. Rust and ONNX
Runtime, no Python at serving time, 195 MB on a CPU host.

**Source:** [attevon-llc/diar-native](https://github.com/attevon-llc/diar-native) · Apache-2.0

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/attevon-llc/diar-native/v0.3.1/install.sh -o install.sh
bash install.sh
```

That is the whole install. It detects your platform, picks the right tag below, prompts once for
the free HuggingFace token it needs, exports the models on your machine, and waits until the
service reports ready. Re-running is safe and skips provisioning entirely.

Deploying by hand instead:
[docs/DEPLOYMENT.md](https://github.com/attevon-llc/diar-native/blob/main/docs/DEPLOYMENT.md).

## Tags

> **Every tag here is single-platform.** Docker does *not* refuse a mismatched image — it
> emulates, so the symptom is "inexplicably slow", not an error. Pick deliberately, or let
> `install.sh` pick for you.

| tag | platform | size | contents |
|---|---|---|---|
| `0.3.1`, `latest` | linux/amd64 | 3.04 GB | CUDA **and** CPU, selected per request |
| `0.3.1-cpu` | linux/amd64 | 195 MB | CPU only, no NVIDIA runtime needed |
| `0.3.1-cpu-arm64` | linux/arm64 | 223 MB | CPU only. ARM Linux, and Apple Silicon under Docker — **on CPU cores**, not the GPU or Neural Engine |
| `0.3.1-provision` | linux/amd64 | 1.94 GB | **Provisioning only.** Serving image + a pinned CPU-only torch/pyannote environment. Delete once the models exist |
| `0.3.1-provision-arm64` | linux/arm64 | 1.86 GB | Same, for arm64 |

All scanned with trivy 0.67.1: **0 HIGH / 0 CRITICAL**. All run as **uid 10001**, non-root.
Digests for pinning are in
[docs/DEPLOYMENT.md](https://github.com/attevon-llc/diar-native/blob/main/docs/DEPLOYMENT.md).

## Use it

Four routes: `POST /diarize`, `POST /embed_window`, `GET /healthz`, `GET /readyz`.

```bash
curl -s -X POST localhost:8701/diarize \
  -H 'content-type: application/json' \
  -d '{"wav_path": "/audio/meeting.wav", "gender": true}'
```

`/diarize` takes a **path inside the container**, not an upload, so mount your audio at `/audio`.
Anything symphonia decodes works: wav, flac, mp3, m4a, ogg, aac, mp4.

Use **`exclusive_segments`** for transcripts — `segments` may overlap, because two people
genuinely can talk at once.

**`/healthz` returns 200 in every model state while serving; `/readyz` is the readiness
signal.** Gate container health on the first and your rollout on the second.

## Accuracy and speed

Runs the **pyannote `speaker-diarization-community-1`** pipeline natively. Scored with
`pyannote.metrics` (collar 0.25, overlap included, official UEMs):

| AMI-16 (full) | Karpathy | VoxConverse |
|---|---|---|
| **13.101%** DER | **8.219%** DER | **4.847%** DER |

Speed, in **× realtime** (seconds of audio diarized per second of wall clock), against the
customized PyAnnote deployment this replaces: **184× vs 80–83×, i.e. 2.2× faster**. On the
acceptance clip that is 66.5 minutes of audio in **21.6 s** on one RTX A6000.

## Models are not bundled

**Nothing is baked into these images.** The first run downloads the upstream weights with
**your** HuggingFace token and exports them locally (~484 MB, about two minutes, once). Later
starts find a provenance marker and skip it, needing no token and no network.

The diarization models derive from
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
(**CC-BY-4.0**, gated, auto-approved); the optional gender classifier derives from
[`prithivMLmods/Common-Voice-Gender-Detection`](https://huggingface.co/prithivMLmods/Common-Voice-Gender-Detection)
(not gated). Neither is redistributed here, and the gated one cannot be — every operator obtains
their own copy. **CC-BY-4.0 requires attribution:** if you ship something built on this, credit
pyannote.

## Links

[Documentation](https://github.com/attevon-llc/diar-native/tree/main/docs) ·
[Configuration](https://github.com/attevon-llc/diar-native/blob/main/docs/CONFIGURATION.md) ·
[API](https://github.com/attevon-llc/diar-native/blob/main/docs/API.md) ·
[Troubleshooting](https://github.com/attevon-llc/diar-native/blob/main/docs/TROUBLESHOOTING.md) ·
[Changelog](https://github.com/attevon-llc/diar-native/blob/main/CHANGELOG.md) ·
[Issues](https://github.com/attevon-llc/diar-native/issues)
