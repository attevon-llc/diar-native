# HTTP API

The `diar-server` sidecar exposes **four routes**. This document is the reference for all of
them. For getting a server running at all, start at the [README](../README.md).

| route | purpose |
|---|---|
| `POST /diarize` | the whole job — segments, exclusive segments, centroids, speaker count, RTTM, optional gender |
| `POST /embed_window` | a speaker embedding for one time window |
| `GET /healthz` | liveness. **Always 200 while the process is serving**, in every model state |
| `GET /readyz` | readiness. 200 **only** once the models are verified; 503 otherwise |

Default bind address is `0.0.0.0:8701` (`DIAR_BIND`).

---

## `POST /diarize`

Diarizes a whole file and returns every speaker turn in it.

### Request

```json
{
  "wav_path": "/audio/meeting.wav",
  "gender": true,
  "device": "cpu",
  "file_id": "any-caller-supplied-string"
}
```

| field | type | required | meaning |
|---|---|---|---|
| `wav_path` | string | **yes** | Path to media **inside the container**, on a shared volume. Not an upload. |
| `gender` | bool | no (`false`) | Also classify each speaker's gender from the same decoded audio — saves the caller a second fetch, decode and model host. |
| `device` | string\|null | no (null) | Execution device for this request (`"cuda"`, `"cpu"`, …). Omitted or null = the server's default device. |
| `file_id` | string\|null | no | Opaque caller-supplied identifier, echoed into the server-side log record. |

**`wav_path` is a path, not an upload**, which is why deployments bind-mount an audio
directory at `/audio`. **`media_path` and `audio_path` are accepted as aliases** for the same
field. The `wav_path` name is kept because the live OpenTranscribe caller sends it, but it
undersells the field and has led third parties to transcode to WAV first for no reason: a
16 kHz mono WAV takes the exact handoff fast path, and **anything else — mp3, m4a, flac, ogg,
aac, mp4, any-rate wav — is decoded and resampled in-process**. Anything symphonia can read
works.

`device` is deliberately a plain string and not a derived enum: axum 0.7 turns a serde variant
mismatch into a bare 422 with no useful body, whereas parsing it in the handler yields a 400
that names the devices this build actually serves.

### Response

```jsonc
{
  "segments": [
    {"start": 0.51, "end": 4.28, "speaker": "SPEAKER_00"}
  ],
  "exclusive_segments": [
    {"start": 0.51, "end": 4.28, "speaker": "SPEAKER_00"}
  ],
  "centroids": [[0.031, -0.114, "…256 floats…"]],
  "num_speakers": 2,
  "rttm": "SPEAKER meeting 1 0.510 3.770 <NA> <NA> SPEAKER_00 <NA> <NA>\n…",
  "speaker_gender": {
    "SPEAKER_00": {"label": "male", "confidence": 0.94, "windows": 12}
  }
}
```

- **`segments`** — raw turns, the pyannote `speaker_diarization` equivalent. These **can
  overlap**, because two people genuinely can talk at once. Good for diarization metrics,
  awkward for anything that needs a single speaker per moment.
- **`exclusive_segments`** — the same turns with overlaps resolved so no two segments cover the
  same instant (the pyannote `exclusive_speaker_diarization` equivalent). **This is the one you
  want** if you are attaching speaker labels to a transcript.
- **`centroids`** — gamma-weighted, **un-normalized** per-speaker 256-d embeddings; row *i*
  corresponds to `SPEAKER_{i:02}`. Consumers that index them (the OpenSearch kNN path) apply
  their own L2 normalization.
- **`num_speakers`** — how many distinct speakers were found. It is **discovered, not
  configured**; you do not have to know it in advance.
- **`rttm`** — the whole result as a standard RTTM string, ready to write to a `.rttm` file and
  feed to any scoring tool.
- **`speaker_gender`** — omitted entirely unless you sent `"gender": true` *and* the gender
  model is deployed. `confidence` is 0-1 and `windows` is how many audio windows voted, which is
  your signal for how much to trust a verdict: a speaker with 2 windows is a guess, one with 40
  is not.

**Speaker labels (`SPEAKER_00`, …) are arbitrary and stable only within one response.** The
same person in two different files will not get the same label.

### Not yet supported

Min / max / exact speaker-count constraints are **not implemented** (T9b). A forced count is
currently warned about and then auto-counted anyway.

---

## `POST /embed_window`

Returns one speaker embedding for one window of audio. Used for boundary rechecks, where the
caller already knows the time range it cares about.

```json
{
  "wav_path": "/audio/meeting.wav",
  "start_s": 12.0,
  "end_s": 18.5,
  "device": "cpu"
}
```

| field | type | meaning |
|---|---|---|
| `wav_path` | string\|null | Path to media, any symphonia-decodable format. Aliases `media_path` / `audio_path`, as on `/diarize`. |
| `samples_b64` | string\|null | …or raw 16 kHz mono `f32` little-endian samples, base64-encoded. For small clips, when the caller has the audio in hand and no shared volume. |
| `start_s` | float\|null | Window start in seconds. `wav_path` input only. |
| `end_s` | float\|null | Window end in seconds. `wav_path` input only. |
| `device` | string\|null | As on `/diarize`. |

Response:

```json
{"embedding": [0.031, -0.114, "…256 floats…"]}
```

---

## `GET /healthz`

```json
{
  "status": "ok",
  "default_device": "cuda",
  "devices": ["cuda", "cpu"],
  "supported_devices": ["cuda", "cpu"],

  "models_verified": true,
  "models_state": "verified",
  "models_dir": "/models",
  "models_set": "fast",
  "models_exporter_version": 2,
  "models_pipeline_revision": "a1b2c3d…",
  "models_smoke_at": "2026-09-01T04:15:22Z",
  "models_gender": true,
  "models_reason": null
}
```

`devices` = loaded and serving in this process (the first is the default). `supported_devices` =
what this **build** can serve — a superset; something listed there but missing from `devices`
needs a `DIAR_DEVICES` change, not a rebuild.

`models_state` is one of `verified | stale | unverified | failed`, and `models_reason` carries a
human sentence plus the remediation command for every non-verified state. `models_gender`
reports whether the gender classifier **file is present** — gender is enabled by file presence,
so a `--skip-gender` deployment answers `diarize(gender=true)` with 200 and no genders, and this
field is the difference between that being a decision and a mystery. (It does **not** report the
model's *precision*; that is `toolchain.gender_precision` in the provenance marker.) The fields
are flat rather than nested so that appending more stays additive.

> **`/healthz` returns 200 in every state — this is a guarantee, not an accident.** The compose
> healthcheck is `curl -sf .../healthz` and OpenTranscribe's `diarizer_native.py` checks
> `resp.status == 200`. Every models directory deployed today has no marker, so a 503 for
> "unverified" would fail every existing healthcheck on the day it shipped, fail `up --wait` for
> the whole stack, and silently fall OpenTranscribe back to in-process PyAnnote — the exact
> quality regression this work exists to prevent. Changing the **body** is safe; changing the
> **code** is not. Use `/readyz` when you want a readiness signal that is allowed to fail.

This route was the bare string `ok` before 0.3.0.

## `GET /readyz`

Same body as `/healthz`, but **200 only when `models_state == "verified"`** and 503 otherwise —
so `stale` and `unverified` both return 503 while the server keeps serving requests normally.
This is where "still provisioning" is distinguished from "broken", with zero blast radius on
existing callers. After provisioning once, move your compose healthcheck here.

---

## Response headers

| header | on | meaning |
|---|---|---|
| `x-diar-device` | `/diarize`, `/embed_window` success | The device that actually ran the job (`cuda`\|`cpu`\|…). A header rather than a body field because `DiarizeOutput` is the consumer's parsed schema. Its **presence** is also the cheapest capability probe for the multi-device feature. |
| `x-request-id` | `/diarize`, `/embed_window`, success **and** error | The id this request was logged under. Echoed from the inbound `x-request-id` if the caller sent one (sanitized), otherwise generated. Present on 4xx/5xx too, so a caller looking at a failure can find the matching server-side record without guessing. |

`x-request-id` is also accepted as a **request** header, so a job keeps one id end to end
through a larger stack. It is sanitized before it is logged (control characters stripped, 64
characters max) — a caller cannot forge a log record with it.

---

## Selecting a device

The CUDA image serves **both `cuda` and `cpu`**, chosen per request; see
[ARCHITECTURE.md](ARCHITECTURE.md#one-image-cpu-and-cuda) for why that costs no extra bytes, and
[CONFIGURATION.md](CONFIGURATION.md) for `DIAR_DEVICES` / `DIAR_MODE` / `DIAR_MAX_INFLIGHT`.

```bash
# a GPU deployment that can also answer CPU requests
DIAR_DEVICES=cuda,cpu diar-server
curl -sX POST localhost:8701/diarize -d '{"wav_path":"/audio/x.wav","device":"cpu"}'
```

CPU and CUDA produce **bit-identical centroids**; segment boundaries can differ by one frame.

> **Silent-ignore trap — read this before sending `device`.** Neither request struct uses
> `deny_unknown_fields`, so an **old** diar-server does not reject `{"device":"cpu"}` — it
> *ignores* it and runs the job on CUDA anyway, returning 200. Serde cannot help you here.
> Consumers MUST negotiate on `/healthz` `supported_devices` (or on the presence of the
> `x-diar-device` response header) before relying on the field. An unknown device name on a
> *new* server is a 400 that names the devices the build serves; on an old one it is a silent
> success on the wrong device.

---

## Errors

Failures carry an `error_class`, which is also what the server-side log record records:
`bad_device`, `admission`, `invalid_input`, `audio_decode`, `inference`, `panic`. Every error
response carries `x-request-id`.

## Concurrency

`DIAR_MAX_INFLIGHT` (default 2) bounds the **total** inflight requests across all devices, so
adding an engine cannot silently double concurrency. `DIAR_MAX_INFLIGHT_CPU` is an optional
inner sub-gate for CPU work only; CPU requests take the global permit first and the inner one
second, always in that order. Requests run on cloned engine handles with no engine mutex, so
concurrent jobs share one engine's VRAM. Full details in
[CONFIGURATION.md](CONFIGURATION.md).

---

See also: [CONFIGURATION.md](CONFIGURATION.md) · [DEPLOYMENT.md](DEPLOYMENT.md) ·
[ARCHITECTURE.md](ARCHITECTURE.md) · [README](../README.md)
