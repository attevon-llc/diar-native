# Installing the Native Engine into OpenTranscribe

The flip procedure for `transcribe-app` specifically. Everything here is about the *consuming*
repo; generic deployment, provisioning and configuration are owned elsewhere and are not repeated
on this page:

| you want | read |
|---|---|
| To get the models | [PROVISIONING.md](PROVISIONING.md) |
| Compose, images, ports, volumes, the container user, exit codes | [DEPLOYMENT.md](DEPLOYMENT.md) |
| Every environment variable `diar-server` reads | [CONFIGURATION.md](CONFIGURATION.md) |
| Endpoint schemas and headers | [API.md](API.md) |

## Step 0 — provision the models

**Do this first.** The models are not shipped with `diar-server` and cannot be — see
[PROVISIONING.md](PROVISIONING.md) for the token, the export and what the marker claims.

```bash
export HF_TOKEN=<your huggingface read token>
diar-server provision-models --models-dir /models --set fast
```

Roughly 484 MB, a couple of minutes, once.

## Step 1 — point OpenTranscribe at the sidecar

Live deployment consumes the **binary**, not this repo's image:
`transcribe-app/backend/Dockerfile.prod` pins `davidamacey/diar-native@sha256:...` purely to
`COPY --from` the `diar-server` binary and three ORT `.so`s into the backend image. The sidecar
then runs as compose service `diar-native` with `command: ["diar-server"]`.

> **A release here does not reach production on its own.** It ships only when that `@sha256:`
> digest is repointed, and that change is made in `transcribe-app`. Background:
> [ARCHITECTURE.md](ARCHITECTURE.md#production-consumes-the-binary-not-the-image), which also
> records why the binary works inside the backend image (all six CUDA sonames already match;
> only `libopenblas0` must be added).

See `transcribe-app/docker-compose.diar-native.yml`. Its `DIAR_NATIVE_*` variables are
compose-level indirection that expand into the `DIAR_*` variables `diar-server` actually reads —
no Rust code reads a `DIAR_NATIVE_*` name. Current defaults in that file:

| compose var | expands to | default |
|---|---|---|
| `DIAR_NATIVE_IMAGE` | the service `image:` | `davidamacey/opentranscribe-backend:${OT_IMAGE_TAG:-latest}` — the **shared backend image**, not a diar-server image |
| `DIAR_NATIVE_MODE` | `DIAR_MODE` | `cuda` |
| `DIAR_NATIVE_MAX_INFLIGHT` | `DIAR_MAX_INFLIGHT` | `2` |
| `DIAR_NATIVE_MODELS_DIR` | the `/models:ro` bind source | `${MODEL_CACHE_DIR:-./models}/diar-native` |
| `DIAR_NATIVE_LAZY_SESSIONS` | `SPEAKRS_LAZY_SESSIONS` | `1` |
| `DIAR_NATIVE_GPU` | `deploy.…device_ids` | `${GPU_DEVICE_ID:-0}` — **not** a bare `0`; enabling the overlay without setting it used to reserve the wrong GPU |

`DIAR_MODELS_DIR=/models` and `DIAR_BIND=0.0.0.0:8701` are set literally in that file, not through
an indirection.

> `DIARIZER_ENGINE` is **obsolete** — the compose file's own comment notes it "now selects
> nothing". Engine selection is handled by `opentranscribe.sh` and the release manifest.

## Step 2 — the healthcheck migration

`GET /healthz` **always returns 200 while the server is up**, in every model state, and that is
load-bearing here specifically: `docker-compose.diar-native.yml` runs `curl -sf .../healthz ||
exit 1` and `diarizer_native.py` checks `resp.status == 200`. Full guarantee and rationale:
[API.md](API.md#get-healthz).

**The migration:** after provisioning once, switch the compose healthcheck from `/healthz` to
`/readyz`, which is 200 only when `models_state == "verified"`. OpenTranscribe repoints
`sidecar_healthy()` in its own PR — not here.

## Step 3 — logging in the consuming compose file

Add these to the `diar-native` service in `transcribe-app/docker-compose.diar-native.yml` (that
file lives in the consuming repo — this change is made there, not here):

```yaml
services:
  diar-native:
    environment:
      - DIAR_MODELS_DIR=/models
      - DIAR_MAX_INFLIGHT=${DIAR_NATIVE_MAX_INFLIGHT:-2}
      # Logging. Both are optional; these are the built-in defaults spelled out.
      - RUST_LOG=${DIAR_NATIVE_LOG_LEVEL:-info}
      - DIAR_LOG_FORMAT=${DIAR_NATIVE_LOG_FORMAT:-text}
```

Set `DIAR_LOG_FORMAT=json` when OpenTranscribe's log shipping wants structured records; leave it
unset for hand debugging. Defaults and the full filter policy:
[CONFIGURATION.md](CONFIGURATION.md#logging).

**Correlating with the caller.** An inbound `x-request-id` header is reused as the request id and
echoed back on the response (including on 4xx/5xx), so one id spans OpenTranscribe's log and the
sidecar's. If the caller sends none, the server generates one; inbound ids are sanitized before
they are logged. Each request record carries `duration_ms`, `outcome`, and either
`num_speakers`/`segments` or an `error_class` — enough to tell a caller error from a sidecar fault
without reproducing anything.

Nothing sensitive is logged: no full media paths, no model weights, and no HuggingFace token.

## Known limitations

- **Forced `num_speakers` counts are not supported.** The native engine logs a warning and runs
  auto-counting (the min=1 / max=20 defaults never bind; the constraint port is pending — T9b).
- **VRAM:** ~4.2 GB eager, lower with `SPEAKRS_LAZY_SESSIONS=1` (the default in the overlay; see
  RESULTS §4.27 and [VRAM_AND_TIERS.md](VRAM_AND_TIERS.md)).

---

See also: [DEPLOYMENT.md](DEPLOYMENT.md) · [PROVISIONING.md](PROVISIONING.md) ·
[ARCHITECTURE.md](ARCHITECTURE.md) · [README](../README.md)
