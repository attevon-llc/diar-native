# Installing the Native Engine into OpenTranscribe

## Step 0 — provision the models

**Do this first.** The models are not shipped with `diar-server` and cannot be: they are
derivatives of the gated `pyannote/speaker-diarization-community-1` weights. There is no
`.onnx` on HuggingFace for any of them — the upstream repos ship `pytorch_model.bin` plus
`plda/*.npz`, so the conversion is mandatory, not an optimisation. Each operator converts
them locally with their own token; nothing is redistributed.

```bash
export HF_TOKEN=<your huggingface read token>
diar-server provision-models --models-dir /models --set fast
```

Expect roughly **484 MB** written and **a couple of minutes** — the acceptance run measured
**119.5 s** cold. About 32 MB is downloaded from the gated repo; the rest is produced locally by
the export. 189.5 MB of the output (~40%) is the gender classifier — `--skip-gender` omits it, at
the cost of speaker gender detection.

> The §7.36 acceptance run reports **673 MB**, not 484 MB. That run predates the fp16 gender fix
> (RESULTS §7.39) and hit the 378.5 MB fp32 fallback. Export recipe 2 restores fp16, so a
> directory provisioned by the current build is ~484 MB. `models_folded/` is the reference:
> 483,411,782 bytes.

### Prerequisites

1. A HuggingFace read token: <https://huggingface.co/settings/tokens>.
2. Accept the terms at
   <https://huggingface.co/pyannote/speaker-diarization-community-1> while signed in as that
   token's account. It is an email-capture prompt, auto-approved, and the pipeline is
   CC-BY-4.0 and free. Check it without exporting anything:

   ```bash
   diar-server check-token          # two HTTPS calls, ~200 ms, no download
   ```

3. A python interpreter with the export dependencies. `provision-models` shells out to
   `DIAR_EXPORT_PYTHON` (default `python3` on `PATH`) — the export needs torch and
   pyannote.audio, which `diar-server` deliberately does not bundle.
   - **In OpenTranscribe's backend image**, most are already present. Only `onnxscript` and a
     constant folder are missing; add them to `requirements.txt`.
   - **With the plain standalone image** (no python at all), use the provisioning image built
     from `docker/Dockerfile.provision`, which ships a pinned CPU-only environment.

   > **CPython 3.13 note.** `onnxsim` publishes no wheel for 3.13 at any version and is a C++
   > extension, so it cannot install on a 3.13 image without cmake and a toolchain. The
   > exporter falls back to `onnxslim` (pure-python wheel), which is numerically **bit-exact**
   > and eliminates the same ops, but emits a differently-shaped graph — so a directory folded
   > with onnxslim is functionally equivalent to `models_folded/` but not byte-comparable to
   > it. Which folder ran is recorded in the marker's `toolchain.folder`.

### Read-only mounts

The serving compose file mounts `/models:ro`. Provisioning needs read-write, so run it
against the host path (or mount `:rw` for that one command) and leave serving read-only.
`provision-models` checks writability up front and exits 7 naming the mount, rather than
discovering it after a 484 MB export.

Serving genuinely never writes to the models directory — the startup gate only `stat`s the
marker — so `:ro` is a guarantee rather than a convention. The one non-provisioning writer is
`verify-models`, which re-attests the marker by default; `--no-attest` suppresses that, and on
a read-only mount it degrades to a warning and still exits 0.

### The container user (uid/gid 10001:10001)

Both serving images (`docker/Dockerfile.server`, `docker/Dockerfile.server-cpu`) run as a
non-root user with a **fixed uid and gid of `10001:10001`**, account name `diar`. That number
is deliberately outside the 1000-1999 range `useradd` allocates on a normal host, so a file
left behind with this ownership is unambiguous and can never collide with a real account.

The same number appears in `docker-compose.yml`, `start.sh`, `QUICKSTART.md` and here; they
must be changed together.

**Serving needs no write access at all.** `/models` is mounted read-only, and the only path
the process writes is `/tmp/diar-native` (mode 1777, the audio handoff directory). So serving
works under this uid against a models directory owned by anybody, provided the files are
world-readable — which a default umask produces.

**Provisioning is the path that writes**, and a container user cannot write a host bind-mount
it does not own. Rather than making every operator `chown` a directory to 10001, the export
runs as the *invoking* user:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -e HUGGINGFACE_TOKEN -v "$PWD/models":/models diar-provision:<ver>
```

`start.sh` and the `provision` service in `docker-compose.yml` both do this (via `DIAR_UID` /
`DIAR_GID`, defaulted from `id -u` / `id -g`). The exported models land owned by the operator,
serving reads them as 10001, and **no `chown` is needed in the normal flow**. This is still
non-root; it is simply not *that* non-root user.

If a directory does end up owned by someone else — a leftover from a pre-0.3.1 root container,
say — the fix is to take ownership, **not** to chown to 10001:

```bash
sudo chown -R "$(id -u):$(id -g)" ./models
```

Chowning to 10001 would make serving work and re-provisioning fail, which is the worse of the
two failure modes because it does not surface until the next export.

`Dockerfile.provision` reclaims `USER root` for its `apt-get`/`pip` layers and drops back to
10001 at the end, so a bare `docker run` of it is not root either. It also sets `HF_HOME=/hf`
(mode 1777) because the export child resolves `~/.cache/huggingface` otherwise, and `~` is not
writable under a `--user` override.

### Which device provisioning uses

**`provision-models` and `verify-models` default to CPU**, deliberately — not to the serving
default of `cuda`. Provisioning is a once-per-deployment step that must work on a build host with
no GPU. `--mode` (or `DIAR_MODE`, or the first entry of `DIAR_DEVICES`) overrides it; an
unrecognized name is exit 2 here rather than the serving path's silent fall-through to `cuda`.

If the requested device genuinely is not usable, both commands exit **9** and write **no marker
at all** — "I could not test this" must never be recorded as "this is broken".

### Idempotency

A valid marker makes this a no-op, so it is safe to run unconditionally on every start.
`--force` re-exports. The no-op is decided before preflight, so it needs neither network nor
python nor a token.

```bash
diar-server verify-models --models-dir /models   # deep check: full sha256 + smoke test
```

### `verify-models` re-attests by default

On a directory that fully verifies (marker present, no hash drift, smoke test green),
`verify-models` **rewrites the marker's `smoke` record** with a fresh pass — mode, clip sha,
speakers, segments, duration, timestamp — and appends `(re-attested by verify-models)` to
`generated_by`. Provenance (`upstream`, `toolchain`, `speakrs`, `files`) is left untouched,
because this run exported nothing.

That is the supported recovery path for a directory carrying a stale `fail` record: it clears the
record without a full re-export. `--no-attest` opts out. A read-only mount is **not** an error —
verification still passes and the exit code is unaffected; the marker simply is not updated.

### Choosing the smoke clip

Stage 4 needs one 16 kHz mono WAV of at least 10 s containing speech. Both serving images bake
one in at `/usr/local/share/diar-native/smoke.wav`; the fallback is
`vendor/speakrs/fixtures/test.wav`. Images that merely *copy the binary out* of a diar-server
image — OpenTranscribe's backend image is exactly that — have neither, so pass
`--smoke-clip /path/to/clip.wav`. Any short recording will do; it is only ever read, never
redistributed.

The clip is resolved **late**, after the writability, idempotency, token and python checks, so
the documented provision-from-the-backend-image route no longer dies at exit 2 before it has so
much as looked at the token.

### Exit codes

Authoritative source: `crates/diar-core/src/provision/mod.rs::exit`.

| code | name | meaning | emitted by |
|---|---|---|---|
| 0 | `OK` | provisioned, or already up to date | all |
| 1 | *(none)* | serve only: any other startup failure (bind failed, engine load failed) | serve |
| 2 | `USAGE` | bad arguments | all |
| 3 | `SMOKE_FAILED` | files were produced but failed the smoke test; in `verify-models`, also recorded-hash drift | provision, verify |
| 4 | `EXPORT_FAILED` | the export subprocess failed | provision |
| 5 | `TOKEN_DENIED` | token missing/invalid, or repo terms not accepted | provision, check-token |
| 6 | `NO_EXPORTER_ENV` | no usable python export environment | provision |
| 7 | `NOT_WRITABLE` | models directory not writable | provision |
| 8 | `MODELS_UNUSABLE` | **serve only:** the models directory is too broken to start against | serve |
| 9 | `DEVICE_UNAVAILABLE` | the requested execution device is not usable here; no marker is written | provision, verify |
| 10 | `UNVERIFIABLE` | **verify only:** the files work, but there is no marker to compare them against | verify |

> **Changed in 0.3.0: the startup model gate exits 8, not 6.** The two used to share a code, so a
> supervisor could not distinguish "install torch into the exporter" from "provision the models".
> Serving needs no python at all. Any script branching on `6` for a startup failure must be
> updated. Codes 9 and 10 are also new.

### What the marker does and does not claim

`provision-models` writes `<models-dir>/diar-provision.json` recording the export recipe
version, the upstream pipeline revision, the toolchain versions, and every file's size and
sha256, plus the smoke-test result.

Be precise about what is checked when:

- **At startup and on `/healthz`** the check is `stat`-only: the marker parses, the recipe
  version is current, the smoke test passed, and every recorded file is present at its
  recorded length. There is deliberately **no hashing** — re-reading ~484 MB on every boot is
  unacceptable, and mtime is useless as a proxy because `docker cp` and volume copies rewrite
  it.
- **`verify-models` and `provision-models`** do the deep tier: full sha256 plus the whole
  smoke test.

So startup answers *"is this the directory that passed?"*, not *"is this directory still
byte-perfect?"*. A file rewritten to the same length passes startup and fails
`verify-models`. Claiming more than that would itself be a fail-open.

### What the smoke test actually checks

Five stages, in Rust, against the same ORT build the server uses (a python-side check would
validate a different runtime). Stages 1-3 and 5 run on CPU — no GPU needed.

1. **Parse** every `.onnx`. Non-obvious: live compose sets `SPEAKRS_LAZY_SESSIONS=1`, and
   speakrs then skips the batch-64 sessions at startup — so a corrupt `*-b64.onnx` is
   invisible to a normal server start.
2. **I/O contract** against a compiled-in table of names and shapes. Catches the
   right-filename/wrong-model case (RESULTS §1).
3. **Cross-path numeric agreement**: fbank b1 vs b32; the fused embedding graph vs the split
   fbank→tail path; multimask vs single tail; the b64 multimask is a byte copy of b32; the
   b64 tail is batch-invariant; **and (3e) the b32 multimask graph agrees with the b1 multimask
   graph under an identical mask.** No reference data is committed (it would be a derivative of
   gated weights), and cross-path agreement is stronger anyway — it cannot be satisfied by a
   consistently-wrong export.

   > 3e is the stage that verifies **what production actually runs**. Live compose sets
   > `SPEAKRS_LAZY_SESSIONS=1`, under which speakrs skips the batched sessions but **not** the
   > multimask ones — so before 3e existed, a `wespeaker-multimask-tail-b32.onnx` with its
   > largest weight tensor zeroed passed all five stages green, earned a `verified` marker and a
   > 200 from `/readyz`, while every file longer than one window was embedded by a broken graph.
   > It is now caught at 9.222e-1 against a 1e-4 bar (RESULTS §7.38).
4. **End-to-end** diarization of a 26 s fixture, with sanity bounds on speakers, segments,
   centroids and gender verdicts.
5. **PLDA** `.npy` headers: exact dtype and shape.

## Step 1 — point OpenTranscribe at the sidecar

Live deployment consumes the **binary**, not this repo's image:
`transcribe-app/backend/Dockerfile.prod` pins `davidamacey/diar-native@sha256:...` purely to
`COPY --from` the `diar-server` binary and three ORT `.so`s into the backend image. The
sidecar then runs as compose service `diar-native` with `command: ["diar-server"]`.

See `transcribe-app/docker-compose.diar-native.yml`. Its `DIAR_NATIVE_*` variables are
compose-level indirection that expand into the `DIAR_*` variables `diar-server` actually reads
(README §6e). Current defaults in that file:

| compose var | expands to | default |
|---|---|---|
| `DIAR_NATIVE_IMAGE` | the service `image:` | `davidamacey/opentranscribe-backend:${OT_IMAGE_TAG:-latest}` — the **shared backend image**, not a diar-server image |
| `DIAR_NATIVE_MODE` | `DIAR_MODE` | `cuda` |
| `DIAR_NATIVE_MAX_INFLIGHT` | `DIAR_MAX_INFLIGHT` | `2` |
| `DIAR_NATIVE_MODELS_DIR` | the `/models:ro` bind source | `${MODEL_CACHE_DIR:-./models}/diar-native` |
| `DIAR_NATIVE_LAZY_SESSIONS` | `SPEAKRS_LAZY_SESSIONS` | `1` |
| `DIAR_NATIVE_GPU` | `deploy.…device_ids` | `${GPU_DEVICE_ID:-0}` — **not** a bare `0`; enabling the overlay without setting it used to reserve the wrong GPU |

`DIAR_MODELS_DIR=/models` and `DIAR_BIND=0.0.0.0:8701` are set literally in that file, not
through an indirection.

> `DIARIZER_ENGINE` is **obsolete** — the compose file's own comment notes it "now selects
> nothing". Engine selection is handled by `opentranscribe.sh` and the release manifest.

## Step 2 — health endpoints

`GET /healthz` **always returns 200 while the server is up**, in every model state. This is
load-bearing and must not be changed: `docker-compose.diar-native.yml` runs
`curl -sf .../healthz || exit 1`, and `diarizer_native.py` checks `resp.status == 200`. If
`/healthz` returned 503 for "unverified", then on the day this ships every existing
deployment — none of which has a marker — would fail its healthcheck, fail `up --wait`, and
silently fall back to in-process PyAnnote. That is exactly the slow quality regression this
work exists to prevent.

The body gains flat `models_*` fields (`models_verified`, `models_state`, `models_dir`,
`models_set`, `models_exporter_version`, `models_pipeline_revision`, `models_smoke_at`,
`models_reason`), where `models_state` is one of `verified | stale | unverified | failed` and
`models_reason` carries a human sentence plus the remediation command for every non-verified
state.

`GET /readyz` is **new**: 200 only when `models_state == "verified"`, 503 otherwise, same
body. That is where "still provisioning" is distinguished from "broken", with zero blast
radius on existing callers.

**Migration.** After provisioning once, switch the compose healthcheck from `/healthz` to
`/readyz`. OpenTranscribe repoints `sidecar_healthy()` in its own PR — not here.

## Step 3 — startup behaviour

Before loading any engine, the server does a `stat`-only pass over the required files. The
asymmetry is deliberate:

- **A missing or zero-length model file is fatal** (exit **8**), with a message naming
  provisioning and the gate URL. Without this, a half-provisioned directory surfaces as "CUDA
  session load failed" once per configured device, inside a `restart: unless-stopped` crash loop
  that also fails `up --wait` — and the operator's actual problem never appears in the logs.
  A marker that records a failed smoke test, or vouches for a file that is now the wrong length,
  is equally fatal.
- **A missing marker is only a warning.** Every models directory deployed before this feature
  shipped has no marker; refusing to start on those would turn a provenance improvement into
  an outage. An unparseable marker, and one written to a newer schema, are likewise warnings.
- **A stale marker is only a warning.** `stale` means the recipe version differs from the one
  this build ships (`EXPORT_RECIPE_VERSION`, currently **2**), or the directory is a `small` set
  being asked to serve `fast`. Stale directories serve normally — but they return 503 from
  `/readyz`, since that gates on `verified` exactly.

Which tier the gate requires is read from the directory's **own marker**, falling back to `fast`
when there is none. `DIAR_MODEL_SET=fast|small` overrides it, for an operator who wants to assert
that a directory ought to be a given tier and get a loud complaint when it is not.

`DIAR_ALLOW_UNVERIFIED_MODELS=1` downgrades the fatal cases to warnings. It matches exactly `1`,
`true`, `TRUE` or `yes` — note that `True` does not work.

## Step 4 — logs

The server logs to **stdout** with `tracing`, so `docker logs diar-native` and `docker compose
logs` show it with no configuration. Fatal startup errors (the gate block above) stay on
stderr. Two knobs (the full env-var list, for every subcommand, is README §6e — that table is
authoritative and this one is a convenience excerpt):

| var | default | meaning |
|---|---|---|
| `RUST_LOG` | `info,ort::logging=warn` | Standard `tracing` filter. The default gives the startup line, warnings, and one line per request. `speakrs=debug` adds the engine's stage timings (fbank, GPU predict, clustering). Empty is treated as unset; a malformed value warns and falls back to the default rather than starting silent. ONNX Runtime's native bridge is held at `warn` because it emits 5797 INFO lines per CUDA startup (RESULTS §7.37); `RUST_LOG=ort=info` brings it back. |
| `DIAR_LOG_FORMAT` | `text` | `text` for humans, `json` for an aggregator (one flattened object per line). Unrecognized values warn and use `text`. |

Add them to the `diar-native` service in `transcribe-app/docker-compose.diar-native.yml`
(that file lives in the consuming repo — this change is made there, not here):

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

Set `DIAR_LOG_FORMAT=json` when OpenTranscribe's log shipping wants structured records; leave
it unset for hand debugging.

**What a request looks like.** Each `/diarize` and `/embed_window` call gets a span with
`request_id`, `endpoint`, `device`, the audio **basename** (never the full path) and the
`gender` flag, and ends with one record carrying `duration_ms`, `outcome`, and either
`num_speakers`/`segments` or an `error_class` — one of `bad_device`, `admission`,
`invalid_input`, `audio_decode`, `inference`, `panic`. That is enough to tell a caller error
from a sidecar fault without reproducing anything.

**Correlating with the caller.** An inbound `x-request-id` header is reused as the request id
and echoed back on the response (including on 4xx/5xx), so one id spans OpenTranscribe's log
and the sidecar's. If the caller sends none, the server generates one. Inbound ids are
sanitized before they are logged.

Nothing sensitive is logged: no full media paths, no model weights, and no HuggingFace token
— `provision-models` scrubs the token from the exporter's stdout and stderr and hides it from
clap's error rendering.

## Known limitations

- `num_speakers` forced counts: the native engine logs a warning and runs auto counting
  (min=1/max=20 defaults never bind; constraint port pending).
- VRAM: ~4.2 GB eager, lower with `SPEAKRS_LAZY_SESSIONS=1` (default in the overlay; see
  RESULTS §4.27).
