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

Expect roughly **470 MB** written and **a few minutes**. About 32 MB is downloaded from the
gated repo; the rest is produced locally by the export. 189 MB of the output (~40%) is the
gender classifier — `--skip-gender` omits it, at the cost of speaker gender detection.

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
discovering it after a 470 MB export.

### Idempotency

A valid marker makes this a no-op, so it is safe to run unconditionally on every start.
`--force` re-exports.

```bash
diar-server verify-models --models-dir /models   # deep check: full sha256 + smoke test
```

### Exit codes

| code | meaning |
|---|---|
| 0 | provisioned, or already up to date |
| 2 | bad arguments |
| 3 | files were produced but failed the smoke test |
| 4 | the export subprocess failed |
| 5 | token missing/invalid, or repo terms not accepted |
| 6 | no usable python export environment |
| 7 | models directory not writable |

### What the marker does and does not claim

`provision-models` writes `<models-dir>/diar-provision.json` recording the export recipe
version, the upstream pipeline revision, the toolchain versions, and every file's size and
sha256, plus the smoke-test result.

Be precise about what is checked when:

- **At startup and on `/healthz`** the check is `stat`-only: the marker parses, the recipe
  version is current, the smoke test passed, and every recorded file is present at its
  recorded length. There is deliberately **no hashing** — re-reading 470 MB on every boot is
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
   b64 tail is batch-invariant. No reference data is committed (it would be a derivative of
   gated weights), and cross-path agreement is stronger anyway — it cannot be satisfied by a
   consistently-wrong export.
4. **End-to-end** diarization of a 26 s fixture, with sanity bounds on speakers, segments,
   centroids and gender verdicts.
5. **PLDA** `.npy` headers: exact dtype and shape.

## Step 1 — point OpenTranscribe at the sidecar

Live deployment consumes the **binary**, not this repo's image:
`transcribe-app/backend/Dockerfile.prod` pins `davidamacey/diar-native@sha256:...` purely to
`COPY --from` the `diar-server` binary and three ORT `.so`s into the backend image. The
sidecar then runs as compose service `diar-native` with `command: ["diar-server"]`.

See `transcribe-app/docker-compose.diar-native.yml`. Compose var defaults:
`DIAR_NATIVE_GPU=0`, `DIAR_NATIVE_MODE=cuda`, `DIAR_NATIVE_MODELS_DIR=<your models dir>`,
`DIAR_NATIVE_MAX_INFLIGHT=2`.

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

- **A missing model file is fatal** (exit 6), with a message naming provisioning and the gate
  URL. Without this, a half-provisioned directory surfaces as "CUDA session load failed" once
  per configured device, inside a `restart: unless-stopped` crash loop that also fails
  `up --wait` — and the operator's actual problem never appears in the logs.
- **A missing marker is only a warning.** Every models directory deployed before this feature
  shipped has no marker; refusing to start on those would turn a provenance improvement into
  an outage.

`DIAR_ALLOW_UNVERIFIED_MODELS=1` downgrades the fatal cases to warnings.

## Known limitations

- `num_speakers` forced counts: the native engine logs a warning and runs auto counting
  (min=1/max=20 defaults never bind; constraint port pending).
- VRAM: ~4.2 GB eager, lower with `SPEAKRS_LAZY_SESSIONS=1` (default in the overlay; see
  RESULTS §4.27).
