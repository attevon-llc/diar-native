# Installing the Native Engine into OpenTranscribe (M2 flip procedure)

Everything below is staged and OFF by default. Nothing runs until you execute these steps.
Rollback at any point = unset `DIARIZER_ENGINE` (fork path is the default) + stop the sidecar.

## Already in place (additive files, no tracked-file edits)

- `transcribe-app/backend/app/transcription/diarizer_native.py` — SpeakerDiarizer-compatible
  client (NEW file; nothing imports it until the hook below is applied).
- `transcribe-app/docker-compose.diar-native.yml` — sidecar service + worker env/volume overlay
  (only active when passed with `-f`).
- `diar-server:latest` image (built from `diar-native/docker/Dockerfile.server`).

## Step 1 — the one tracked-file edit (apply when your app testing is done)

In `backend/app/transcription/model_manager.py`, `get_diarizer()`, replace:

```python
            diarizer = SpeakerDiarizer(config)
```

with:

```python
            if os.environ.get("DIARIZER_ENGINE", "python").lower() == "native":
                from app.transcription.diarizer_native import NativeSpeakerDiarizer

                diarizer = NativeSpeakerDiarizer(config)
            else:
                diarizer = SpeakerDiarizer(config)
```

(`import os` at top if absent.) Type annotation stays `SpeakerDiarizer` — duck-typed surface is
identical; loosen to a Protocol later if desired. `load_model()` on the native path is a
health-check; failures raise, and the worker's existing error handling applies. For automatic
fallback-on-failure, wrap in try/except and fall back to `SpeakerDiarizer(config)` — recommended
for shadow phase.

## Step 2 — bring up the sidecar

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
               -f docker-compose.diar-native.yml up -d diar-native
curl -s localhost: (mapped port or exec) http://diar-native:8701/healthz  # → ok
```

Compose var defaults: `DIAR_NATIVE_GPU=0`, `DIAR_NATIVE_MODE=cuda`,
`DIAR_NATIVE_MODELS_DIR=/path/to/diar-native/models_folded`,
`DIAR_NATIVE_MAX_INFLIGHT=2`. Under the gpu-split profile, move the env/volume overlay from
`celery-worker` to your diarize worker service name.

## Step 3 — flip + verify (M2 gate, from validation/TESTPLAN.md)

```bash
DIARIZER_ENGINE=native docker compose ... up -d celery-worker
```
1. Upload a `test_videos/` file through the app; verify diarized transcript + speaker labels
   + OpenSearch embedding ingestion.
2. Kill-sidecar drill: `docker stop diar-native` mid-queue → jobs must complete via fallback
   (with the try/except variant of the hook).
3. Karpathy **10-minute** clip WSER through the app pipeline ≤ 0.27% smoothed — that is the clip
   the 0.27% figure was measured on (`docs/diarization-boundary-results/cloud-comparison.md`);
   scoring the full 66.5-min clip instead gives ~0.86-0.89% for *either* engine, so compare
   against the fork there rather than against 0.27% (RESULTS §7.9).

## Known limitations at flip time (tracked in PLAN.md M1 remainder)

- `num_speakers` forced counts: native engine logs a warning and runs auto counting
  (min=1/max=20 defaults never bind; constraint port pending).
- Sidecar restart policy covers the upstream ORT-CUDA teardown crash; supervisor recycling
  beyond `restart: unless-stopped` not yet needed in testing.
- VRAM: ~4.2 GB eager / lower with `SPEAKRS_LAZY_SESSIONS=1` (default in the overlay; see
  RESULTS §4.27 for the measured delta).
