# Troubleshooting

The things that actually go wrong, in rough order of how often they do. For the one-command
install, start at the [README](../README.md).

Exit codes are a stable contract and are tabulated in
[DEPLOYMENT.md](DEPLOYMENT.md#exit-codes).

---

## 1. The token is rejected → HTTP 401, exit 5

```
error: Your HuggingFace token was rejected (HTTP 401).
```

It is not a read token, it was revoked, or it was pasted with a stray character. Make a fresh
**read** token at <https://huggingface.co/settings/tokens> and re-check with
`diar-server check-token` (~200 ms, no download).

The token belongs to *you* — this project has no token of its own to fall back on.

## 2. The terms were never accepted → HTTP 403, exit 5

This one is sneaky, because the token is perfectly valid and so it *feels* like it should work.
The gate is **per-account**: accepting the terms while signed in as a different HuggingFace
account than the one that issued the token fails in exactly the same way.

Fix: visit <https://huggingface.co/pyannote/speaker-diarization-community-1> **signed in as the
token's own account**, accept, and re-run. It is auto-approved — no waiting list, no human
review.

## 3. `models directory is not writable` → exit 7

```
error: the models directory is not writable (exit 7)
```

You ran the export against the read-only mount. The serving service mounts `/models:ro` on
purpose; only the `provision` service mounts it read-write. Use
`docker compose --profile provision run --rm provision`, or `./start.sh --provision`, rather than
`docker compose exec diar-native provision-models`.

The check happens **before** the export, not after 484 MB of work.

If the directory is genuinely owned by someone else, see
[DEPLOYMENT.md](DEPLOYMENT.md#the-container-user) — and note you should never need a `chown`, and
must never chown to `10001`.

## 4. `No python interpreter at 'python3'` → exit 6

You asked a **serving** image to provision. None of the serving images contain Python — that is
why the CPU one is 195 MB rather than ~2 GB. Use the `-provision` image tag. See
[PROVISIONING.md](PROVISIONING.md#you-need-a-python-interpreter-but-only-for-this).

Exit 6 means "install torch into the exporter". It is deliberately distinct from exit **8**,
which means "the models directory is too broken to serve against" — they were one code before
0.3.0 and a supervisor could not tell them apart.

## 5. No GPU, or a GPU Docker cannot reach

A working `nvidia-smi` on the host is **not sufficient**. The NVIDIA container runtime must also
be installed and registered with Docker. That second check is the one people forget: a host can
have a perfectly good driver and still be unable to pass the GPU into a container.

`install.sh` verifies both (it actually runs a throwaway container with `--gpus all`) and falls
back to the CPU image if either is missing. Pass `--gpu` to make that failure **loud** instead of
falling back. Install `nvidia-container-toolkit`, restart Docker, and re-run.

The CPU path is fully supported and produces the same output — only slower. Forcing
`DIAR_DEVICES=cuda` against the CPU image is fatal at startup, deliberately: a diarizer that
quietly falls back to the CPU is a performance regression nobody notices.

## 6. It runs, but it is inexplicably slow on arm64

You are almost certainly running an **amd64 image under emulation**. `:latest` and the bare
`:<ver>` tag are the CUDA image and are **linux/amd64 only, permanently** — Docker Desktop will
emulate them rather than refuse them, so there is no error to act on.

Name the `-arm64` tags instead. See
[DEPLOYMENT.md](DEPLOYMENT.md#on-macos-and-arm64-linux).

And be clear-eyed about the ceiling even when the tag is right: on Apple Silicon under Docker you
get **CPU cores**, not the GPU and not the Neural Engine. Docker on macOS has no Metal access at
any image architecture.

## 7. Gender is silently missing

Two different causes, and `/healthz` tells them apart via `models_gender`:

- **The file is not there.** Gender is enabled by file presence, so a `--skip-gender` deployment
  answers `diarize(gender=true)` with a 200 and no genders. Re-provision without `--skip-gender`.
- **You are on linux/arm64 with an old build.** The fp16 gender model used to fail to load there
  while the server still answered 200. This is **fixed** — see
  [DEPLOYMENT.md](DEPLOYMENT.md#the-fp16-gender-model-on-linuxarm64) — but it is the first thing
  to check on an older image.

## 8. `/readyz` returns 503 but the service answers requests

That is the design. **`/healthz` is 200 in every state while the process is serving** — it is
liveness, and the compose healthcheck gates on it. `/readyz` is 503 until
`models_state == "verified"`, so `stale` and `unverified` both show 503 while the server works
normally.

Read `models_reason` on either endpoint: it carries a human sentence **plus the remediation
command** for every non-verified state.

## 9. The `device` field appears to be ignored

You are talking to an **old** diar-server. Neither request struct uses `deny_unknown_fields`, so
an old server does not reject `{"device":"cpu"}` — it *ignores* it, runs the job on CUDA anyway,
and returns 200. Serde cannot help you here.

Gate on `/healthz` `supported_devices` (or on the presence of the `x-diar-device` response
header) before relying on the field. On a **new** server an unknown device name is a 400 that
names the devices the build serves. See [API.md](API.md#selecting-a-device).

## 10. Provisioning dies with an `httpx.UnsupportedProtocol` error

```
httpx.UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol
```

It reads like a network fault and is not one. **`HF_ENDPOINT` is set to the empty string.** The
Rust side treats empty as unset, but it does not *strip* the variable, and the Python export child
inherits it — `huggingface_hub` does `os.environ.get("HF_ENDPOINT", "https://…")`, which returns
the empty string when the key exists and is blank, so the download URL loses its scheme.

Easy to hit from compose, where `HF_ENDPOINT: ${HF_ENDPOINT:-}` is the natural way to make a
variable optional. Either leave it unset entirely or give it a real value (RESULTS §7.43).

## 11. Every request hangs forever

Check `DIAR_MAX_INFLIGHT`. **`0` is accepted and deadlocks every request** — the global admission
gate has no `> 0` guard, unlike `DIAR_MAX_INFLIGHT_CPU`, which explicitly treats `0` as unset for
exactly this reason.

## 12. The process aborts at exit, after doing its work

You set `RUST_LOG=trace`. Enabling `ort::lifetime` at TRACE makes ort's `.fini_array` destructor
log the drop of its global `Environment` after the main thread's TLS is gone, which aborts the
process — *after* the output is written and flushed. In a container the binary is PID 1, so this
shows up as exit 139 rather than 134.

`RUST_LOG=speakrs=trace` is the safe way to get engine stage timings and does **not** trigger it.
Full story: [`ORT_ATEXIT_TEARDOWN.md`](ORT_ATEXIT_TEARDOWN.md).

## 13. A clean clone will not build

Run `./scripts/bootstrap_vendor_speakrs.sh`. `vendor/speakrs/` is gitignored and is not a
submodule, so a fresh clone has nothing there. See [DEVELOPMENT.md](DEVELOPMENT.md).

## 14. Logs are empty, or drowning in ONNX Runtime noise

**Unset `RUST_LOG` does not mean silent** — the default is `info,ort::logging=warn`, and the
container is useful out of the box. A malformed value warns and falls back to that same default
rather than starting the process blind.

ORT's native log bridge emits **5812 INFO lines on a CUDA startup** against 3 lines from
diar-server, which is why that one target is held at `warn` by default. `RUST_LOG=ort=info` or
`RUST_LOG=debug` gets the firehose back. See
[CONFIGURATION.md](CONFIGURATION.md#logging).

---

## Getting more detail

```bash
docker compose logs -f                       # follow
curl -s localhost:8701/healthz | jq          # model state + reason, always 200
diar-server verify-models --models-dir /models   # deep check: full sha256 + smoke test
```

Every response carries an `x-request-id` — on errors too — so a failure can be matched to its
server-side record without guessing. Send your own inbound `x-request-id` to keep one id across a
larger stack.

Remember what verification does **not** prove: it checks that the models are *usable*, not that
they are *accurate*. See [PROVISIONING.md](PROVISIONING.md#what-verification-does-not-cover).

---

See also: [DEPLOYMENT.md](DEPLOYMENT.md) · [PROVISIONING.md](PROVISIONING.md) ·
[CONFIGURATION.md](CONFIGURATION.md) · [API.md](API.md) · [README](../README.md)
