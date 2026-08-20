# Alternative: embed diar-server in the OpenTranscribe backend image

**Status: researched, NOT applied.** This is a reference for whoever edits
`transcribe-app/backend/Dockerfile.prod` — nothing here has been written to that repo. Standard
deployment stays the sidecar in `docker-compose.diar-native.yml` / `docs/INSTALL_NATIVE.md`
until this is deliberately adopted there.

## Why this exists

`diar-native`'s image and OpenTranscribe's backend image each carry a full CUDA/cuDNN runtime
from unrelated sources (backend: `torch`'s pip-bundled `nvidia-*-cu12` wheels; diar-native: apt
packages on an `nvidia/cuda` base) — no automatic overlap, so running both costs disk/registry
space twice for the same GPU dependency. Investigated whether the `diar-server` binary could
instead ride inside the already-built backend image, using its existing CUDA libs, at zero
additional CUDA install cost.

## Compatibility — verified against the live `opentranscribe-backend:latest` image

`diar-server` (built with `--features cuda`, statically linking nothing CUDA-related — it dlopens
ONNX Runtime, which in turn links these at load time) needs exactly 6 shared libs (checked via
`ldd` on `libonnxruntime_providers_cuda.so`, the only EP we use — TensorRT was rolled back,
RESULTS §7.26):

| needed soname | backend has (pip, `torch==2.11.0+cu128`) | path in backend image |
|---|---|---|
| `libcublas.so.12` | `nvidia-cublas-cu12` 12.8.4.1 | `.../nvidia/cublas/lib/` |
| `libcublasLt.so.12` | (same package) | `.../nvidia/cublas/lib/` |
| `libcufft.so.11` | `nvidia-cufft-cu12` 11.3.3.83 | `.../nvidia/cufft/lib/` |
| `libcurand.so.10` | `nvidia-curand-cu12` 10.3.9.90 | `.../nvidia/curand/lib/` |
| `libcudart.so.12` | `nvidia-cuda-runtime-cu12` 12.8.90 | `.../nvidia/cuda_runtime/lib/` |
| `libcudnn.so.9` | `nvidia-cudnn-cu12` 9.19.0.56 | `.../nvidia/cudnn/lib/` |

All 6 sonames match exactly (major version, which is what dynamic linking cares about). Full
paths in the running `opentranscribe-backend:latest` image (`appuser`'s pip user-install):
`/home/appuser/.local/lib/python3.13/site-packages/nvidia/{cublas,cufft,cuda_runtime,cudnn,curand}/lib/`.

**Conclusion: zero additional CUDA packages needed in the backend image.** `diar-server` +
ONNX Runtime's own libs (~375MB: 33MB binary + 342MB ORT `.so`s, unaffected by this change) can
be `COPY`'d in from a multi-stage build referencing the published `davidamacey/diar-native`
image, with `LD_LIBRARY_PATH` extended to those existing pip lib dirs — no `apt-get` CUDA install
in `backend/Dockerfile.prod` at all.

## Sketch (illustrative — not tested end-to-end, not applied)

```dockerfile
# in backend/Dockerfile.prod, alongside the existing deno-bin stage:
FROM davidamacey/diar-native:0.2.0 AS diar-native-bin

# ...in the final backend stage, after the existing COPY --from=builder steps:
COPY --from=diar-native-bin /usr/local/bin/diar-server /usr/local/bin/diar-server
COPY --from=diar-native-bin /usr/local/lib/libonnxruntime*.so* /usr/local/lib/
ENV LD_LIBRARY_PATH="/home/appuser/.local/lib/python3.13/site-packages/nvidia/cublas/lib:\
/home/appuser/.local/lib/python3.13/site-packages/nvidia/cufft/lib:\
/home/appuser/.local/lib/python3.13/site-packages/nvidia/curand/lib:\
/home/appuser/.local/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:\
/home/appuser/.local/lib/python3.13/site-packages/nvidia/cudnn/lib:\
/usr/local/lib:${LD_LIBRARY_PATH}"
```

`diar-server` would then run as a **second process inside the backend container** (e.g. under
`supervisord`/`tini -s`, or as a separate `celery-worker`-side subprocess call), not merged into
the FastAPI/Celery process itself.

## Why this hasn't been recommended as the default — read before adopting

`docs/UPSTREAM_PRS.md` (PR-5/bug-reports) documents a known **ORT-CUDA teardown crash**: a glibc
"corrupted double-linked list" crash at process exit under `cuda` mode. Today's mitigation is
exactly the container boundary — `restart: unless-stopped` on the isolated `diar-native` sidecar
means that crash is contained and auto-recovers without affecting anything else. Running
`diar-server` as a second process sharing the backend container's PID namespace/lifecycle risks
that crash taking the API/Celery process down too, unless whatever supervises the two processes
inside one container gives `diar-server` genuinely independent restart semantics (not just "the
whole container restarts"). That supervision design is unsolved here — worth resolving deliberately
before adopting this, not as a side effect of a disk-space optimization.

## If/when this gets applied to `transcribe-app`

1. Confirm the process-supervision plan for isolated `diar-server` crash recovery inside a shared
   container (this is the actual blocker, not the library compatibility — that part's proven).
2. Use the **slim** `davidamacey/diar-native` image as the `COPY --from=` source (see
   `docker/Dockerfile.server` — base-tier CUDA + only the 6 needed libs, ~3.15GB vs. the prior
   ~5.28GB) so the multi-stage `COPY` itself stays small even though none of its CUDA layer ships
   (only the binary + ORT libs cross over, ~375MB).
3. Keep `DIAR_NATIVE_URL`/HTTP contract identical either way — whether `diar-native` runs as a
   sidecar or embedded, `celery-worker` should keep talking to it over the same
   `/diarize`/`/embed_window`/`/healthz` API, just at `localhost` instead of a service DNS name
   if embedded. That keeps the SpeakerDiarizer integration code unchanged either way.
