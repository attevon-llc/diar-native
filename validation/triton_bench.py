#!/usr/bin/env python3
"""Latency/throughput bench of the Triton spike models (gRPC, batch-32 fixed graphs).

STATUS: FUTURE WORK, not part of the current pipeline. Kept deliberately.

TensorRT-in-ort was implemented, measured and ROLLED BACK (RESULTS §7.26): 1.33-1.48x warm
diarization for +0.030 pp AMI DER, judged not worth the compatibility surface for speed that
hides behind transcription anyway. `docker/Dockerfile.server` deletes
`libonnxruntime_providers_tensorrt.so`, and `SPEAKRS_TRT`/`SPEAKRS_TRT_CACHE` have no read
sites.

None of that closes the door. Triton (deployment tier T2) remains the intended path for
multi-user / multi-job serving — measured 2.14x throughput at 8 concurrent clients on one weight
copy — and running TensorRT locally is a separate question from embedding the EP in `ort`. This
harness is what measured those numbers; deleting it would mean re-deriving the setup rather than
re-running it.

Read before reusing:
  * RESULTS §7.26  — the TensorRT rollback, with the recipe if it is revisited
  * docs/ASR_TRITON_NOTES.md
  * docs/DETAILED_SPECS.md  — the T2 Triton design (S-T11)
  * triton/  — the model repository layout from the original spike
"""

from __future__ import annotations

import sys
import time

import numpy as np
import tritonclient.grpc as grpcclient

URL = sys.argv[1] if len(sys.argv) > 1 else "localhost:8611"
WARMUP, ITERS = 5, 20


def bench(client: grpcclient.InferenceServerClient, model: str, inputs: dict[str, np.ndarray]) -> None:
    infer_inputs = []
    for name, arr in inputs.items():
        ii = grpcclient.InferInput(name, arr.shape, "FP32")
        ii.set_data_from_numpy(arr)
        infer_inputs.append(ii)
    for _ in range(WARMUP):
        client.infer(model, infer_inputs)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        res = client.infer(model, infer_inputs)
    dt = (time.perf_counter() - t0) / ITERS * 1000
    out = res.as_numpy(res.get_response().outputs[0].name)
    print(f"{model:14s} {dt:8.2f} ms/req (batch 32, incl. gRPC transfer)  out={out.shape}")


def main() -> None:
    client = grpcclient.InferenceServerClient(URL)
    rng = np.random.default_rng(0)
    wave = rng.standard_normal((32, 1, 160000), dtype=np.float32)
    weights = np.ones((32, 589), dtype=np.float32)
    bench(client, "seg_unfolded", {"input": wave})
    bench(client, "seg_folded", {"input": wave})
    bench(client, "embedding", {"waveform": wave, "weights": weights})


if __name__ == "__main__":
    main()
