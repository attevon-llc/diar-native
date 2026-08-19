#!/usr/bin/env python3
"""Latency/throughput bench of the Triton spike models (gRPC, batch-32 fixed graphs)."""

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
