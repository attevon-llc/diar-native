#!/usr/bin/env python3
"""ORT CUDA EP microbench: folded vs unfolded segmentation graph vs eager torch.

Re-tests the phase-6 claim that ORT CUDA EP is 5.8x slower than eager for the
segmentation model (measured then on an UNFOLDED graph with Sin/Cos/If on CPU).
Runs inside the opentranscribe-backend image (onnxruntime-gpu 1.28).
"""

from __future__ import annotations

import sys
import time

import numpy as np

BATCH, SAMPLES, WARMUP, ITERS = 32, 160000, 5, 20


def bench_ort(path: str) -> None:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 2  # warnings: shows CPU-fallback / Memcpy insertion
    sess = ort.InferenceSession(so=so, path_or_bytes=path,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    x = np.random.randn(BATCH, 1, SAMPLES).astype(np.float32)
    for _ in range(WARMUP):
        sess.run(None, {input_name: x})
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = sess.run(None, {input_name: x})
    dt = (time.perf_counter() - t0) / ITERS * 1000
    print(f"ORT-CUDA {path.split('/')[-1]:36s} {dt:8.2f} ms/batch  out={out[0].shape}")
    return out[0]


def bench_torch() -> None:
    import torch
    from pyannote.audio import Model

    model = Model.from_pretrained(
        "pyannote/speaker-diarization-community-1", subfolder="segmentation"
    ).eval().to("cuda")
    x = torch.randn(BATCH, 1, SAMPLES, device="cuda")
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            model(x)
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS * 1000
    print(f"torch-eager segmentation (community-1)    {dt:8.2f} ms/batch")


def parity(a_path: str, b_path: str) -> None:
    import onnxruntime as ort

    x = np.random.randn(BATCH, 1, SAMPLES).astype(np.float32)
    outs = []
    for p in (a_path, b_path):
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        outs.append(sess.run(None, {sess.get_inputs()[0].name: x})[0])
    diff = np.abs(outs[0] - outs[1]).max()
    argmax_mismatch = (outs[0].argmax(-1) != outs[1].argmax(-1)).mean()
    print(f"parity folded-vs-unfolded: max_abs_diff={diff:.3e} argmax_mismatch={argmax_mismatch:.6f}")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/work/models"
    parity(f"{base}/segmentation-3.0-b32.onnx", f"{base}/segmentation-3.0-b32-sim.onnx")
    bench_ort(f"{base}/segmentation-3.0-b32.onnx")
    bench_ort(f"{base}/segmentation-3.0-b32-sim.onnx")
    bench_torch()
