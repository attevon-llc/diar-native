"""Step 2b: constant-fold the segmentation graphs.

MANDATORY, and until now undocumented as a provisioning step — which is why this file
exists. `export_models.py` alone does not reproduce `models_folded/`.

What folding buys (RESULTS §4.1 / §4.2 / §4.10):

* 144 -> 40 nodes, and every `Sin`, `Cos` and `If` is eliminated. That matters because the
  ORT CUDA EP has no `Sin`/`Cos` kernels, so an unfolded graph silently falls back to CPU
  for those nodes and pays a Memcpy round trip per chunk.
* Segmentation inference roughly 2x faster; end-to-end ES2004a 39.4 s -> 36.7 s with the
  RTTM **bit-identical**.
* Parity is exact, not approximate: max_abs_diff 0.000e+00.

NOT folding is therefore not a "slightly slower" outcome — it is a silent ~2x regression on
segmentation plus the CPU-fallback tax, with no error anywhere.

The folded graphs are written under the PLAIN filenames (`segmentation-3.0.onnx`, not
`-sim.onnx`) because speakrs loads models by filename and has no notion of a folded variant.
Verified: onnxsim's output is BYTE-IDENTICAL (md5) to the shipped `models_folded/` graphs.

## Folder choice, and why there are two

`onnxsim` is preferred and reproduces the shipped bytes exactly. But onnxsim publishes **no
wheel for CPython 3.13** at any version (checked: 0.4.x through 0.7.3 — 13 wheels, zero
cp313), and it is a C++ extension, so on 3.13 `pip install onnxsim` needs cmake and a
toolchain that the OpenTranscribe backend image does not have.

`onnxslim` is the fallback: a pure-python `py3-none-any` wheel that installs anywhere. It is
numerically **bit-exact** as well (measured max_abs_diff 0.0), and it also eliminates every
`Sin`/`Cos`/`If`. It emits a differently-shaped graph though (37 nodes, `MatMul`+`Add` where
onnxsim emits `Gemm`), so a directory folded with onnxslim will NOT be byte-comparable to
`models_folded/`. Which folder ran is recorded in the marker's `toolchain.folder`, because
an unexplained byte difference months later is exactly the kind of thing that costs a day.

Whichever runs, the invariants below are ASSERTED, not assumed.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

import numpy as np
import onnx

#: Written under the plain names. b1 is dynamic in the sample axis; b32/b64 are fixed.
TARGETS = ("segmentation-3.0.onnx", "segmentation-3.0-b32.onnx", "segmentation-3.0-b64.onnx")

#: Ops that MUST be gone after folding. `Sin`/`Cos` have no ORT CUDA kernel; `If` blocks
#: whole-graph fusion and forces a shape-dependent branch at run time.
FORBIDDEN_OPS = ("Sin", "Cos", "If")

#: Generous upper bound. onnxsim gives 40, onnxslim 37; anything near the unfolded 144 means
#: folding did not actually happen.
MAX_FOLDED_NODES = 60


def _load_folder() -> tuple[str, Any]:
    """Return (name, fold_fn). Prefers onnxsim; falls back to onnxslim on CPython 3.13."""
    try:
        from onnxsim import simplify

        def fold_onnxsim(model: onnx.ModelProto) -> onnx.ModelProto:
            folded, ok = simplify(model)
            if not ok:
                raise RuntimeError("onnxsim reported its own validity check as failed")
            return folded

        return "onnxsim", fold_onnxsim
    except ImportError:
        pass

    try:
        import onnxslim

        def fold_onnxslim(model: onnx.ModelProto) -> onnx.ModelProto:
            return onnxslim.slim(model)

        return "onnxslim", fold_onnxslim
    except ImportError as exc:
        raise RuntimeError(
            "No ONNX constant folder available. Install `onnxsim` (preferred; reproduces "
            "the reference bytes exactly, but has no CPython 3.13 wheel) or `onnxslim` "
            "(pure-python wheel, works everywhere, numerically identical but emits a "
            "differently-shaped graph). Folding is NOT optional: an unfolded segmentation "
            "graph is ~2x slower on CUDA and silently falls back to CPU for Sin/Cos."
        ) from exc


def _op_histogram(model: onnx.ModelProto) -> Counter:
    return Counter(n.op_type for n in model.graph.node)


def _numeric_parity(src_path: str, folded_path: str, batch: int, samples: int) -> float:
    """Run both graphs on the same input and return max|Δ|. Expected to be exactly 0.0."""
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.standard_normal((batch, 1, samples)).astype(np.float32)
    opts = ort.SessionOptions()
    # Disable ORT's own optimizer so this compares the two GRAPHS, not ORT's rewriting of
    # them. Without this, ORT would partly fold the unfolded graph and mask a real difference.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    a = ort.InferenceSession(src_path, opts, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    b = ort.InferenceSession(folded_path, opts, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    return float(np.abs(a - b).max())


def fold_all(models_dir: str, verify_numeric: bool = True) -> str:
    """Fold every segmentation graph in place. Returns the folder name for the marker."""
    name, fold = _load_folder()
    print(f"Folding segmentation graphs with {name}...")

    for filename in TARGETS:
        path = os.path.join(models_dir, filename)
        if not os.path.isfile(path):
            # b64 is absent for --set small only AFTER trimming, which happens later; at
            # this point every target should exist.
            raise RuntimeError(f"{path} is missing — step 2a did not complete")

        model = onnx.load(path)
        before = len(model.graph.node)
        folded = fold(model)
        after = len(folded.graph.node)

        ops = _op_histogram(folded)
        present = [op for op in FORBIDDEN_OPS if ops.get(op, 0)]
        if present:
            raise RuntimeError(
                f"{filename}: folding left {present} in the graph. The ORT CUDA EP has no "
                f"Sin/Cos kernel, so this graph would silently run those nodes on CPU. "
                f"Folder={name}."
            )
        if after > MAX_FOLDED_NODES:
            raise RuntimeError(
                f"{filename}: {after} nodes after folding (expected <= {MAX_FOLDED_NODES}). "
                f"Folding did not take effect. Folder={name}."
            )

        # Write to a temp file first so a parity failure cannot leave a half-folded graph
        # under the name speakrs loads.
        staged = path + ".folding"
        onnx.save(folded, staged)

        if verify_numeric and filename == "segmentation-3.0.onnx":
            # Only the batch-1 graph is checked numerically. The fold is the same
            # transformation over the same weights for b32/b64, and running 64 sequences of
            # 4-layer LSTM twice on CPU costs minutes for no additional signal; those two
            # are covered structurally here and by smoke stages 1-2 in Rust.
            diff = _numeric_parity(path, staged, batch=1, samples=160000)
            if diff != 0.0:
                os.unlink(staged)
                raise RuntimeError(
                    f"{filename}: folded graph differs from the original by {diff:.3e}. "
                    f"Folding must be BIT-EXACT (RESULTS §4.1 measured 0.000e+00); a "
                    f"non-zero difference means the folder rewrote semantics, not just "
                    f"constants. Folder={name}."
                )
            print(f"  {filename}: {before} -> {after} nodes, max_abs_diff = {diff:.3e}")
        else:
            print(f"  {filename}: {before} -> {after} nodes, Sin/Cos/If eliminated")

        os.replace(staged, path)

    return name


if __name__ == "__main__":
    import sys

    fold_all(sys.argv[1])
