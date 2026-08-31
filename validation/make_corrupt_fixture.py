"""Build the weight-corruption fixture for the provisioning smoke test.

Produces a models directory that is byte-valid ONNX everywhere — every graph parses, every
signature is unchanged — but in which one initializer of one graph has been zeroed. That is
the failure the smoke test exists to catch and the one a protobuf parse cannot see: a subtly
wrong export does not fail loudly, it silently degrades diarization.

Only stage 3b (fused `wespeaker-voxceleb-resnet34.onnx` vs split
`wespeaker-fbank.onnx` -> `wespeaker-voxceleb-resnet34-tail.onnx`) can detect this, because
the two paths stop agreeing while each remains individually loadable.

The graphs are hardlinked, so this costs no meaningful disk.

Usage:
    venv/bin/python validation/make_corrupt_fixture.py models_folded /tmp/models_zeroed
    venv/bin/python validation/make_corrupt_fixture.py models_folded /tmp/models_zeroed_mm \\
        wespeaker-multimask-tail-b32.onnx

The optional third argument picks which graph to zero. `wespeaker-multimask-tail-b32.onnx` is
a special case and is handled as one: the b64 file must remain a BYTE COPY of it (stage 3d
asserts that), so both are written with the same corrupted bytes. Corrupting only b32 would
be caught by the hash equality check and would prove nothing about numeric coverage of the
graph production actually executes.
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import onnx
from onnx import numpy_helper

#: Corrupt the split tail by default. It participates in stage 3b (against the fused graph)
#: and 3c (against the multimask tail), so a single edit is visible to two independent checks.
DEFAULT_TARGET = "wespeaker-voxceleb-resnet34-tail.onnx"

#: Graphs that must stay byte-identical to their target. `wespeaker-multimask-tail-b64.onnx`
#: is a byte copy of the b32 graph (RESULTS §4.15) and stage 3d asserts the sha256 equality,
#: so a fixture that corrupted only one of the pair would be caught by the HASH check and the
#: numeric check would never run.
MIRRORS = {
    "wespeaker-multimask-tail-b32.onnx": ["wespeaker-multimask-tail-b64.onnx"],
}


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    target = sys.argv[3] if len(sys.argv) == 4 else DEFAULT_TARGET
    mirrors = MIRRORS.get(target, [])
    rewritten = {target, *mirrors}

    if not os.path.isfile(os.path.join(src, target)):
        print(f"error: {target} is not in {src}")
        return 2

    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    for name in sorted(os.listdir(src)):
        s = os.path.join(src, name)
        if not os.path.isfile(s):
            continue
        d = os.path.join(dst, name)
        if name in rewritten:
            continue
        try:
            os.link(s, d)
        except OSError:
            shutil.copyfile(s, d)

    model = onnx.load(os.path.join(src, target))

    # Zero the largest initializer: the biggest weight tensor is guaranteed to be on the
    # compute path, so the output must change. Zeroing (rather than perturbing) keeps the
    # tensor's dtype, shape and byte length identical, so the protobuf stays exactly as
    # valid as it was.
    biggest = max(model.graph.initializer, key=lambda t: len(t.raw_data) or 1)
    arr = numpy_helper.to_array(biggest)
    zeroed = numpy_helper.from_array(np.zeros_like(arr), biggest.name)
    biggest.CopyFrom(zeroed)

    out = os.path.join(dst, target)
    onnx.save(model, out)
    for mirror in mirrors:
        shutil.copyfile(out, os.path.join(dst, mirror))

    # Prove the premise of the test: the corrupted graph must still LOAD. If it does not,
    # the fixture is testing stage 1 by accident and proves nothing about stage 3.
    import onnxruntime as ort

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    sig_in = [(i.name, i.shape) for i in sess.get_inputs()]
    sig_out = [(o.name, o.shape) for o in sess.get_outputs()]

    print(f"zeroed initializer '{biggest.name}' {arr.shape} {arr.dtype} in {target}")
    print(f"  graph still loads: inputs={sig_in} outputs={sig_out}")
    for mirror in mirrors:
        print(f"  mirrored into {mirror} (kept a byte copy, so stage 3d still passes)")
    print(f"  fixture at {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
