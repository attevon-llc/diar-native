"""Report what ORT's optimizer rewrote each graph into.

Feed it a directory of optimized graphs written by `ort-fusion-probe`. For each it prints the
node count, how many `Erf` nodes survived (the GELU pattern issue #14 is about), which
`com.microsoft::*` contrib ops the optimizer created, and the initializer dtypes.

The dtype column is the load-bearing one. A graph is exposed to this whole class of bug when
the optimizer rewrites it into a contrib op AND its tensors are fp16, because the aarch64
builds carry fp32-only kernels for those ops (§7.40: `Gelu` and `FusedConv` are both
`<float>`-only). 11 of 15 diarization graphs get `FusedConv` and are safe purely by being
fp32 — not by immunity.

    python validation/ort_fusion_probe/inspect_dumps.py <dumpdir>
"""

import collections
import glob
import os
import sys

import onnx

DTYPE = {1: "FLOAT", 7: "INT64", 10: "FLOAT16", 11: "DOUBLE"}


def main(dump_dir: str) -> None:
    paths = sorted(glob.glob(os.path.join(dump_dir, "*.onnx")))
    if not paths:
        sys.exit(f"no .onnx dumps in {dump_dir}")
    for p in paths:
        m = onnx.load(p)
        ops = collections.Counter((n.domain or "ai.onnx") + "::" + n.op_type for n in m.graph.node)
        contrib = {k.split("::")[1]: v for k, v in ops.items() if k.startswith("com.microsoft")}
        dts = collections.Counter(DTYPE.get(i.data_type, i.data_type) for i in m.graph.initializer)
        name = os.path.basename(p)
        print(
            f"{name:52s} nodes={len(m.graph.node):5d} Erf={ops.get('ai.onnx::Erf', 0):3d} "
            f"contrib={contrib or '{}'} init_dtypes={dict(dts)}"
        )
        if contrib and "FLOAT16" in dts:
            print(f"{'':52s} ^^ EXPOSED: contrib op on fp16 tensors — check it has an fp16 kernel")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
