"""Compare two models directories for equivalence.

Answers the acceptance line "produces a models dir byte-comparable to a current
`models_folded/` export (same file set; ONNX graphs functionally equivalent)" honestly.

The honest reading matters, because a naive `sha256` diff would report failure on a
perfectly correct export. `torch.onnx.export(dynamo=True)` is **not** bit-reproducible: node
names, value-info ordering and graph metadata carry trace-time identifiers that vary between
runs on the same machine with the same seed. What must NOT vary is the numbers.

So three tiers, weakest to strongest:

* **Tier A — inventory.** Same filenames; sizes within +/-1%.
* **Tier B — graph structure and weights.** Per `.onnx`: identical op-type histogram,
  identical initializer count, and EVERY initializer tensor byte-identical after sorting by
  name. The weights come from the same checkpoint, so they MUST match bit-for-bit; only
  names and metadata are allowed to drift. `plda_*.npy` and `min_num_samples.txt` are
  compared byte-for-byte with NO tolerance — those are copied, not exported, so any
  difference is a real one.
* **Tier C — numerical.** 8 seeded fixed inputs through ORT on both graphs; `max|Δ|` must be
  exactly 0.

Tier B is the one that actually bites. A model exported from the wrong checkpoint passes
Tier A comfortably and fails Tier B on the first initializer.

Usage:
    venv/bin/python scripts/compare_model_sets.py models_folded /tmp/models_new
    venv/bin/python scripts/compare_model_sets.py A B --tier c   # include the ORT tier
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter

import numpy as np
import onnx
from onnx import numpy_helper

SIZE_TOLERANCE = 0.01
#: Files that are copied rather than exported, so they must be byte-identical.
EXACT_FILES_SUFFIX = (".npy", ".txt")

#: Fixed-shape probes per graph, keyed by input name. `None` picks a sensible default from
#: the declared shape.
NUM_PROBES = 8


class Result:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  FAIL {msg}")

    def ok(self, msg: str) -> None:
        print(f"  ok   {msg}")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tier_a(a: str, b: str, res: Result) -> list[str]:
    print("\n=== Tier A — inventory ===")
    fa = {n for n in os.listdir(a) if os.path.isfile(os.path.join(a, n))}
    fb = {n for n in os.listdir(b) if os.path.isfile(os.path.join(b, n))}
    # The marker describes the run, not the model, so it is expected to differ.
    fa.discard("diar-provision.json")
    fb.discard("diar-provision.json")

    only_a, only_b = sorted(fa - fb), sorted(fb - fa)
    if only_a:
        res.fail(f"only in {a}: {only_a}")
    if only_b:
        res.fail(f"only in {b}: {only_b}")
    if not only_a and not only_b:
        res.ok(f"same {len(fa)} filenames")

    shared = sorted(fa & fb)
    size_ok = True
    for name in shared:
        sa = os.path.getsize(os.path.join(a, name))
        sb = os.path.getsize(os.path.join(b, name))
        if sa == 0 or abs(sa - sb) / sa > SIZE_TOLERANCE:
            res.fail(f"{name}: size {sa} vs {sb} (>{SIZE_TOLERANCE:.0%})")
            size_ok = False
    if size_ok:
        res.ok(f"all {len(shared)} sizes within {SIZE_TOLERANCE:.0%}")
    return shared


def tier_b(a: str, b: str, shared: list[str], res: Result) -> None:
    print("\n=== Tier B — graph structure + weights ===")
    for name in shared:
        pa, pb = os.path.join(a, name), os.path.join(b, name)

        if name.endswith(EXACT_FILES_SUFFIX):
            ha, hb = sha256(pa), sha256(pb)
            if ha != hb:
                res.fail(f"{name}: sha256 differs ({ha[:12]} vs {hb[:12]}) — copied, not exported, so this is a real difference")
            else:
                res.ok(f"{name}: byte-identical")
            continue

        if not name.endswith(".onnx"):
            continue

        ma, mb = onnx.load(pa), onnx.load(pb)

        oa = Counter(n.op_type for n in ma.graph.node)
        ob = Counter(n.op_type for n in mb.graph.node)
        if oa != ob:
            diff = {k: (oa.get(k, 0), ob.get(k, 0)) for k in set(oa) | set(ob) if oa.get(k) != ob.get(k)}
            res.fail(f"{name}: op histogram differs {diff}")
            continue

        ia = {t.name: t for t in ma.graph.initializer}
        ib = {t.name: t for t in mb.graph.initializer}
        if len(ia) != len(ib):
            res.fail(f"{name}: {len(ia)} initializers vs {len(ib)}")
            continue

        # Compare the MULTISET of tensors, keyed by (dtype, shape, content hash) — never by
        # name and never by sorted-name position.
        #
        # `torch.onnx.export(dynamo=True)` assigns initializer names at TRACE time, so the
        # same weight legitimately carries a different name across two runs: comparing a
        # real second export of wespeaker-fbank.onnx, the shipped graph had {eps, val_45}
        # where the new one had {val_44, val_46}. An earlier version of this function zipped
        # the two name-sorted lists, which silently paired DIFFERENT tensors and reported
        # "13/15 initializers differ, max |Δ| 2.0" for a pair of graphs whose weights are in
        # fact bit-identical. A comparison tool that cries wolf is worse than none, because
        # the next person tunes a tolerance instead of reading it.
        #
        # What genuinely must not change is the multiset of weight tensors, which is exactly
        # what this compares.
        def fingerprints(tensors):
            out = []
            for t in tensors:
                arr = numpy_helper.to_array(t)
                out.append((str(arr.dtype), arr.shape, hashlib.sha256(arr.tobytes()).hexdigest()))
            return sorted(out)

        fa, fb = fingerprints(ia.values()), fingerprints(ib.values())
        if fa != fb:
            only_a = [x for x in fa if x not in fb]
            only_b = [x for x in fb if x not in fa]
            res.fail(
                f"{name}: {len(only_a)}/{len(fa)} initializer tensors differ in CONTENT "
                f"(shapes only in {a}: {[x[1] for x in only_a][:4]}; only in {b}: "
                f"{[x[1] for x in only_b][:4]}). Weights come from the same checkpoint and "
                f"MUST be bit-identical."
            )
        else:
            res.ok(
                f"{name}: {len(oa)} op types, {len(fa)} initializer tensors identical "
                f"(names may differ — dynamo assigns them at trace time)"
            )


def _probe_inputs(sess, rng) -> dict:
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        # Give the sample axis a real length rather than 1 where the graph is dynamic.
        if len(shape) == 3 and shape[-1] == 1:
            shape[-1] = 160000
        feeds[inp.name] = rng.standard_normal(shape).astype(np.float32)
    return feeds


def tier_c(a: str, b: str, shared: list[str], res: Result) -> None:
    print("\n=== Tier C — numerical (ORT) ===")
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for name in shared:
        if not name.endswith(".onnx"):
            continue
        pa, pb = os.path.join(a, name), os.path.join(b, name)
        try:
            sa = ort.InferenceSession(pa, opts, providers=["CPUExecutionProvider"])
            sb = ort.InferenceSession(pb, opts, providers=["CPUExecutionProvider"])
        except Exception as exc:  # a graph that will not load is a Tier B/A problem
            res.fail(f"{name}: could not load ({exc})")
            continue

        worst = 0.0
        rng = np.random.default_rng(0)
        for _ in range(NUM_PROBES):
            feeds = _probe_inputs(sa, rng)
            ra = sa.run(None, feeds)
            rb = sb.run(None, feeds)
            for xa, xb in zip(ra, rb):
                worst = max(worst, float(np.abs(xa - xb).max()))
        if worst != 0.0:
            res.fail(f"{name}: outputs differ, max |Δ| = {worst:.3e} (must be exactly 0)")
        else:
            res.ok(f"{name}: {NUM_PROBES} probes, max |Δ| = 0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tier", choices=("a", "b", "c"), default="b",
                    help="highest tier to run (default b; c also runs ORT probes)")
    args = ap.parse_args()

    res = Result()
    shared = tier_a(args.a, args.b, res)
    if args.tier in ("b", "c"):
        tier_b(args.a, args.b, shared, res)
    if args.tier == "c":
        tier_c(args.a, args.b, shared, res)

    print()
    if res.failures:
        print(f"RESULT: {len(res.failures)} failure(s)")
        for f in res.failures:
            print(f"  - {f}")
        return 1
    print("RESULT: equivalent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
