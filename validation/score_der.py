#!/usr/bin/env python3
"""Score hypothesis RTTMs against reference RTTMs (DER, collar 0.25, overlap included).

Supports per-file UEM cropping (AMI protocol). Hypothesis files are <label>_run<N>.rttm;
references are <label>.rttm. Reports per-file DER per run, speaker counts, and aggregates.

Usage:
  python score_der.py --ref-dir <dir with <label>.rttm> --hyp-dir <dir with <label>_run<N>.rttm> \
      [--uem-dir <dir with <label>.uem>] [--collar 0.25] [--json-out out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--hyp-dir", required=True)
    ap.add_argument("--uem-dir", default=None)
    ap.add_argument("--collar", type=float, default=0.25)
    ap.add_argument("--skip-overlap", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from pyannote.database.util import load_rttm, load_uem
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref_dir, hyp_dir = Path(args.ref_dir), Path(args.hyp_dir)
    hyp_files = sorted(hyp_dir.glob("*_run*.rttm"))
    if not hyp_files:
        print(f"no hypothesis RTTMs in {hyp_dir}", file=sys.stderr)
        return 1

    by_label: dict[str, list[Path]] = defaultdict(list)
    for h in hyp_files:
        m = re.match(r"(.+)_run(\d+)\.rttm$", h.name)
        if m:
            by_label[m.group(1)].append(h)

    results = []
    agg = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    for label, hyps in sorted(by_label.items()):
        ref_path = ref_dir / f"{label}.rttm"
        if not ref_path.exists():
            print(f"SKIP {label}: no reference {ref_path}", file=sys.stderr)
            continue
        ref_ann = list(load_rttm(str(ref_path)).values())[0]
        uem = None
        if args.uem_dir:
            uem_path = Path(args.uem_dir) / f"{label}.uem"
            if uem_path.exists():
                uem = list(load_uem(str(uem_path)).values())[0]
        for h in sorted(hyps):
            hyp_loaded = load_rttm(str(h))
            if not hyp_loaded:  # empty hypothesis (no speech found)
                from pyannote.core import Annotation

                hyp_ann = Annotation()
            else:
                hyp_ann = list(hyp_loaded.values())[0]
            metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
            der = metric(ref_ann, hyp_ann, uem=uem)
            agg(ref_ann, hyp_ann, uem=uem)
            row = {
                "label": label,
                "hyp": h.name,
                "der_pct": round(100 * der, 3),
                "ref_speakers": len(ref_ann.labels()),
                "hyp_speakers": len(hyp_ann.labels()),
            }
            results.append(row)
            print(
                f"{label:24s} {h.name:36s} DER {row['der_pct']:7.3f}%  "
                f"spk ref={row['ref_speakers']} hyp={row['hyp_speakers']}"
            )

    aggregate = 100 * abs(agg)
    print(f"\nAGGREGATE DER (collar={args.collar}, skip_overlap={args.skip_overlap}): "
          f"{aggregate:.3f}% over {len(results)} scored files")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"aggregate_der_pct": round(aggregate, 3), "rows": results}, indent=2)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
