#!/usr/bin/env python3
"""Summarize the T1 E2E baseline legs into the RESULTS comparison table.

Reads the per-file CSVs written by ``run_e2e_baseline.sh`` (one directory per engine
configuration) and, when the leg's stdout log is available, the per-step VRAM-profile block
that carries the transcription/diarization split of the last run.

    python summarize_e2e_baseline.py results/e2e_baseline --logs /tmp/leg_%s.log
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

# audio seconds per corpus file, for realtime factors
DURATIONS = {
    "test_ai_video_24s": 24.0,
    "pyramids_239s": 238.8,
    "warpdrive_358s": 358.2,
    "karpathy_3989s": 3989.0,
    "seed_7558s": 7558.0,
}
ORDER = list(DURATIONS)
METRICS = ["total_dispatch_to_postprocess", "gpu_duration", "fully_indexed_duration"]


def read_leg(leg_dir: Path) -> dict[str, dict[str, float]]:
    """Median of each metric per file for one engine configuration."""
    out: dict[str, dict[str, float]] = {}
    for csv_path in leg_dir.glob("*.csv"):
        rows = list(csv.DictReader(csv_path.open()))
        if not rows:
            continue
        out[csv_path.stem] = {
            m: statistics.median(float(r[m]) for r in rows if r.get(m))
            for m in METRICS
            if any(r.get(m) for r in rows)
        }
        out[csv_path.stem]["n_runs"] = float(len(rows))
    return out


def read_steps(log_path: Path) -> dict[str, dict[str, float]]:
    """Per-file transcription/diarization seconds from the leg log's step tables."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="replace")
    steps: dict[str, dict[str, float]] = {}
    current: str | None = None
    for line in text.splitlines():
        header = re.match(r"=== \[[^\]]+\] (\S+) ===", line)
        if header:
            current = header.group(1)
            continue
        step = re.match(r"^(transcription|diarization)\s+([\d.]+)\s+\d+", line.strip())
        if step and current:
            steps.setdefault(current, {})[step.group(1)] = float(step.group(2))
    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="dir holding one subdir per engine config")
    ap.add_argument("--logs", default="", help="printf-style path to leg logs, e.g. /tmp/leg_%%s.log")
    args = ap.parse_args()

    legs = {d.name: read_leg(d) for d in sorted(args.root.iterdir()) if d.is_dir()}
    step_data = {
        name: read_steps(Path(args.logs % name)) if args.logs else {} for name in legs
    }
    if not legs:
        print(f"no leg directories under {args.root}")
        return 1

    names = list(legs)
    print(f"\n## upload -> presented (total_dispatch_to_postprocess, median of N runs)\n")
    header = f"| {'file':18s} | {'audio':>8s} |" + "".join(f" {n:>16s} |" for n in names)
    print(header)
    print("|" + "---|" * (len(names) + 2))
    for f in ORDER:
        if not any(f in legs[n] for n in names):
            continue
        row = f"| {f:18s} | {DURATIONS[f]:7.0f}s |"
        for n in names:
            v = legs[n].get(f, {}).get("total_dispatch_to_postprocess")
            row += f" {v:9.1f}s ({DURATIONS[f] / v:4.0f}x) |" if v else f" {'—':>16s} |"
        print(row)

    for metric in ("gpu_duration", "fully_indexed_duration"):
        print(f"\n## {metric} (median)\n")
        print(header)
        print("|" + "---|" * (len(names) + 2))
        for f in ORDER:
            if not any(f in legs[n] for n in names):
                continue
            row = f"| {f:18s} | {DURATIONS[f]:7.0f}s |"
            for n in names:
                v = legs[n].get(f, {}).get(metric)
                row += f" {v:15.1f}s |" if v else f" {'—':>16s} |"
            print(row)

    if any(step_data.values()):
        print("\n## GPU-stage split, last run of each leg (s)\n")
        print(f"| {'file':18s} |" + "".join(f" {n + ' transcribe':>22s} | {n + ' diarize':>20s} |" for n in names))
        print("|" + "---|" * (2 * len(names) + 1))
        for f in ORDER:
            row = f"| {f:18s} |"
            for n in names:
                s = step_data.get(n, {}).get(f, {})
                t, d = s.get("transcription"), s.get("diarization")
                row += f" {t:21.1f}s |" if t else f" {'—':>22s} |"
                row += f" {d:19.1f}s |" if d else f" {'—':>20s} |"
            print(row)

    print("\n(speedups are audio-seconds / wall-seconds = realtime factor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
