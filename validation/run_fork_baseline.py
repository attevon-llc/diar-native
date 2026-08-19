#!/usr/bin/env python3
"""Run the pinned pyannote fork (speaker-diarization-community-1) over audio files.

Emits one RTTM per file per run (<label>_run<N>.rttm) plus a timing JSON.
Runs inside the opentranscribe-backend image with the fork bind-mounted, so the
code path is identical to production. Nothing in transcribe-app is modified.

Usage:
  python run_fork_baseline.py --out /work/results/rttm/<tag> \
      [--runs 1] [--device cuda] [--label-mode first-dot] FILES...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def label_for(path: Path, mode: str) -> str:
    if mode == "first-dot":  # EN2002a.Mix-Headset.wav -> EN2002a
        return path.name.split(".")[0]
    return path.stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label-mode", default="stem", choices=["stem", "first-dot"])
    ap.add_argument("--label", default=None, help="Explicit label override (single-file runs)")
    ap.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    args = ap.parse_args()

    import torch
    from pyannote.audio import Audio, Pipeline

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pipeline = Pipeline.from_pretrained(args.model)
    pipeline.to(torch.device(args.device))
    load_s = time.perf_counter() - t0
    print(f"pipeline loaded in {load_s:.1f}s on {args.device}", flush=True)

    def load_wav(path: Path):
        # scipy first for plain WAVs (torchcodec/AudioDecoder is unavailable in this
        # image outside the compose env — same fallback benchmark-pyannote-direct uses).
        if path.suffix.lower() == ".wav":
            import numpy as np
            from scipy.io import wavfile

            sr, data = wavfile.read(str(path))
            if data.dtype == np.int16:
                data = data.astype("float32") / 32768.0
            elif data.dtype == np.int32:
                data = data.astype("float32") / 2147483648.0
            else:
                data = data.astype("float32")
            if data.ndim == 2:  # downmix
                data = data.mean(axis=1)
            wf = torch.from_numpy(data).unsqueeze(0)
            if sr != 16000:
                import torchaudio

                wf = torchaudio.functional.resample(wf, sr, 16000)
                sr = 16000
            return wf, sr
        loader = Audio(sample_rate=16000, mono="downmix")
        return loader({"audio": str(path)})

    records = []
    for f in args.files:
        path = Path(f)
        label = args.label if args.label and len(args.files) == 1 else label_for(path, args.label_mode)
        try:
            waveform, sample_rate = load_wav(path)
        except Exception as e:  # noqa: BLE001 — skip undecodable files, keep sweep going
            print(f"SKIP {label}: audio load failed ({e})", flush=True)
            continue
        duration_s = waveform.shape[1] / sample_rate
        for run_idx in range(args.runs):
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            annotation = getattr(result, "speaker_diarization", result)
            speakers = sorted(annotation.labels())
            peak_vram_mb = (
                torch.cuda.max_memory_allocated() / 1e6 if args.device.startswith("cuda") else 0
            )
            rttm_path = out_dir / f"{label}_run{run_idx}.rttm"
            with rttm_path.open("w") as fh:
                annotation.uri = label
                annotation.write_rttm(fh)
            rec = {
                "label": label,
                "run": run_idx,
                "duration_s": round(duration_s, 1),
                "elapsed_s": round(elapsed, 2),
                "rtf_x": round(duration_s / elapsed, 1),
                "num_speakers": len(speakers),
                "peak_vram_mb": round(peak_vram_mb, 0),
            }
            records.append(rec)
            print(json.dumps(rec), flush=True)

    (out_dir / "timing.json").write_text(
        json.dumps({"device": args.device, "model": args.model, "records": records}, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
