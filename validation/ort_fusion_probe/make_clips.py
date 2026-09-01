"""Regenerate `clips.bin` — the exact 6-clip gate corpus `scripts/provision/export_gender.py` uses.

Same seeds, same lengths, same `do_normalize` preprocessing as `_gate_inputs()` there, so a
number this probe prints is comparable to a number that script's fp16 gate prints. Lengths
bracket `gender.rs`'s real operating range: MIN_SAMPLES (1 s) to the DIAR_GENDER_MAX_SECONDS
default cap (5 s) at 16 kHz.

Written as a flat binary rather than .npy so the Rust probe needs no numpy reader:
    u32 clip_count, then per clip: u32 sample_count, sample_count little-endian f32.

    python validation/ort_fusion_probe/make_clips.py clips.bin
"""

import struct
import sys

import numpy as np

# Must stay in step with FP16_GATE_LENGTHS in scripts/provision/export_gender.py.
LENGTHS = (16_000, 24_000, 32_000, 48_000, 64_000, 80_000)


def main(out: str) -> None:
    with open(out, "wb") as f:
        f.write(struct.pack("<I", len(LENGTHS)))
        for seed, n in enumerate(LENGTHS):
            raw = np.random.default_rng(seed).standard_normal((1, n)).astype(np.float32)
            # Mirrors GenderModel::classify: zero mean, unit variance, eps 1e-7.
            clip = ((raw - raw.mean()) / np.sqrt(raw.var() + 1e-7)).astype(np.float32)
            f.write(struct.pack("<I", n))
            f.write(clip.tobytes())
    print(f"wrote {out}: {len(LENGTHS)} clips, lengths {LENGTHS}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "clips.bin")
