"""Convert the missing wespeaker-voxceleb-resnet34-tail-b64.mlmodelc (Apple Silicon).

CoreML counterpart to validation/export_tail_b64_addendum.py. On CoreML,
EmbeddingModel::split_primary_batch_size() gates on
has_native_tail_model(model_path, mode, PRIMARY_BATCH_SIZE=64), i.e. on
fp32_coreml_path(wespeaker-voxceleb-resnet34-tail-b64.onnx) =>
wespeaker-voxceleb-resnet34-tail-b64.mlmodelc existing. scripts/native_coreml/
convert_coreml.py only enumerates TAIL_BATCH_SIZES = (1, 3, 32), so the b64
artifact is never produced and split-primary batching is dead on CoreML.

Deliberately converts a SEPARATE fixed-shape-64 model rather than adding 64 to
TAIL_BATCH_SIZES, so the shipped b1/b3/b32 tail artifacts are not regenerated and
production CoreML accuracy is untouched.

Run on the Apple Silicon box (needs coremltools + the gated pyannote pipeline cache):
    python validation/convert_tail_b64_coreml_addendum.py <models_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "vendor" / "speakrs" / "scripts" / "native_coreml"))

from common import (  # noqa: E402
    FBANK_FEATURES,
    FBANK_FRAMES,
    SEGMENTATION_FRAMES,
    SEGMENTATION_SAMPLES,
    build_fbank_wrapper,
    build_tail_wrapper,
    coreml_packages_dir,
    load_pipeline,
    save_model_artifacts,
)

BATCH = 64
STEM = f"wespeaker-voxceleb-resnet34-tail-b{BATCH}"


def deployment_target() -> object:
    return getattr(ct.target, "macOS15", ct.target.iOS18)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    output_dir = Path(sys.argv[1])

    pipeline = load_pipeline()
    fbank_wrapper = build_fbank_wrapper()
    tail_wrapper = build_tail_wrapper(pipeline)

    dummy_fbank = fbank_wrapper(torch.randn(BATCH, 1, SEGMENTATION_SAMPLES))
    dummy_weights = torch.ones(BATCH, SEGMENTATION_FRAMES)
    with torch.inference_mode():
        traced = torch.jit.trace(tail_wrapper, (dummy_fbank, dummy_weights))

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="fbank",
                shape=(BATCH, FBANK_FRAMES, FBANK_FEATURES),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="weights",
                shape=(BATCH, SEGMENTATION_FRAMES),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT32,
    )

    print(f"Saving {STEM} CoreML artifacts (FP32)...")
    save_model_artifacts(
        mlmodel,
        coreml_packages_dir(output_dir) / f"{STEM}.mlpackage",
        [output_dir / f"{STEM}.mlmodelc"],
    )
    print(f"wrote {output_dir / f'{STEM}.mlmodelc'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
