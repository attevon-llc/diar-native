"""Provisioning driver: the five-step export recipe, in order.

Invoked as a subprocess by `diar-server provision-models`. The Rust side owns everything
around this — token/gate preflight, the smoke test, the marker — so this file does exactly
one job: turn a HuggingFace token into a correct models directory, and report what it used.

## The recipe, and why it is five steps rather than one

`export_models.py` alone does NOT reproduce `models_folded/`. It emits 20 files; the shipped
set has 24, and three of its 20 are subsequently REPLACED. Reconstructed by checksum:

  2a  base export (segmentation, embedding, fbank, multimask, PLDA)     -> 20 files
  2b  constant-fold the 3 segmentation graphs, IN PLACE                 -> replaces 3
  2c  copy multimask-tail-b32 -> multimask-tail-b64 (a COPY, not an export)
  2d  export the genuine batch-64 embedding tail
  2e  export + fp16-quantize the gender classifier, and write its meta sidecar

Step 2c is the counter-intuitive one and is the reason this is written down: a REAL
batch-64 multimask graph under that filename crashes the worker, because speakrs sizes its
multimask runtime buffers for 32 (RESULTS §4.15). The b32 graph under the b64 name is the
fix, not a mistake to be tidied up later.

`--set small` runs all five steps and then DELETES the five fast-only files. That is cheaper
than maintaining a second export path, and it is what guarantees the 19 shared files stay
byte-identical between the two sets — which they are today.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
from typing import Any

import export_gender
import export_models
import export_tail_b64
import fold_segmentation

#: Files present in the fast set but not the small set. Deleted for `--set small`.
FAST_ONLY = (
    "segmentation-3.0-b64.onnx",
    "wespeaker-voxceleb-resnet34-b64.onnx",
    "wespeaker-multimask-tail-b64.onnx",
    "wespeaker-voxceleb-resnet34-tail-b64.onnx",
    "gender-wav2vec2.meta.json",
)


def _version(module: str) -> str | None:
    try:
        return importlib.metadata.version(module)
    except Exception:
        try:
            return getattr(importlib.import_module(module), "__version__", None)
        except Exception:
            return None


def copy_multimask_b64(models_dir: str) -> None:
    """Step 2c — a byte copy, deliberately.

    speakrs' loader requests `wespeaker-multimask-tail-b64.onnx` (PRIMARY_BATCH_SIZE=64)
    while its runtime buffers are sized for 32. A genuine batch-64 graph under that name
    overruns them and the worker dies with "receiver disconnected" — verified, RESULTS
    §4.15. Placing the b32-shaped graph under the b64 filename engages batching and leaves
    the RTTM bit-identical.

    The Rust smoke test asserts these two files have the same sha256 (stage 3d), so this
    cannot be silently "fixed" into a real export later.
    """
    src = os.path.join(models_dir, "wespeaker-multimask-tail-b32.onnx")
    dst = os.path.join(models_dir, "wespeaker-multimask-tail-b64.onnx")
    if not os.path.isfile(src):
        raise RuntimeError(f"{src} is missing — step 2a did not complete")
    shutil.copyfile(src, dst)
    print(f"  wespeaker-multimask-tail-b64.onnx (byte copy of b32, {os.path.getsize(dst) / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--set", dest="model_set", choices=("fast", "small"), default="fast")
    ap.add_argument("--report", help="write a provenance JSON here for the Rust caller")
    ap.add_argument("--skip-gender", action="store_true")
    args = ap.parse_args()

    models_dir = args.models_dir
    os.makedirs(models_dir, exist_ok=True)

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": _version("torch"),
        "torchaudio": _version("torchaudio"),
        "onnx": _version("onnx"),
        "onnxscript": _version("onnxscript"),
        "onnxsim": _version("onnxsim") or _version("onnxslim"),
        "pyannote_audio": _version("pyannote.audio"),
        "transformers": _version("transformers"),
    }

    print("=== 2a: base export ===")
    pipeline = export_models.main(models_dir)

    print("=== 2b: fold segmentation graphs ===")
    report["folder"] = fold_segmentation.fold_all(models_dir)

    print("=== 2c: multimask tail b64 (copy of b32) ===")
    copy_multimask_b64(models_dir)

    print("=== 2d: genuine batch-64 embedding tail ===")
    export_tail_b64.export(pipeline, models_dir)

    if args.skip_gender:
        print("=== 2e: gender classifier SKIPPED (--skip-gender) ===")
        print("  NOTE: without gender-wav2vec2.onnx the sidecar returns no speaker genders.")
    else:
        print("=== 2e: gender classifier ===")
        report.update(export_gender.export(models_dir, write_meta=args.model_set == "fast"))

    # Record the pipeline revision the weights actually came from. The Rust preflight also
    # captures this from `x-repo-commit`; whichever is available wins.
    try:
        from huggingface_hub import model_info

        report["pipeline_revision"] = getattr(
            model_info(export_models.PIPELINE_REPO, token=os.environ.get("HF_TOKEN") or None),
            "sha",
            None,
        )
    except Exception:
        pass

    if args.model_set == "small":
        print("=== trimming to the small set ===")
        for name in FAST_ONLY:
            path = os.path.join(models_dir, name)
            if os.path.isfile(path):
                os.unlink(path)
                print(f"  removed {name}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)

    print("Provisioning export complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
