"""Step 2e: export the gender classifier — the 189 MB nobody documented.

This step does not appear in the issue text at all, and omitting it is a SILENT feature
loss, not a loud one. Without `gender-wav2vec2.onnx` in the models dir:
`GenderModel::load_optional` returns `Ok(None)`, `DIAR_GENDER_MAX_SECONDS` becomes inert,
and `diarize_with(..., gender=true)` logs one warning and returns no genders while the
request still succeeds with HTTP 200. That is exactly the failure class provisioning exists
to close.

Provenance (`docs/DETAILED_SPECS.md`): `prithivMLmods/Common-Voice-Gender-Detection`.
Verified UNGATED (`gated: False`) — this step needs NO HuggingFace token, which is why it
still runs when the pipeline gate is the thing that failed.

## Two things that are easy to get wrong here

1. **Tensor names are load-bearing.** `crates/diar-core/src/gender.rs` feeds `"input_values"`
   and reads `"logits"` by name. An export with torch's default generated names loads fine
   and then fails at the first request. They are pinned explicitly below.

2. **The shipped model is FP16, not FP32.** This is not an optimisation you can skip and
   still match: a plain fp32 export is ~361 MB, while the artifact in `models_folded/` is
   189,431,659 bytes — and every one of its 213 initializers is FLOAT16 with `keep_io_types`
   preserving fp32 in and out (confirmed by reading the shipped graph). RESULTS §7.18
   adopted fp16 after checking 67 verdicts across 16 AMI meetings came out 67/67 identical,
   with VRAM 5396 -> 4890 MiB. Note this is the OPPOSITE conclusion to §4.18, where fp16 was
   rejected for the diarization graph — so it must be applied here and only here.

`optimum` is deliberately not used (upstream's documented route was
`optimum-cli export onnx`): it drags in a large dependency tree to wrap the same
`torch.onnx.export` call, and it does not let us pin the I/O names as directly.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch

GENDER_REPO = "prithivMLmods/Common-Voice-Gender-Detection"
MODEL_FILE = "gender-wav2vec2.onnx"
META_FILE = "gender-wav2vec2.meta.json"

#: A 2 s clip at 16 kHz — enough to trace the graph. The exported graph is dynamic in both
#: batch and sample count, matching how `gender.rs` calls it (one variable-length clip).
TRACE_SAMPLES = 32_000

#: RESULTS §7.16 measured 5.96e-06 for the ONNX-vs-torch parity gate with a 1e-4 bar.
PARITY_TOL = 1e-4
#: fp16 loses precision by design; §7.18 measured max Δ 0.0118 on probabilities with zero
#: label flips. The gate that matters is that the ARGMAX never moves.
FP16_PROB_TOL = 0.05


def export(models_dir: str, write_meta: bool = True) -> dict[str, Any]:
    """Export, quantize to fp16, verify, and return provenance for the marker."""
    from huggingface_hub import model_info
    from transformers import AutoConfig, AutoModelForAudioClassification

    print(f"Exporting gender classifier from {GENDER_REPO}...")
    # Ungated, so no token is passed even if one is set.
    info = model_info(GENDER_REPO)
    revision = getattr(info, "sha", None)

    config = AutoConfig.from_pretrained(GENDER_REPO)
    model = AutoModelForAudioClassification.from_pretrained(GENDER_REPO)
    model.eval()

    id2label = {int(k): v for k, v in config.id2label.items()}
    labels = [id2label[i] for i in sorted(id2label)]
    # diar-core compiles ID2LABEL = ["female", "male"] and indexes logits with it directly.
    # If upstream ever reorders these, every verdict inverts silently, so refuse here.
    if labels != ["female", "male"]:
        raise RuntimeError(
            f"{GENDER_REPO} now reports id2label={labels}, but diar-core is compiled with "
            f'["female", "male"] (crates/diar-core/src/gender.rs). Exporting anyway would '
            f"invert every gender verdict. Update the constant and this check together."
        )

    class Wrapper(torch.nn.Module):
        """Strips the HF output dataclass down to the bare logits tensor."""

        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_values: torch.Tensor) -> torch.Tensor:
            return self.inner(input_values=input_values).logits

    wrapper = Wrapper(model)
    wrapper.eval()

    torch.manual_seed(0)
    dummy = torch.randn(1, TRACE_SAMPLES)
    fp32_path = os.path.join(models_dir, MODEL_FILE + ".fp32")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            fp32_path,
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {0: "batch", 1: "samples"},
                "logits": {0: "batch"},
            },
            opset_version=14,
            dynamo=False,
        )

    # Parity gate 1: the fp32 ONNX graph must match torch.
    import onnxruntime as ort

    with torch.no_grad():
        torch_logits = wrapper(dummy).numpy()
    onnx_logits = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"]).run(
        None, {"input_values": dummy.numpy()}
    )[0]
    diff = float(np.abs(torch_logits - onnx_logits).max())
    if diff > PARITY_TOL:
        os.unlink(fp32_path)
        raise RuntimeError(
            f"gender ONNX export disagrees with torch by {diff:.3e} (bar {PARITY_TOL:.0e})"
        )
    print(f"  fp32 parity vs torch: max |logit diff| = {diff:.2e}")

    # fp16 with keep_io_types: the graph runs in half precision while still accepting and
    # returning fp32, so gender.rs needs no change at all.
    import onnx
    from onnxconverter_common import float16

    fp16_model = float16.convert_float_to_float16(
        onnx.load(fp32_path), keep_io_types=True
    )
    out_path = os.path.join(models_dir, MODEL_FILE)
    onnx.save(fp16_model, out_path)

    # Parity gate 2: fp16 must not move the decision. Probabilities may drift; the argmax
    # must not — that is the property §7.18 actually validated (67/67 labels identical).
    fp16_logits = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"]).run(
        None, {"input_values": dummy.numpy()}
    )[0]
    if int(np.argmax(fp16_logits)) != int(np.argmax(onnx_logits)):
        os.unlink(fp32_path)
        raise RuntimeError(
            "fp16 conversion changed the predicted class on the trace input — refusing to "
            "ship a classifier whose verdicts moved."
        )

    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    prob_delta = float(np.abs(softmax(fp16_logits[0]) - softmax(onnx_logits[0])).max())
    if prob_delta > FP16_PROB_TOL:
        os.unlink(fp32_path)
        raise RuntimeError(
            f"fp16 conversion moved probabilities by {prob_delta:.3e} "
            f"(bar {FP16_PROB_TOL}); §7.18 measured 0.0118"
        )
    print(
        f"  fp16 conversion: max Δp = {prob_delta:.2e}, label unchanged, "
        f"{os.path.getsize(fp32_path) / 1e6:.0f} MB -> {os.path.getsize(out_path) / 1e6:.0f} MB"
    )
    os.unlink(fp32_path)

    if write_meta:
        meta = {
            "id2label": {str(k): v for k, v in sorted(id2label.items())},
            "do_normalize": True,
            "sampling_rate": 16000,
        }
        with open(os.path.join(models_dir, META_FILE), "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")

    return {"gender_repo": GENDER_REPO, "gender_revision": revision}
