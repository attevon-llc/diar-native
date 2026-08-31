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

2. **The shipped model is FP16, not FP32** — which the documented `optimum-cli` route does
   not tell you. A plain fp32 export is ~378 MB, while the artifact in `models_folded/` is
   189,431,659 bytes, and every one of its 213 initializers is FLOAT16 with `keep_io_types`
   preserving fp32 in and out (confirmed by reading the shipped graph). RESULTS §7.18
   adopted fp16 after 67 verdicts across 16 AMI meetings came out 67/67 identical, with VRAM
   5396 -> 4890 MiB. Note this is the OPPOSITE conclusion to §4.18, where fp16 was rejected
   for the diarization graph.

   The conversion is **best-effort with a validated fp32 fallback** — both precisions are
   functionally correct, so a failed conversion degrades rather than aborts and the precision
   actually produced is reported back for the marker. It does succeed on torch 2.13 once the
   exporter's no-op `Cast` nodes are elided first; see `_elide_identity_casts`.

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

#: Clip lengths the fp16 gate is evaluated over, in samples at 16 kHz. These bracket
#: `gender.rs`'s real operating range: `MIN_SAMPLES` (1 s) to the `DIAR_GENDER_MAX_SECONDS`
#: default cap (5 s). A single trace-length input cannot catch a length-dependent conversion
#: fault, and §7.18's original 67-verdict AMI sweep was ad hoc and left nothing runnable.
FP16_GATE_LENGTHS = (16_000, 24_000, 32_000, 48_000, 64_000, 80_000)


def _elide_identity_casts(model: Any) -> tuple[Any, list[str]]:
    """Drop `Cast` nodes whose `to` already equals their input's element type.

    torch 2.13 emits two such no-op casts in wav2vec2 that torch 2.11 did not — one
    `Cast(to=FLOAT)` on an already-fp32 `Transpose` output feeding `encoder/Add`, and one
    `Cast(to=INT64)` on an already-int64 value. They are semantically invisible in fp32,
    which is why nothing noticed them, but they break fp16 conversion: `onnxconverter_common`
    treats every `Cast` as an authoritative precision boundary (casts are the mechanism it
    inserts itself to bridge fp16/fp32), so it retypes the value to fp16 while leaving
    `to=FLOAT` alone. ORT then rejects the model — as a `Cast` type error with shape
    inference on, or, with `disable_shape_infer=True`, as the same contradiction surfacing
    one node later at the `Add` the cast feeds.

    Elision is a pure no-op on the fp32 graph, and the caller proves that bitwise rather than
    trusting it. `onnx.shape_inference` supplies the input types; a cast whose input type is
    unknown is left alone.
    """
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model)
    types: dict[str, int] = {}
    for vi in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output):
        if vi.type.HasField("tensor_type"):
            types[vi.name] = vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        types[init.name] = init.data_type
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    types[node.output[0]] = attr.t.data_type

    outputs = {o.name for o in model.graph.output}
    removed: list[str] = []
    rewire: dict[str, str] = {}
    kept = []
    for node in model.graph.node:
        if node.op_type == "Cast":
            to = next((a.i for a in node.attribute if a.name == "to"), None)
            # Follow chains: an elided cast's input is what a later cast really reads.
            src = rewire.get(node.input[0], node.input[0])
            # Renaming a graph output would break the caller's `outputs["logits"]` lookup.
            if to is not None and types.get(src) == to and node.output[0] not in outputs:
                rewire[node.output[0]] = src
                removed.append(f"{node.name} (to={to})")
                continue
        kept.append(node)

    del model.graph.node[:]
    model.graph.node.extend(kept)
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in rewire:
                node.input[i] = rewire[name]
    # Inferred types cached on the graph would now contradict the rewiring.
    del model.graph.value_info[:]
    return model, removed


def _gate_inputs() -> list[Any]:
    """Seeded, `do_normalize`-preprocessed clips spanning gender.rs's real clip lengths."""
    clips = []
    for seed, n in enumerate(FP16_GATE_LENGTHS):
        raw = np.random.default_rng(seed).standard_normal((1, n)).astype(np.float32)
        # Mirrors GenderModel::classify: zero mean, unit variance, eps 1e-7.
        clips.append(((raw - raw.mean()) / np.sqrt(raw.var() + 1e-7)).astype(np.float32))
    return clips


def export(models_dir: str, write_meta: bool = True) -> dict[str, Any]:
    """Export, quantize to fp16, verify, and return provenance for the marker."""
    from huggingface_hub import model_info
    from transformers import AutoConfig, AutoModelForAudioClassification

    print(f"Exporting gender classifier from {GENDER_REPO}...")
    # Ungated, so no token is passed even if one is set. The revision is provenance for the
    # marker, not a requirement — provisioning from a warm cache with HF_HUB_OFFLINE=1 is a
    # legitimate mode (air-gapped hosts, and re-exports that must not re-download), so a
    # failure to reach the API must not fail the export.
    try:
        revision = getattr(model_info(GENDER_REPO), "sha", None)
    except Exception as exc:
        print(f"  (could not resolve upstream revision: {exc})")
        revision = None

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
            opset_version=17,
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

    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    # fp16 with keep_io_types: the graph runs in half precision while still accepting and
    # returning fp32, so gender.rs needs no change at all.
    #
    # Still BEST-EFFORT with a validated fp32 fallback. fp16 is a real but non-blocking win —
    # §7.18 measured VRAM 5396 -> 4890 MiB and disk 361 -> 181 MB with 67/67 labels identical
    # — so failing the whole provisioning run over it would trade a working deployment for a
    # smaller file. The fp32 model is correct and is what `gender.rs` loads either way; only
    # VRAM and disk differ. Which precision was produced is REPORTED for the marker.
    #
    # The blocker on torch 2.13 was two no-op `Cast` nodes; `_elide_identity_casts` explains
    # the mechanism. Everything below the elision is unchanged in intent, only widened from
    # one trace input to FP16_GATE_LENGTHS.
    out_path = os.path.join(models_dir, MODEL_FILE)
    precision = "fp32"

    import onnx

    try:
        from onnxconverter_common import float16

        elided, removed = _elide_identity_casts(onnx.load(fp32_path))
        elided_path = out_path + ".elided"
        onnx.save(elided, elided_path)

        # Gate 2: the elision must be invisible. Casts are load-bearing when they are real,
        # so prove these were not — bitwise, over every gate clip, before converting.
        gate_clips = _gate_inputs()
        base = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        elided_sess = ort.InferenceSession(elided_path, providers=["CPUExecutionProvider"])
        fp32_logits = [base.run(None, {"input_values": c})[0] for c in gate_clips]
        for clip, want in zip(gate_clips, fp32_logits):
            got = elided_sess.run(None, {"input_values": clip})[0]
            if not np.array_equal(got, want):
                raise RuntimeError(
                    f"eliding {len(removed)} no-op Cast(s) changed the fp32 output by "
                    f"{np.abs(got - want).max():.3e} — not a no-op, refusing to convert"
                )
        del base, elided_sess
        print(
            f"  elided {len(removed)} no-op Cast node(s) [{', '.join(removed)}]; "
            f"fp32 output bitwise unchanged on {len(gate_clips)} clips"
        )

        fp16_path = out_path + ".fp16"
        onnx.save(
            float16.convert_float_to_float16(onnx.load(elided_path), keep_io_types=True),
            fp16_path,
        )
        fp16_sess = ort.InferenceSession(fp16_path, providers=["CPUExecutionProvider"])

        # Gate 3: fp16 must not move the decision. Probabilities may drift; the argmax must
        # not — that is the property §7.18 actually validated.
        prob_delta = 0.0
        logit_delta = 0.0
        for i, (clip, want) in enumerate(zip(gate_clips, fp32_logits)):
            got = fp16_sess.run(None, {"input_values": clip})[0]
            if int(np.argmax(got)) != int(np.argmax(want)):
                raise RuntimeError(
                    f"fp16 conversion changed the predicted class on clip {i} "
                    f"({len(clip[0])} samples)"
                )
            logit_delta = max(logit_delta, float(np.abs(got - want).max()))
            prob_delta = max(
                prob_delta, float(np.abs(softmax(got[0]) - softmax(want[0])).max())
            )
        if prob_delta > FP16_PROB_TOL:
            raise RuntimeError(
                f"fp16 moved probabilities by {prob_delta:.3e} (bar {FP16_PROB_TOL})"
            )
        del fp16_sess

        os.replace(fp16_path, out_path)
        precision = "fp16"
        print(
            f"  fp16 conversion: {len(gate_clips)}/{len(gate_clips)} labels unchanged, "
            f"max Δlogit = {logit_delta:.2e}, max Δp = {prob_delta:.2e}, "
            f"{os.path.getsize(fp32_path) / 1e6:.0f} MB -> "
            f"{os.path.getsize(out_path) / 1e6:.0f} MB"
        )
    except Exception as exc:
        print(
            f"  fp16 conversion unavailable ({str(exc)[:160]}); keeping fp32. "
            f"The classifier is correct either way — it costs ~500 MiB more VRAM and "
            f"~190 MB more disk than the fp16 build (RESULTS §7.18)."
        )
    finally:
        for stale in (out_path + ".fp16", out_path + ".elided"):
            if os.path.isfile(stale):
                os.unlink(stale)

    if precision == "fp32":
        os.replace(fp32_path, out_path)
    else:
        os.unlink(fp32_path)

    print(f"  {MODEL_FILE}: {os.path.getsize(out_path) / 1e6:.0f} MB ({precision})")

    if write_meta:
        meta = {
            "id2label": {str(k): v for k, v in sorted(id2label.items())},
            "do_normalize": True,
            "sampling_rate": 16000,
        }
        with open(os.path.join(models_dir, META_FILE), "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")

    return {
        "gender_repo": GENDER_REPO,
        "gender_revision": revision,
        "gender_precision": precision,
    }
