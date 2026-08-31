"""Step 2a of provisioning: download the community-1 pipeline and export the base graphs.

ADAPTED COPY of `vendor/speakrs/scripts/export_models.py` (Apache-2.0, avencera/speakrs).
See `scripts/provision/UPSTREAM.md` for the vendor pin and the exact diff command. It is a
copy rather than an edit-in-place because editing anything under `vendor/` forces a
regeneration of `patches/0001-cuda-performance-patch-set.patch`, which feeds seven upstream
PR-prep branches scoped to CUDA *performance* work.

Two deliberate divergences from upstream:

1. `export_plda()` is rewritten. See its docstring — the upstream version is a fail-open.
2. `main()` takes an explicit models_dir argument so the driver can import it as a module.

This emits the 20 BASE files. It is NOT the whole recipe: `provision.py` then folds the
segmentation graphs (2b), copies the multimask b64 (2c), exports the real tail-b64 (2d) and
the gender model (2e). Running this file alone does NOT reproduce `models_folded/`.

Gate: https://huggingface.co/pyannote/speaker-diarization-community-1 (CC-BY-4.0,
auto-approved). That single repo is self-contained — do NOT also fetch
`pyannote/segmentation-3.0` or `wespeaker-voxceleb-resnet34-LM`, whose weights DIFFER from
community-1's (proven by checkpoint sha256, RESULTS §1).

Env: HF_TOKEN (never passed on the command line — argv is world-readable via `ps`).
"""

from __future__ import annotations

import os
import sys
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.compliance.kaldi import get_mel_banks

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main(models_dir: str | None = None) -> Any:
    models_dir = models_dir or sys.argv[1]
    token = os.environ.get("HF_TOKEN")
    os.makedirs(models_dir, exist_ok=True)

    print("Loading community-1 pipeline...")
    from pyannote.audio import Pipeline

    from_pretrained: Any = Pipeline.from_pretrained
    if token:
        try:
            pipeline = from_pretrained(
                "pyannote/speaker-diarization-community-1", token=token
            )
        except TypeError:
            try:
                legacy_kwargs: dict[str, Any] = {"use_auth_token": token}
                pipeline = from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    **legacy_kwargs,
                )
            except TypeError:
                pipeline = from_pretrained("pyannote/speaker-diarization-community-1")
    else:
        pipeline = from_pretrained("pyannote/speaker-diarization-community-1")
    # `raise`, not `assert`: `python -O` / PYTHONOPTIMIZE=1 compiles asserts out entirely, so
    # every gate written as one silently ceases to exist. The exporter runs as a subprocess of
    # diar-server, which inherits the environment; a PYTHONOPTIMIZE baked into a base image
    # would have deleted the parity check below without a word. (diar-server now also clears
    # the variable — belt and braces, because this file is runnable by hand too.)
    if pipeline is None:
        raise RuntimeError(
            "Pipeline.from_pretrained returned None for "
            "pyannote/speaker-diarization-community-1. This normally means the token was "
            "accepted but the repo gate has not been, or the cached config is corrupt."
        )
    pipeline.to(torch.device("cpu"))

    export_segmentation(pipeline, models_dir)
    export_embedding(pipeline, models_dir)
    export_plda(models_dir)

    print("Done!")
    return pipeline


def export_segmentation(pipeline: Any, models_dir: str) -> None:
    print("Exporting segmentation model...")
    seg_model = pipeline._segmentation.model
    seg_model.eval()

    dummy = torch.randn(1, 1, 160000)
    with torch.no_grad():
        torch.onnx.export(
            seg_model,
            (dummy,),
            os.path.join(models_dir, "segmentation-3.0.onnx"),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {2: "samples"}, "output": {1: "frames"}},
            opset_version=14,
            dynamo=False,
        )
        torch.onnx.export(
            seg_model,
            (torch.randn(32, 1, 160000),),
            os.path.join(models_dir, "segmentation-3.0-b32.onnx"),
            input_names=["input"],
            output_names=["output"],
            opset_version=14,
            dynamo=False,
        )
        torch.onnx.export(
            seg_model,
            (torch.randn(64, 1, 160000),),
            os.path.join(models_dir, "segmentation-3.0-b64.onnx"),
            input_names=["input"],
            output_names=["output"],
            opset_version=14,
            dynamo=False,
        )

    sz = os.path.getsize(os.path.join(models_dir, "segmentation-3.0.onnx")) / 1e6
    print(f"  segmentation-3.0.onnx ({sz:.1f} MB)")
    bsz = os.path.getsize(os.path.join(models_dir, "segmentation-3.0-b32.onnx")) / 1e6
    print(f"  segmentation-3.0-b32.onnx ({bsz:.1f} MB)")
    b64sz = os.path.getsize(os.path.join(models_dir, "segmentation-3.0-b64.onnx")) / 1e6
    print(f"  segmentation-3.0-b64.onnx ({b64sz:.1f} MB)")


def export_embedding(pipeline: Any, models_dir: str) -> None:
    """Export the exact WeSpeaker embedding path for batch-1 and batch-32 inference"""
    print("Exporting embedding model...")

    class FbankWrapper(nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.scale = float(1 << 15)
            self.preemph = 0.97

            window = torch.hamming_window(400, periodic=False, alpha=0.54, beta=0.46)
            mel, _ = get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)

            self.register_buffer("window", window)
            self.register_buffer("mel", F.pad(mel, (0, 1), value=0.0).T.contiguous())
            self.register_buffer("eps", torch.tensor(torch.finfo(torch.float32).eps))

        def compute_fbank(self, waveforms: torch.Tensor) -> torch.Tensor:
            window = cast(torch.Tensor, self.window)
            mel_filters = cast(torch.Tensor, self.mel)
            eps = cast(torch.Tensor, self.eps)

            frames = waveforms[:, 0, :] * self.scale
            frames = frames.unfold(1, 400, 160)
            frames = frames - frames.mean(dim=2, keepdim=True)

            previous = F.pad(frames, (1, 0), mode="replicate")[..., :-1]
            frames = frames - self.preemph * previous
            frames = frames * window.view(1, 1, -1)
            frames = F.pad(frames, (0, 112))

            spectrum = torch.fft.rfft(frames, dim=2).abs().pow(2.0)
            mel = torch.matmul(spectrum, mel_filters.to(dtype=spectrum.dtype))
            mel = torch.clamp_min(mel, eps.to(device=mel.device, dtype=mel.dtype)).log()
            return mel - mel.mean(dim=1, keepdim=True)

        def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
            return self.compute_fbank(waveforms)

    class EmbeddingTailWrapper(nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.resnet = model.resnet

        def pool(self, sequences: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            weights = weights.unsqueeze(1)
            num_frames = sequences.size(-1)
            if weights.size(-1) != num_frames:
                weights = F.interpolate(weights, size=num_frames, mode="nearest")

            weight_sum = weights.sum(dim=2)
            safe_sum = torch.where(
                weight_sum > 0.0, weight_sum, torch.ones_like(weight_sum)
            )
            mean = torch.sum(sequences * weights, dim=2) / safe_sum
            dx2 = torch.square(sequences - mean.unsqueeze(2))
            weight_sq_sum = torch.square(weights).sum(dim=2)
            denom = safe_sum - weight_sq_sum / safe_sum + 1e-8
            var = torch.sum(dx2 * weights, dim=2) / denom
            std = torch.sqrt(torch.clamp_min(var, 1e-10))

            stats = torch.cat([mean, std], dim=-1)
            zero_stats = torch.cat(
                [torch.zeros_like(mean), torch.full_like(std, 1e-5)], dim=-1
            )
            zero_mask = (weight_sum <= 0.0).repeat(1, stats.size(1))
            return torch.where(zero_mask, zero_stats, stats)

        def forward(self, fbank: torch.Tensor, weights: torch.Tensor) -> Any:
            frames = self.resnet.forward_frames(fbank)
            frames = frames.reshape(
                frames.size(0), frames.size(1) * frames.size(2), frames.size(3)
            )
            stats = self.pool(frames, weights)
            embed_a = self.resnet.seg_1(stats)
            if self.resnet.two_emb_layer:
                out = F.relu(embed_a)
                out = self.resnet.seg_bn_1(out)
                return self.resnet.seg_2(out)

            return embed_a

    NUM_SPEAKERS = 3

    class MultiMaskTailWrapper(nn.Module):
        """fbanks [B, 998, 80] + masks [B*3, 589] -> embeddings [B*3, 256]"""

        def __init__(self, model: Any) -> None:
            super().__init__()
            self.resnet = model.resnet
            self.num_speakers = NUM_SPEAKERS

        def pool(self, sequences: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            weights = weights.unsqueeze(1)
            num_frames = sequences.size(-1)
            if weights.size(-1) != num_frames:
                weights = F.interpolate(weights, size=num_frames, mode="nearest")

            weight_sum = weights.sum(dim=2)
            safe_sum = torch.where(
                weight_sum > 0.0, weight_sum, torch.ones_like(weight_sum)
            )
            mean = torch.sum(sequences * weights, dim=2) / safe_sum
            dx2 = torch.square(sequences - mean.unsqueeze(2))
            weight_sq_sum = torch.square(weights).sum(dim=2)
            denom = safe_sum - weight_sq_sum / safe_sum + 1e-8
            var = torch.sum(dx2 * weights, dim=2) / denom
            std = torch.sqrt(torch.clamp_min(var, 1e-10))

            stats = torch.cat([mean, std], dim=-1)
            zero_stats = torch.cat(
                [torch.zeros_like(mean), torch.full_like(std, 1e-5)], dim=-1
            )
            zero_mask = (weight_sum <= 0.0).repeat(1, stats.size(1))
            return torch.where(zero_mask, zero_stats, stats)

        def forward(self, fbank: torch.Tensor, masks: torch.Tensor) -> Any:
            frames = self.resnet.forward_frames(fbank)
            B = frames.size(0)
            C = frames.size(1) * frames.size(2)
            T = frames.size(3)
            frames = frames.reshape(B, C, T)
            frames = torch.repeat_interleave(frames, self.num_speakers, dim=0)
            stats = self.pool(frames, masks)
            embed_a = self.resnet.seg_1(stats)
            if self.resnet.two_emb_layer:
                out = F.relu(embed_a)
                out = self.resnet.seg_bn_1(out)
                return self.resnet.seg_2(out)
            return embed_a

    emb_model = pipeline._embedding.model_
    emb_model.eval()
    fbank_wrapper = FbankWrapper(emb_model)
    fbank_wrapper.eval()
    tail_wrapper = EmbeddingTailWrapper(emb_model)
    tail_wrapper.eval()
    multi_mask_wrapper = MultiMaskTailWrapper(emb_model)
    multi_mask_wrapper.eval()

    dummy_waveform = torch.randn(1, 1, 160000)
    dummy_weights = torch.ones(1, 589)
    dummy_fbank = fbank_wrapper(dummy_waveform)

    # Verify multi-mask parity with the existing tail wrapper.
    #
    # An explicit `raise`, matching export_tail_b64.py, NOT an `assert`. Under
    # PYTHONOPTIMIZE this whole gate would compile out and the multimask graphs — the ones
    # production executes on every batched job — would be exported with nothing checking they
    # agree with the single-mask tail they are supposed to reproduce.
    MULTI_MASK_PARITY_TOL = 1e-6
    with torch.no_grad():
        tail_output = tail_wrapper(dummy_fbank, dummy_weights)
        multi_masks = dummy_weights.repeat(NUM_SPEAKERS, 1)
        multi_output = multi_mask_wrapper(dummy_fbank, multi_masks)
        max_diff = (tail_output - multi_output[0:1]).abs().max().item()
        if not max_diff < MULTI_MASK_PARITY_TOL:
            raise RuntimeError(
                f"multi-mask parity check failed: max diff = {max_diff:.3e} "
                f"(bar {MULTI_MASK_PARITY_TOL:.0e}). The multi-mask wrapper does not "
                f"reproduce the single-mask tail, so every batched embedding would be "
                f"computed by a graph that disagrees with the reference path. Refusing to "
                f"export."
            )
        print(f"  multi-mask parity check passed (max diff = {max_diff:.2e})")

    with torch.no_grad():
        torch.onnx.export(
            fbank_wrapper,
            (dummy_waveform,),
            os.path.join(models_dir, "wespeaker-fbank.onnx"),
            input_names=["waveform"],
            output_names=["fbank"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
        torch.onnx.export(
            fbank_wrapper,
            (torch.randn(32, 1, 160000),),
            os.path.join(models_dir, "wespeaker-fbank-b32.onnx"),
            input_names=["waveform"],
            output_names=["fbank"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
    fbank_sz = os.path.getsize(os.path.join(models_dir, "wespeaker-fbank.onnx")) / 1e6
    print(f"  wespeaker-fbank.onnx ({fbank_sz:.1f} MB)")
    fbank_b32_sz = (
        os.path.getsize(os.path.join(models_dir, "wespeaker-fbank-b32.onnx")) / 1e6
    )
    print(f"  wespeaker-fbank-b32.onnx ({fbank_b32_sz:.1f} MB)")

    export_embedding_model(
        fbank_wrapper,
        tail_wrapper,
        models_dir,
        "wespeaker-voxceleb-resnet34.onnx",
        batch_size=1,
    )
    export_embedding_model(
        fbank_wrapper,
        tail_wrapper,
        models_dir,
        "wespeaker-voxceleb-resnet34-b32.onnx",
        batch_size=32,
    )
    export_embedding_model(
        fbank_wrapper,
        tail_wrapper,
        models_dir,
        "wespeaker-voxceleb-resnet34-b64.onnx",
        batch_size=64,
    )
    export_embedding_tail_model(
        tail_wrapper,
        models_dir,
        "wespeaker-voxceleb-resnet34-tail.onnx",
        dummy_fbank,
        dummy_weights,
    )
    export_embedding_tail_model(
        tail_wrapper,
        models_dir,
        "wespeaker-voxceleb-resnet34-tail-b3.onnx",
        dummy_fbank.repeat(3, 1, 1),
        dummy_weights.repeat(3, 1),
    )
    export_embedding_tail_model(
        tail_wrapper,
        models_dir,
        "wespeaker-voxceleb-resnet34-tail-b32.onnx",
        dummy_fbank.repeat(32, 1, 1),
        dummy_weights.repeat(32, 1),
    )

    # export multi-mask models
    export_multi_mask_model(
        multi_mask_wrapper,
        models_dir,
        "wespeaker-multimask-tail.onnx",
        dummy_fbank,
        dummy_weights.repeat(NUM_SPEAKERS, 1),
    )
    export_multi_mask_model(
        multi_mask_wrapper,
        models_dir,
        "wespeaker-multimask-tail-b32.onnx",
        dummy_fbank.repeat(32, 1, 1),
        dummy_weights.repeat(32 * NUM_SPEAKERS, 1),
    )

    with open(
        os.path.join(models_dir, "wespeaker-voxceleb-resnet34.min_num_samples.txt"), "w"
    ) as f:
        f.write(f"{pipeline._embedding.min_num_samples}\n")


def export_embedding_model(
    fbank_wrapper: nn.Module,
    tail_wrapper: nn.Module,
    models_dir: str,
    filename: str,
    batch_size: int,
) -> None:
    dummy_waveform = torch.randn(batch_size, 1, 160000)
    dummy_weights = torch.ones(batch_size, 589)
    dummy_fbank = fbank_wrapper(dummy_waveform)

    class ExactEmbeddingWrapper(nn.Module):
        def __init__(self, fbank_model: nn.Module, tail_model: nn.Module) -> None:
            super().__init__()
            self.fbank_model = fbank_model
            self.tail_model = tail_model

        def forward(self, waveforms: torch.Tensor, weights: torch.Tensor) -> Any:
            fbank = self.fbank_model(waveforms)
            return self.tail_model(fbank, weights)

    wrapper = ExactEmbeddingWrapper(fbank_wrapper, tail_wrapper)
    wrapper.eval()
    output_path = os.path.join(models_dir, filename)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_waveform, dummy_weights),
            output_path,
            input_names=["waveform", "weights"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )

    sz = os.path.getsize(output_path) / 1e6
    print(f"  {filename} ({sz:.1f} MB)")


def export_embedding_tail_model(
    wrapper: nn.Module,
    models_dir: str,
    filename: str,
    dummy_fbank: torch.Tensor,
    dummy_weights: torch.Tensor,
) -> None:
    output_path = os.path.join(models_dir, filename)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_fbank, dummy_weights),
            output_path,
            input_names=["fbank", "weights"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )

    sz = os.path.getsize(output_path) / 1e6
    print(f"  {filename} ({sz:.1f} MB)")


def export_multi_mask_model(
    wrapper: nn.Module,
    models_dir: str,
    filename: str,
    dummy_fbank: torch.Tensor,
    dummy_masks: torch.Tensor,
) -> None:
    output_path = os.path.join(models_dir, filename)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_fbank, dummy_masks),
            output_path,
            input_names=["fbank", "masks"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )

    sz = os.path.getsize(output_path) / 1e6
    print(f"  {filename} ({sz:.1f} MB)")


PIPELINE_REPO = "pyannote/speaker-diarization-community-1"

#: The six arrays speakrs' VBx clustering loads, and where each comes from.
#: `xvec_transform.npz` carries the x-vector projection (lda + the two means);
#: `plda.npz` carries the PLDA model itself (mu/psi/tr).
PLDA_SOURCES = {
    "plda/xvec_transform.npz": ("lda", "mean1", "mean2"),
    "plda/plda.npz": ("mu", "psi", "tr"),
}
EXPECTED_PLDA = ("lda", "mean1", "mean2", "mu", "psi", "tr")


def export_plda(models_dir: str) -> None:
    """Extract the PLDA parameters, BY NAME, from the two npz files in the gated repo.

    Rewritten relative to upstream, which was a fail-open in three separate ways:

    1. It hardcoded ``~/.cache/huggingface/hub/...``, so it silently found nothing whenever
       ``HF_HOME``/``HF_HUB_CACHE`` pointed elsewhere — which is the normal case in a
       container, and is exactly how this runs.
    2. It blind-scanned every blob for a ``PK`` magic and loaded whatever it found, so the
       set of arrays written depended on cache layout rather than on the model.
    3. Every failure was swallowed by a bare ``except: pass``, and a missing cache merely
       printed "skipping". A read error therefore produced a models directory with NO PLDA
       files and a zero exit status — the export "succeeded" and clustering was broken.

    Now: resolve the files through ``hf_hub_download`` (honours HF_HOME and the token),
    load them by name, and assert all six arrays were written.
    """
    print("Extracting PLDA params...")
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or None
    written: dict[str, tuple[Any, ...]] = {}

    for repo_file, expected in PLDA_SOURCES.items():
        path = hf_hub_download(PIPELINE_REPO, repo_file, token=token)
        with np.load(path, allow_pickle=False) as data:
            available = set(data.files)
            missing = [n for n in expected if n not in available]
            if missing:
                raise RuntimeError(
                    f"{repo_file} is missing expected array(s) {missing}; "
                    f"it contains {sorted(available)}. The upstream repo layout has "
                    f"changed and the exporter needs updating."
                )
            for name in expected:
                arr = data[name]
                out = os.path.join(models_dir, f"plda_{name}.npy")
                np.save(out, arr)
                written[name] = (arr.shape, arr.dtype)
                print(f"  plda_{name}.npy: shape={arr.shape} dtype={arr.dtype}")

    missing = [n for n in EXPECTED_PLDA if n not in written]
    if missing:
        raise RuntimeError(f"PLDA extraction produced no plda_{missing}.npy — refusing to continue")
    for name in EXPECTED_PLDA:
        out = os.path.join(models_dir, f"plda_{name}.npy")
        if not os.path.isfile(out) or os.path.getsize(out) == 0:
            raise RuntimeError(f"{out} was not written or is empty")


if __name__ == "__main__":
    main()
