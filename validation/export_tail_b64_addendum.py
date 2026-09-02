"""Export the missing wespeaker-voxceleb-resnet34-tail-b64.onnx.

Same bug class as the missing multimask b64 tail (RESULTS §4.15), but NOT the same
fix: that one ships as a byte COPY of the b32 graph, because speakrs sizes its
multimask runtime buffers for 32 and a genuine batch-64 graph under that filename
crashes the worker. This file is a REAL batch-64 export. See
scripts/provision/export_tail_b64.py, which contrasts the two treatments in full.

speakrs' loader asks for
split_tail_model_path(model_path, PRIMARY_BATCH_SIZE=64) =>
wespeaker-voxceleb-resnet34-tail-b64.onnx (load/sessions.rs:70), and on CoreML for
the .mlmodelc compiled from it (embedding/native/loaders.rs::load_native_tail).
scripts/export_models.py only emits the b1/b3/b32 tails, so
EmbeddingModel::split_primary_batch_size() returns 0 on every model set we ship and
the split-primary batching path (EmbeddingPath::Split) is silently never taken.

EmbeddingTailWrapper is copied verbatim from vendor/speakrs/scripts/export_models.py
(Apache-2.0, avencera/speakrs) so the b64 graph is bit-consistent with the shipped
b1/b3/b32 tails.

Usage: python validation/export_tail_b64_addendum.py <models_dir> [<models_dir> ...]
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

BATCH = 64
FBANK_FRAMES = 998
FBANK_FEATURES = 80
MASK_FRAMES = 589


class EmbeddingTailWrapper(nn.Module):
    """fbank [B, 998, 80] + weights [B, 589] -> embeddings [B, 256]"""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.resnet = model.resnet

    def pool(self, sequences: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.unsqueeze(1)
        num_frames = sequences.size(-1)
        if weights.size(-1) != num_frames:
            weights = F.interpolate(weights, size=num_frames, mode="nearest")

        weight_sum = weights.sum(dim=2)
        safe_sum = torch.where(weight_sum > 0.0, weight_sum, torch.ones_like(weight_sum))
        mean = torch.sum(sequences * weights, dim=2) / safe_sum
        dx2 = torch.square(sequences - mean.unsqueeze(2))
        weight_sq_sum = torch.square(weights).sum(dim=2)
        denom = safe_sum - weight_sq_sum / safe_sum + 1e-8
        var = torch.sum(dx2 * weights, dim=2) / denom
        std = torch.sqrt(torch.clamp_min(var, 1e-10))

        stats = torch.cat([mean, std], dim=-1)
        zero_stats = torch.cat([torch.zeros_like(mean), torch.full_like(std, 1e-5)], dim=-1)
        zero_mask = (weight_sum <= 0.0).repeat(1, stats.size(1))
        return torch.where(zero_mask, zero_stats, stats)

    def forward(self, fbank: torch.Tensor, weights: torch.Tensor) -> Any:
        frames = self.resnet.forward_frames(fbank)
        frames = frames.reshape(frames.size(0), frames.size(1) * frames.size(2), frames.size(3))
        stats = self.pool(frames, weights)
        embed_a = self.resnet.seg_1(stats)
        if self.resnet.two_emb_layer:
            out = F.relu(embed_a)
            out = self.resnet.seg_bn_1(out)
            return self.resnet.seg_2(out)
        return embed_a


def main() -> int:
    models_dirs = sys.argv[1:]
    if not models_dirs:
        print(__doc__)
        return 2

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
    emb_model = pipeline._embedding.model_
    emb_model.eval()
    wrapper = EmbeddingTailWrapper(emb_model)
    wrapper.eval()

    torch.manual_seed(0)
    dummy_fbank = torch.randn(BATCH, FBANK_FRAMES, FBANK_FEATURES)
    dummy_weights = torch.rand(BATCH, MASK_FRAMES)

    # Batch invariance check: row i of the b64 wrapper must equal the b1 wrapper on
    # row i alone. If the graph were batch-coupled, split-primary batching would be
    # numerically wrong rather than merely absent.
    with torch.no_grad():
        batched = wrapper(dummy_fbank, dummy_weights)
        single = wrapper(dummy_fbank[7:8], dummy_weights[7:8])
        max_diff = (batched[7:8] - single).abs().max().item()
    assert max_diff < 1e-4, f"batch-invariance check failed: max diff = {max_diff}"
    print(f"  batch-invariance check passed (max diff = {max_diff:.2e})")

    for models_dir in models_dirs:
        out_path = os.path.join(models_dir, f"wespeaker-voxceleb-resnet34-tail-b{BATCH}.onnx")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_fbank, dummy_weights),
                out_path,
                input_names=["fbank", "weights"],
                output_names=["output"],
                opset_version=18,
                dynamo=True,
                external_data=False,
            )
        print(f"exported {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
