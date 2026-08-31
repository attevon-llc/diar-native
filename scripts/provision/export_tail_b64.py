"""Step 2d: export the genuine batch-64 embedding tail.

ADAPTED COPY of `validation/export_tail_b64_addendum.py`, whose `EmbeddingTailWrapper` is
itself copied verbatim from `vendor/speakrs/scripts/export_models.py` (Apache-2.0,
avencera/speakrs) so the b64 graph is bit-consistent with the shipped b1/b3/b32 tails.

Why this is a separate step rather than another line in `export_models.py`: speakrs' loader
asks for `split_tail_model_path(model_path, PRIMARY_BATCH_SIZE=64)` =>
`wespeaker-voxceleb-resnet34-tail-b64.onnx` (`load/sessions.rs:70`), but upstream's exporter
only emits b1/b3/b32. `EmbeddingModel::split_primary_batch_size()` therefore returned 0 on
every model set we ship, and the split-primary batching path was silently never taken
(RESULTS §7.33).

CONTRAST THIS WITH STEP 2c. Both files end in `-b64.onnx` and the correct treatment is
opposite in each case:

* `wespeaker-multimask-tail-b64.onnx` must be a byte COPY of the b32 graph. speakrs sizes
  its multimask runtime buffers for 32, so a real batch-64 graph there kills the worker
  ("receiver disconnected", RESULTS §4.15).
* `wespeaker-voxceleb-resnet34-tail-b64.onnx` — this file — must be a REAL batch-64 export.

Getting those two the wrong way round produces either a crash or dead code, and neither
announces itself.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH = 64
FBANK_FRAMES = 998
FBANK_FEATURES = 80
MASK_FRAMES = 589

#: RESULTS §7.33 measured 7.8e-08 for a correct export. The bar is far looser than that on
#: purpose: it separates "batch-coupled graph" from "float reassociation".
BATCH_INVARIANCE_TOL = 1e-4


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


def export(pipeline: Any, models_dir: str) -> None:
    emb_model = pipeline._embedding.model_
    emb_model.eval()
    wrapper = EmbeddingTailWrapper(emb_model)
    wrapper.eval()

    torch.manual_seed(0)
    dummy_fbank = torch.randn(BATCH, FBANK_FRAMES, FBANK_FEATURES)
    dummy_weights = torch.rand(BATCH, MASK_FRAMES)

    # Batch-invariance gate: row i of the batched wrapper must equal the wrapper run on row
    # i alone. If the graph were batch-coupled, split-primary batching would be numerically
    # WRONG rather than merely absent — a far worse failure than the one being fixed.
    with torch.no_grad():
        batched = wrapper(dummy_fbank, dummy_weights)
        single = wrapper(dummy_fbank[7:8], dummy_weights[7:8])
        max_diff = (batched[7:8] - single).abs().max().item()
    if max_diff >= BATCH_INVARIANCE_TOL:
        raise RuntimeError(
            f"batch-invariance check failed: max diff = {max_diff:.3e} "
            f"(bar {BATCH_INVARIANCE_TOL:.0e})"
        )
    print(f"  batch-invariance check passed (max diff = {max_diff:.2e})")

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
    print(f"  {os.path.basename(out_path)} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
