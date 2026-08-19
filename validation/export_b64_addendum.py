"""Export the missing wespeaker-multimask-tail-b64.onnx.

speakrs' loader (load/sessions.rs:162) requests multi_mask_model_path(model_path,
PRIMARY_BATCH_SIZE=64) => wespeaker-multimask-tail-b64.onnx, but scripts/export_models.py
only writes -b32 (MULTI_MASK_BATCH_SIZE). The missing file silently disables multimask
batching (batch size falls back to 1 => per-chunk fbank + batch-1 GPU predicts).

MultiMaskTailWrapper is copied verbatim from vendor/speakrs/scripts/export_models.py
(Apache-2.0, avencera/speakrs) so the b64 graph is bit-consistent with upstream's b1/b32.
"""

import os
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

NUM_SPEAKERS = 3
BATCH = 64


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


def main() -> int:
    models_dir = sys.argv[1]
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
    emb_model = pipeline._embedding.model_
    emb_model.eval()
    wrapper = MultiMaskTailWrapper(emb_model)
    wrapper.eval()

    dummy_fbank = torch.randn(BATCH, 998, 80)
    dummy_masks = torch.ones(BATCH * NUM_SPEAKERS, 589)
    out_path = os.path.join(models_dir, f"wespeaker-multimask-tail-b{BATCH}.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_fbank, dummy_masks),
            out_path,
            input_names=["fbank", "weights"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
    print(f"exported {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
