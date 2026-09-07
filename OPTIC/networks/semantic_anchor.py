"""Language-induced, input-dependent semantic anchor for GeoLISA."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NormActDepthwiseConv(nn.Module):
    """NORM-ACT-CONV block with a depthwise-separable convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SemanticAnchorInducer(nn.Module):
    """Expand BioMedCLIP semantics and frozen visual features into an anchor.

    The visual feature may come from SAM (paper mode) or from the frozen source
    encoder (lightweight debug mode).  Only this small module is trainable.
    """

    def __init__(
        self,
        text_embeddings: Tensor,
        visual_dim: int,
        hidden_dim: int = 24,
    ) -> None:
        super().__init__()
        if text_embeddings.ndim != 2 or text_embeddings.shape[1] != 512:
            raise ValueError("text embeddings must have shape [num_prompts, 512]")
        self.register_buffer(
            "text_embeddings", F.normalize(text_embeddings.float(), dim=-1)
        )
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 512),
        )
        # The paper uses five inducing vectors.  A shared visual projection and
        # small learned offsets keep the trainable module close to 0.06M params.
        self.prompt_offsets = nn.Parameter(torch.zeros(text_embeddings.shape[0], 512))
        self.to_seed = nn.Linear(512, hidden_dim)
        self.spatial_seed = nn.Parameter(torch.zeros(1, hidden_dim, 32, 32))
        # Four NORM-ACT-DSC blocks followed by the paper's standard output conv.
        channels = [hidden_dim, 32, 24, 16, 8]
        self.blocks = nn.ModuleList(
            [
                NormActDepthwiseConv(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)
            ]
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(channels[-1]),
            nn.GELU(),
            nn.Conv2d(channels[-1], 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, visual_features: Tensor, output_size: tuple[int, int]):
        pooled = visual_features.mean(dim=(-2, -1))
        shared = self.visual_projector(pooled).unsqueeze(1)
        inducing = F.normalize(shared + self.prompt_offsets.unsqueeze(0), dim=-1)
        semantic = self.text_embeddings.unsqueeze(0).expand_as(inducing)
        alignment_loss = (1.0 - (inducing * semantic).sum(dim=-1)).mean()

        fused = F.normalize(inducing + semantic, dim=-1).mean(dim=1)
        seed = self.to_seed(fused).unsqueeze(-1).unsqueeze(-1)
        x = seed + self.spatial_seed
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = block(x)
        anchor = self.output(x)
        if anchor.shape[-2:] != output_size:
            anchor = F.interpolate(
                anchor, output_size, mode="bilinear", align_corners=False
            )
        return anchor, alignment_loss


class SAMVisualEncoder(nn.Module):
    """Frozen SAM ViT-B image encoder used by the paper's inducing vector."""

    def __init__(self, checkpoint: str, image_size: int = 256) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
        from anchor_inducer.models.segment_anything import sam_model_registry

        registry_key = "vit_b_256" if image_size == 256 else "vit_b_1024"
        sam = sam_model_registry[registry_key](checkpoint=str(checkpoint_path))
        self.image_encoder = sam.image_encoder.eval()
        self.image_size = image_size
        self.register_buffer("pixel_mean", sam.pixel_mean)
        self.register_buffer("pixel_std", sam.pixel_std)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return 256

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        images = F.interpolate(
            images, (self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        images = (images * 255.0 - self.pixel_mean) / self.pixel_std
        return self.image_encoder(images)


def load_text_embeddings(path: str, device: torch.device) -> Tensor:
    value = torch.load(path, map_location=device, weights_only=True)
    if isinstance(value, dict):
        value = value["text_embeddings"]
    if not isinstance(value, Tensor):
        raise TypeError("text embedding file must contain a tensor")
    return value.to(device=device, dtype=torch.float32)
