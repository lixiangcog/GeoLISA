"""Distribution-Geometry Feature Alignment for continual test-time adaptation.

This module follows Eqs. (7)--(11) of the GeoLISA paper.  A source-domain
warm-up estimates a sliced-Wasserstein barycenter and first/second moments.
Target features are then aligned to those frozen statistics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SlicedWassersteinPrior(nn.Module):
    """Streaming source prior and differentiable DGFA loss.

    Quantile resampling makes the prior independent of batch/spatial size.  It
    also fixes the running-center update in the supplied BeTTA reference, whose
    ``... + sorted_proj / ns`` term is undefined for the intended average.
    """

    def __init__(
        self,
        channels: int,
        num_projections: int = 64,
        num_quantiles: int = 256,
        seed: int = 32,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        directions = torch.randn(channels, num_projections, generator=generator)
        directions = F.normalize(directions, dim=0)
        self.channels = channels
        self.num_projections = num_projections
        self.num_quantiles = num_quantiles
        self.register_buffer("directions", directions)
        self.register_buffer(
            "barycenter", torch.zeros(num_quantiles, num_projections)
        )
        self.register_buffer("source_sum", torch.zeros(channels))
        self.register_buffer("source_sq_sum", torch.zeros(channels))
        self.register_buffer("point_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("warmup_batches", torch.zeros((), dtype=torch.long))

    def _points(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f"expected [B,{self.channels},H,W], got {tuple(features.shape)}"
            )
        return features.permute(0, 2, 3, 1).reshape(-1, self.channels)

    def _sorted_quantiles(self, points: Tensor) -> Tensor:
        projected = points @ self.directions.to(dtype=points.dtype)
        projected = projected.sort(dim=0).values.T.unsqueeze(0)
        quantiles = F.interpolate(
            projected,
            size=self.num_quantiles,
            mode="linear",
            align_corners=True,
        )
        return quantiles.squeeze(0).T

    @torch.no_grad()
    def update(self, features: Tensor) -> None:
        points = self._points(features.detach()).float()
        quantiles = self._sorted_quantiles(points)
        next_batch = self.warmup_batches + 1
        self.barycenter.add_((quantiles - self.barycenter) / next_batch.float())
        self.source_sum.add_(points.sum(dim=0))
        self.source_sq_sum.add_(points.square().sum(dim=0))
        self.point_count.add_(points.shape[0])
        self.warmup_batches.copy_(next_batch)

    @property
    def initialized(self) -> bool:
        return bool(self.point_count.item() > 0 and self.warmup_batches.item() > 0)

    def source_moments(self) -> tuple[Tensor, Tensor]:
        if not self.initialized:
            raise RuntimeError("source DGFA prior has not been warmed up")
        count = self.point_count.clamp_min(1).to(self.source_sum.dtype)
        mean = self.source_sum / count
        variance = (self.source_sq_sum / count - mean.square()).clamp_min(0)
        return mean, variance

    def loss(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(L_geo, L_align, L_bn)`` from paper Eqs. (8)--(11)."""
        if not self.initialized:
            raise RuntimeError("source DGFA prior has not been warmed up")
        points = self._points(features)
        target_quantiles = self._sorted_quantiles(points)
        align = F.l1_loss(target_quantiles, self.barycenter)

        source_mean, source_variance = self.source_moments()
        target_mean = points.mean(dim=0)
        target_variance = points.var(dim=0, unbiased=False)
        # Eq. (9): squared L2 distances, normalized by channel count.
        bn = (target_mean - source_mean).square().mean()
        bn = bn + (target_variance - source_variance).square().mean()
        return align + bn, align, bn

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, projections={self.num_projections}, "
            f"quantiles={self.num_quantiles}, initialized={self.initialized}"
        )


def binary_entropy(probabilities: Tensor, eps: float = 1e-6) -> Tensor:
    p = probabilities.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def refine_pseudo_labels(
    logits: Tensor,
    features: Tensor,
    confidence_threshold: float = 0.75,
    entropy_threshold: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Paper Eqs. (12)--(16), applied independently to OD and OC channels.

    Returns detached binary labels and a detached reliability mask at logit
    resolution.  The cup channel is constrained to remain inside the disc.
    """
    probabilities = logits.detach().sigmoid()
    pseudo = probabilities.ge(confidence_threshold)
    reliable = binary_entropy(probabilities).lt(entropy_threshold)

    feature_size = features.shape[-2:]
    prob_small = F.interpolate(
        probabilities, feature_size, mode="bilinear", align_corners=False
    )
    pseudo_small = F.interpolate(pseudo.float(), feature_size, mode="nearest").bool()
    reliable_small = F.interpolate(
        reliable.float(), feature_size, mode="nearest"
    ).bool()

    b, channels, _, _ = prob_small.shape
    feature_vectors = features.detach().flatten(2).transpose(1, 2)
    consistency = torch.zeros_like(reliable_small)
    eps = torch.finfo(features.dtype).eps

    for batch_index in range(b):
        vectors = feature_vectors[batch_index]
        for channel in range(channels):
            probs = prob_small[batch_index, channel].flatten()
            labels = pseudo_small[batch_index, channel].flatten()
            mask = reliable_small[batch_index, channel].flatten()
            obj_weights = mask * labels * probs
            bg_weights = mask * (~labels) * (1.0 - probs)
            if obj_weights.sum() <= eps or bg_weights.sum() <= eps:
                continue
            obj_proto = (vectors * obj_weights[:, None]).sum(0) / obj_weights.sum()
            bg_proto = (vectors * bg_weights[:, None]).sum(0) / bg_weights.sum()
            obj_dist = torch.linalg.vector_norm(vectors - obj_proto, dim=1)
            bg_dist = torch.linalg.vector_norm(vectors - bg_proto, dim=1)
            agrees = torch.where(labels, obj_dist < bg_dist, bg_dist < obj_dist)
            consistency[batch_index, channel] = (mask & agrees).reshape(feature_size)

    reliability = F.interpolate(
        consistency.float(), logits.shape[-2:], mode="nearest"
    ).bool()
    pseudo = pseudo & reliability
    if pseudo.shape[1] >= 2:
        pseudo[:, 1] &= pseudo[:, 0]
        reliability[:, 1] &= pseudo[:, 0]
    return pseudo.to(logits.dtype), reliability.to(logits.dtype)


def masked_binary_cross_entropy(
    logits: Tensor, pseudo_labels: Tensor, reliability: Tensor
) -> Tensor:
    per_pixel = F.binary_cross_entropy_with_logits(
        logits, pseudo_labels, reduction="none"
    )
    denominator = reliability.sum().clamp_min(1.0)
    return (per_pixel * reliability).sum() / denominator
