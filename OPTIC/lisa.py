"""GeoLISA continual test-time adaptation for OD/OC segmentation.

This implementation is calibrated to the equations and implementation details
in the paper.  It requires no target labels for adaptation; labels are read
only after each prediction to report Dice and ASD.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.transform import collate_fn_wo_transform
from networks.ResUnet import ResUnet
from networks.dgfa import (
    SlicedWassersteinPrior,
    masked_binary_cross_entropy,
    refine_pseudo_labels,
)
from networks.semantic_anchor import (
    SAMVisualEncoder,
    SemanticAnchorInducer,
    load_text_embeddings,
)
from utils.metrics import calculate_metrics


def domain_csvs(domain: str) -> list[str]:
    if domain == "REFUGE_Valid":
        return ["REFUGE_Valid.csv"]
    return [f"{domain}_train.csv", f"{domain}_test.csv"]


def build_loader(
    root: str,
    domains: list[str],
    image_size: int,
    workers: int,
    max_samples: int | None = None,
) -> DataLoader:
    csvs = [csv_name for domain in domains for csv_name in domain_csvs(domain)]
    images, masks = convert_labeled_list(root, csvs)
    dataset = OPTIC_dataset(root, images, masks, image_size, img_normalize=True)
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn_wo_transform,
    )


def source_forward(model: ResUnet, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return logits, the latent geometry feature, and penultimate feature."""
    bottleneck, skips = model.res(images)
    x = F.relu(bottleneck)
    x = model.up1(x, skips[3])
    x = model.up2(x, skips[2])
    x = model.up3(x, skips[1])
    x = model.up4(x, skips[0])
    x = model.up5(x)
    head = F.relu(model.bnout(x))
    return model.seg_head(head), bottleneck, head


class GeoLISA(nn.Module):
    """Frozen source segmenter plus a small trainable semantic anchor inducer."""

    def __init__(
        self,
        source_checkpoint: str,
        text_embeddings: Tensor,
        semantic_backend: str,
        sam_checkpoint: str | None,
        sam_image_size: int,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.segmenter = ResUnet(resnet="resnet34", num_classes=2, pretrained=False)
        checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
        self.segmenter.load_state_dict(checkpoint, strict=True)
        self.segmenter.eval()
        for parameter in self.segmenter.parameters():
            parameter.requires_grad_(False)

        self.semantic_backend = semantic_backend
        if semantic_backend == "sam":
            if not sam_checkpoint:
                raise ValueError("--sam-checkpoint is required for --semantic-backend sam")
            self.visual_encoder = SAMVisualEncoder(sam_checkpoint, sam_image_size)
            visual_dim = self.visual_encoder.output_dim
        elif semantic_backend == "source":
            self.visual_encoder = None
            visual_dim = 512
        else:
            raise ValueError(f"unknown semantic backend: {semantic_backend}")

        self.anchor_inducer = SemanticAnchorInducer(text_embeddings, visual_dim)
        self.epsilon = epsilon

    def train(self, mode: bool = True):
        super().train(mode)
        self.segmenter.eval()
        if self.visual_encoder is not None:
            self.visual_encoder.eval()
        return self

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.visual_encoder is None:
            with torch.no_grad():
                _, visual_features, _ = source_forward(self.segmenter, images)
        else:
            visual_features = self.visual_encoder(images)
        anchor, semantic_loss = self.anchor_inducer(visual_features, images.shape[-2:])
        logits, geometry, penultimate = source_forward(
            self.segmenter, (images + self.epsilon * anchor).clamp(0.0, 1.0)
        )
        return logits, geometry, penultimate, semantic_loss


@torch.no_grad()
def warm_up_prior(
    model: GeoLISA,
    prior: SlicedWassersteinPrior,
    loader: DataLoader,
    device: torch.device,
) -> None:
    model.eval()
    for batch in tqdm(loader, desc="source DGFA warm-up", ncols=90):
        images = torch.from_numpy(batch["data"]).float().to(device)
        _, features, _ = source_forward(model.segmenter, images)
        prior.update(features)


def summarize_metrics(values: dict[str, list[float]]) -> dict[str, float]:
    return {
        name: float(np.nanmean(items)) if items else float("nan")
        for name, items in values.items()
    }


def run(config: argparse.Namespace) -> dict:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(config.device)
    text_embeddings = load_text_embeddings(config.text_embeddings, device)
    checkpoint = Path(config.model_root) / config.source_dataset / "last-Res_Unet.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {checkpoint}")

    model = GeoLISA(
        str(checkpoint),
        text_embeddings,
        config.semantic_backend,
        config.sam_checkpoint,
        config.sam_image_size,
        config.epsilon,
    ).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            trainable,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            trainable, lr=config.lr, weight_decay=config.weight_decay
        )
    trainable_count = sum(p.numel() for p in trainable)
    print(f"trainable parameters: {trainable_count:,} ({trainable_count / 1e6:.3f}M)")

    prior = SlicedWassersteinPrior(
        channels=512,
        num_projections=config.projections,
        num_quantiles=config.quantiles,
        seed=config.seed,
    ).to(device)
    prior_path = Path(config.prior_path) if config.prior_path else None
    if prior_path and prior_path.is_file():
        prior.load_state_dict(torch.load(prior_path, map_location=device, weights_only=True))
        print(f"loaded DGFA prior: {prior_path}")
    else:
        source_loader = build_loader(
            config.dataset_root,
            [config.source_dataset],
            config.image_size,
            config.num_workers,
            config.max_source_samples,
        )
        warm_up_prior(model, prior, source_loader, device)
        if prior_path:
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(prior.state_dict(), prior_path)
            print(f"saved DGFA prior: {prior_path}")

    metric_names = ["disc_dice", "disc_assd", "cup_dice", "cup_assd"]
    all_results: dict[str, dict] = {}
    model.train()
    for domain in config.target_datasets:
        loader = build_loader(
            config.dataset_root,
            [domain],
            config.image_size,
            config.num_workers,
            config.max_target_samples,
        )
        metrics: dict[str, list[float]] = defaultdict(list)
        losses: dict[str, list[float]] = defaultdict(list)
        for batch in tqdm(loader, desc=f"adapt {domain}", ncols=90):
            images = torch.from_numpy(batch["data"]).float().to(device)
            labels = torch.from_numpy(batch["mask"]).float()
            for _ in range(config.steps):
                logits, geometry, penultimate, semantic_loss = model(images)
                pseudo, reliable = refine_pseudo_labels(
                    logits,
                    penultimate,
                    confidence_threshold=config.gamma,
                    entropy_threshold=config.eta,
                )
                mce = masked_binary_cross_entropy(logits, pseudo, reliable)
                geo, align, bn = prior.loss(geometry)
                loss = mce + config.alpha * geo + config.semantic_weight * semantic_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                logits, _, _, _ = model(images)
            model.train()
            sample_metrics = calculate_metrics(logits.sigmoid().cpu(), labels)
            for name, value in zip(metric_names, sample_metrics):
                metrics[name].extend(float(x) for x in value)
            losses["total"].append(float(loss.detach()))
            losses["mce"].append(float(mce.detach()))
            losses["geo"].append(float(geo.detach()))
            losses["align"].append(float(align.detach()))
            losses["bn"].append(float(bn.detach()))
            losses["semantic"].append(float(semantic_loss.detach()))
            losses["reliable_fraction"].append(float(reliable.mean().detach()))

        metric_summary = summarize_metrics(metrics)
        metric_summary["mean_dice"] = (
            metric_summary["disc_dice"] + metric_summary["cup_dice"]
        ) / 2.0
        metric_summary["mean_assd"] = (
            metric_summary["disc_assd"] + metric_summary["cup_assd"]
        ) / 2.0
        all_results[domain] = {
            "samples": len(loader.dataset),
            "metrics": metric_summary,
            "losses": summarize_metrics(losses),
        }
        print(json.dumps({domain: all_results[domain]}, indent=2))

    result = {
        "source_dataset": config.source_dataset,
        "target_datasets": config.target_datasets,
        "semantic_backend": config.semantic_backend,
        "trainable_parameters": trainable_count,
        "optimizer": config.optimizer,
        "learning_rate": config.lr,
        "adaptation_steps_per_sample": config.steps,
        "dgfa_projections": config.projections,
        "dgfa_quantiles": config.quantiles,
        "paper_hyperparameters": {
            "epsilon": config.epsilon,
            "gamma": config.gamma,
            "eta": config.eta,
            "alpha": config.alpha,
        },
        "results": all_results,
    }
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote results: {output}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--source-dataset", default="RIM_ONE_r3")
    parser.add_argument("--target-datasets", nargs="+", default=["Drishti_GS"])
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--semantic-backend", choices=["sam", "source"], default="sam")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--sam-image-size", type=int, choices=[256, 1024], default=256)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--projections", type=int, default=64)
    parser.add_argument("--quantiles", type=int, default=256)
    parser.add_argument("--prior-path")
    parser.add_argument("--max-source-samples", type=int)
    parser.add_argument("--max-target-samples", type=int)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--output", default="results/geolisa.json")
    args = parser.parse_args()
    overlap = set(args.target_datasets) & {args.source_dataset}
    if overlap:
        parser.error(f"source domain cannot also be a target domain: {sorted(overlap)}")
    return args


if __name__ == "__main__":
    run(parse_args())
