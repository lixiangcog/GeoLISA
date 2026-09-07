"""Cache the five BioMedCLIP prompt embeddings used by GeoLISA."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer


MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
PROMPTS = [
    "A retinal fundus photograph containing the optic disc, the bright circular region where the optic nerve exits the eye.",
    "A retinal fundus photograph containing the optic cup, the pale central depression located inside the optic disc.",
    "Segment the complete boundary and interior of the optic disc in this color fundus image.",
    "Segment the complete boundary and interior of the optic cup in this color fundus image.",
    "A color fundus image for joint medical segmentation of the optic disc and optic cup, with the cup contained inside the disc.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, _ = create_model_from_pretrained(MODEL_ID)
    tokenizer = get_tokenizer(MODEL_ID)
    model = model.to(device).eval()
    tokens = tokenizer(PROMPTS, context_length=256).to(device)
    with torch.no_grad():
        embeddings = F.normalize(model.encode_text(tokens), dim=-1).cpu()
    if embeddings.shape != (5, 512):
        raise RuntimeError(f"unexpected BioMedCLIP text shape: {tuple(embeddings.shape)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": MODEL_ID, "prompts": PROMPTS, "text_embeddings": embeddings},
        output,
    )
    print(f"saved {tuple(embeddings.shape)} embeddings to {output}")


if __name__ == "__main__":
    main()
