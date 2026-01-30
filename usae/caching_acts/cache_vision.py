"""
Cache forward feature pass of vision encoders on ImageNet (train/val).

Produces per-image NPZ files mirroring ImageNet directory structure, e.g.
    <cache_root>/<split>/n01440764/n01440764_18_ViT.npz

Each NPZ contains:
    - activation: (N_tokens, D)
    - label: (,) int
"""

import os
from tqdm import tqdm
import torch
from torchvision.datasets import ImageNet

from .io_utils import save_npz

# Local vision encoders
from usae.models.vision_encoders import ViT, DinoV2, SigLIP, CLIP


def cache_model_activations(
    model,
    model_name: str,
    imagenet_root: str,
    cache_root: str,
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 8,
    device: str = "cuda",
):
    print(f"Caching vision activations: {model_name} split={split}")
    dataset = ImageNet(imagenet_root, split=split, transform=model.preprocess)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = model.to(device).eval()

    for i, (x, y) in enumerate(tqdm(loader, desc=f"{model_name} {split}")):
        x = x.to(device, non_blocking=True)
        y = y.cpu()

        # Retrieve batch filenames from the dataset
        image_paths = loader.dataset.samples[i * loader.batch_size:(i + 1) * loader.batch_size]

        with torch.no_grad():
            # Most encoders already return (B, N, D); DinoV2 special handling is inside DinoV2.forward_features
            feats = model.forward_features(x)  # (B, N, D)

        feats = feats.cpu()

        for j in range(feats.size(0)):
            image_path = image_paths[j][0]
            rel_path = os.path.relpath(image_path, os.path.join(loader.dataset.root, split))
            act_path = os.path.join(cache_root, split, rel_path)
            os.makedirs(os.path.dirname(act_path), exist_ok=True)

            out_path = act_path.replace('.JPEG', f"_{model_name}.npz")
            save_npz(out_path, activation=feats[j], label=y[j])


if __name__ == "__main__":
    # Example usage:
    #   python -m usae.caching_acts.cache_vision --imagenet_root ... --cache_root ... --split train --model ViT
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_root", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--model", type=str, default="ViT", choices=["ViT", "DinoV2", "SigLIP", "CLIP"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model_map = {"ViT": ViT, "DinoV2": DinoV2, "SigLIP": SigLIP, "CLIP": CLIP}
    model = model_map[args.model](device=args.device)

    cache_model_activations(
        model=model,
        model_name=args.model,
        imagenet_root=args.imagenet_root,
        cache_root=args.cache_root,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )
