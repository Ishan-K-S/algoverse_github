"""
Cache diffusion transformer activations across denoising timesteps on ImageNet images.

Goal: mirror the vision-encoder caching format so downstream combining can store
*all* sources for the same ImageNet image together.

Output per image:
    <cache_root>/<split>/<class>/<img_basename>_<diffusion_name>.npz

Each NPZ contains:
    - activation: (T, N_tokens, D)   (stack of per-timestep activations)
    - timesteps:  (T,) int64         (scheduler timesteps)
    - sigmas:     (T,) float32       (noise levels)
    - label:      (,) int
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from torchvision.datasets import ImageNet

from .io_utils import save_npz
from usae.models.diffusion_extractors import SD3ActivationExtractor, FLUXActivationExtractor


def cache_diffusion_activations(
    extractor,
    diffusion_name: str,
    imagenet_root: str,
    cache_root: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = "cuda",
):
    print(f"Caching diffusion activations: {diffusion_name} split={split}")
    dataset = ImageNet(imagenet_root, split=split, transform=extractor.preprocess)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    extractor = extractor.to(device).eval()

    for i, (x, y) in enumerate(tqdm(loader, desc=f"{diffusion_name} {split}")):
        # x comes from extractor.preprocess -> already in [0,1] then Normalize -> [-1,1] for diffusion extractor
        x = x.to(device, non_blocking=True)
        y = y.cpu()

        image_paths = loader.dataset.samples[i * loader.batch_size:(i + 1) * loader.batch_size]

        # Extract activations for the whole batch in one go
        with torch.no_grad():
            out = extractor.extract_activations(x)

        # out.activations is a list length T of (B, N, D)
        # stack to (B, T, N, D)
        acts_btnd = torch.stack(out.activations, dim=1).cpu()  # (B, T, N, D)
        # timesteps list of tensors (scalar) -> (T,)
        timesteps = torch.stack([t.detach().cpu().to(torch.int64) for t in out.timesteps], dim=0).numpy()
        sigmas = np.asarray(out.sigmas, dtype=np.float32)

        for j in range(acts_btnd.size(0)):
            image_path = image_paths[j][0]
            rel_path = os.path.relpath(image_path, os.path.join(loader.dataset.root, split))
            act_path = os.path.join(cache_root, split, rel_path)
            os.makedirs(os.path.dirname(act_path), exist_ok=True)

            out_path = act_path.replace('.JPEG', f"_{diffusion_name}.npz")
            save_npz(
                out_path,
                activation=acts_btnd[j],  # (T, N, D)
                timesteps=timesteps,
                sigmas=sigmas,
                label=y[j],
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_root", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--diffusion", type=str, default="FLUX", choices=["SD3", "FLUX"])
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.diffusion == "SD3":
        extractor = SD3ActivationExtractor(num_inference_steps=args.num_steps, device=args.device)
        name = "SD3"
    else:
        extractor = FLUXActivationExtractor(num_inference_steps=args.num_steps, device=args.device)
        name = "FLUX"

    cache_diffusion_activations(
        extractor=extractor,
        diffusion_name=name,
        imagenet_root=args.imagenet_root,
        cache_root=args.cache_root,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )
