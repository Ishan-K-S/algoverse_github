"""
Cache transformer activations for ImageNet images.

For training Sparse Autoencoders (SAEs) on vision transformer features.

- Stable Diffusion 3 (MMDiT transformer activations during denoising)
- DINOv2 (vision transformer patch tokens)
"""

import sys
sys.path.insert(0, "..")

import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.datasets import ImageNet

from overcomplete.models import StableDiffusion
# from overcomplete.models import DinoV2  

IMAGENET_ROOT = "/path/to/imagenet"  # e.g., "/data/imagenet" or "/datasets/ILSVRC2012"
CACHE_ROOT = "/path/to/cache"        # e.g., "/data/imagenet_cache/train/"

BATCH_SIZE = 256
USE_FP16_STORAGE = False  # optional disk saving

MODEL_NAME = "StableDiffusion"

def save_activation_npz(activation: torch.Tensor, label: torch.Tensor, timestep: torch.Tensor, path: str):
    """
    Save transformer activation, label, and timestep to compressed NPZ.
    """
    if USE_FP16_STORAGE:
        activation = activation.half()

    np.savez_compressed(
        path,
        activation=activation.cpu().numpy(),
        label=label.cpu().numpy(),
        timestep=timestep.cpu().numpy() if timestep is not None else None
    )


def load_activation_npz(path: str):
    """
    Load activation, label, and timestep from NPZ.
    """
    data = np.load(path)
    activation = torch.tensor(data["activation"])
    label = torch.tensor(data["label"])
    timestep = torch.tensor(data["timestep"]) if "timestep" in data else None
    return activation, label, timestep

if __name__ == "__main__":
    model = StableDiffusion().cuda()
    model.eval()

    print(f"Using model: {MODEL_NAME}")
    print(f"FP16 storage: {USE_FP16_STORAGE}")

    dataset = ImageNet(
        root=IMAGENET_ROOT,
        split="train",
        transform=model.preprocess
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    print("ImageNet loaded")
    print("Cache root:", CACHE_ROOT)


    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):

            images = images.cuda(non_blocking=True)
            labels = labels.cpu()

            output = model.model.forward_features(images)

            # Extract raw transformer activations for SAE training
            tokens = output["sampled_features"]  # (B, N_sampled, D)# ============================================================

            output = model.model.forward_features(images)

            # SD3 returns sampled_features and current_timestep
            tokens = output["sampled_features"]  # (B, N_sampled, D)
            
            # Note: tokens are already sampled (1/16) and timestep-modulated

            # ============================================================
            # DINOv2 FEATURE EXTRACTION (COMMENTED)
            # ============================================================

            # model = DinoV2().cuda()
            # model.eval()
            #
            # output = model.model.forward_features(images)
            #
            # cls_token = output["x_norm_clstoken"].unsqueeze(1)        # (B, 1, D)
            # patch_tokens = output["x_norm_patchtokens"]               # (B, N, D)
            # tokens = torch.cat([cls_token, patch_tokens], dim=1)     # (B, 1+N, D)

            # ============================================================

            # Resolve dataset paths for this batch
            start = batch_idx * BATCH_SIZE
            end = start + images.size(0)
            batch_samples = dataset.samples[start:end]

            for i in range(images.size(0)):

                image_path = batch_samples[i][0]
                label = labels[i]
                activation = activations[i]  # (N_sampled, D) or (N, D)
                
                # Get timestep for this sample (SD3 only)
                if timesteps_batch is not None:
                    timestep = timesteps_batch if timesteps_batch.dim() == 0 else timesteps_batch[i]
                else:
                    timestep = None

                # Mirror ImageNet directory structure
                relative_path = os.path.relpath(image_path, dataset.root)

                # IMPORTANT: add model name suffix
                relative_path = relative_path.replace(
                    ".JPEG", f"_{MODEL_NAME}.npz"
                )

                save_path = os.path.join(CACHE_ROOT, relative_path)

                # Resume-safe
                if os.path.exists(save_path):
                    continue

                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                save_activation_npz(activation, label, timestep, save_path)

    print("Caching complete.")
