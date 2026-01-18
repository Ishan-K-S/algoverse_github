import argparse
import json
import logging
import os
import time
from datetime import datetime

import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageNet
from torchvision import transforms
import h5py
import numpy as np

from transformers import AutoModel, AutoFeatureExtractor
from diffusers import AutoencoderKL  # For Flux VAE

# -------------------- Flux wrapper --------------------
class FluxWrapper(torch.nn.Module):
    """
    Wraps the HuggingFace Flux model for activation extraction.
    -sets up Flux and its VAE
    -Extracts activations from the last transformer block
    -records timsteps for each activation
    -samples images from ImageNet before extracting for features
    """
    def __init__(self, model_name="black-forest-labs/FLUX.1-schnell", device="cuda", sampling_ratio=1/16, use_half=False):
        super().__init__()
        self.device = device
        self.use_half = use_half
        self.sampling_ratio = sampling_ratio

        # load model and autoencoder
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)

        if use_half:
            self.model = self.model.half()
            self.vae = self.vae.half()

        # Feature extractor for preprocessing
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    @torch.no_grad()
    def forward_features(self, images: torch.Tensor):
        """
        Input: images tensor (B, 3, H, W) in [0,1]
        Returns:
            activations: (B, num_tokens, feature_dim)
            timesteps: (B,) torch tensor
        """
        B = images.shape[0]

        # Preprocess images using Flux's feature extractor
        images_np = images.cpu().permute(0, 2, 3, 1).numpy()  # (B,H,W,C)
        inputs = self.feature_extractor(images_np, return_tensors="pt").to(self.device)

        # Encode images to latents via VAE
        latents = self.vae.encode(inputs["pixel_values"]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Choose a timestep per batch (for now, random uniform)
        timesteps = torch.randint(low=0, high=1000, size=(B,), device=self.device)

        # Run the model, extracting activations from the last transformer block
        activations_dict = {}
        hook_handle = None

        # Register hook to last transformer block
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "blocks"):
            last_block = self.model.transformer.blocks[-1]
            def hook_fn(module, input, output):
                activations_dict["last_block"] = output
            hook_handle = last_block.register_forward_hook(hook_fn)

        # Forward pass (latent + timestep embedding if required)
        outputs = self.model(
            latents,
            timesteps,
            return_dict=True,
            output_hidden_states=True
        )

        # Remove hook
        if hook_handle:
            hook_handle.remove()

        # Extract activations
        if "last_block" in activations_dict:
            features = activations_dict["last_block"]
        else:
            features = latents  # fallback

        # Ensure 3D tensor: (B, tokens, dim)
        if features.dim() == 4:
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        elif features.dim() == 3:
            # already (B, tokens, dim)
            pass
        else:
            raise ValueError(f"Unexpected features shape: {features.shape}")

        # Token sampling (1/16 of tokens)
        num_tokens = features.shape[1]
        sample_count = max(1, int(num_tokens * self.sampling_ratio))
        indices = torch.randperm(num_tokens, device=self.device)[:sample_count]
        sampled_features = features[:, indices, :]

        return sampled_features, timesteps


# -------------------- Lambda framework --------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Cache Flux activations for ImageNet')
    parser.add_argument('--imagenet_root', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--cache_file', type=str, required=True, help='Output path for HDF5 cache file')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--fp16_storage', action='store_true', help='Store activations as fp16 to save space')
    parser.add_argument('--model_name', type=str, default='Flux', help='Model name for metadata')
    parser.add_argument('--max_batches', type=int, default=None, help='Max batches to process (for testing)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing cache')
    return parser.parse_args()


def setupLogging(cachePath):
    log_file = cachePath.replace('.h5', '.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def check_or_resume_cache(cachePath, overwrite):
    if os.path.exists(cachePath):
        if overwrite:
            os.remove(cachePath)
            return 0
        else:
            with h5py.File(cachePath, 'r') as f:
                labels = f['labels'][:]
                num_cached = len(labels[labels >= 0])
            return num_cached
    else:
        return 0


def save_run_config(args, dataset_size, output_dir, num_tokens, feature_dim):
    config = {
        'imagenet_root': args.imagenet_root,
        'cache_file': args.cache_file,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'fp16_storage': args.fp16_storage,
        'model_name': args.model_name,
        'num_tokens': num_tokens,
        'feature_dim': feature_dim,
        'dataset_size': dataset_size,
        'date_created': str(datetime.now()),
        'platform': 'lambda_labs'
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    return config


def createHDF5storage(cachePath, num_samples, num_tokens, feature_dim, fp16_storage, model_name):
    dtype = "float16" if fp16_storage else "float32"
    with h5py.File(cachePath, 'w') as f:
        f.create_dataset("activations", shape=(num_samples, num_tokens, feature_dim),
                         dtype=dtype, chunks=(32, num_tokens, feature_dim), compression="gzip", compression_opts=4)
        f.create_dataset("labels", shape=(num_samples,), dtype="int64")
        f.create_dataset("timesteps", shape=(num_samples,), dtype="int64")
        f.attrs['model'] = model_name
        f.attrs['num_tokens'] = num_tokens
        f.attrs['feature_dim'] = feature_dim
        f.attrs['storage_type'] = dtype


def loadFeaturesFromHDF5(cache_path, indices=None):
    with h5py.File(cache_path, 'r') as f:
        if indices is None:
            activations = torch.tensor(f['activations'][:])
            labels = torch.tensor(f['labels'][:])
            timesteps = torch.tensor(f['timesteps'][:])
        else:
            activations = torch.tensor(f['activations'][indices])
            labels = torch.tensor(f['labels'][indices])
            timesteps = torch.tensor(f['timesteps'][indices])
    return activations, labels, timesteps


# -------------------- Main Script --------------------
def main():
    args = parse_args()
    logger = setupLogging(args.cache_file)
    logger.info(f"Starting Flux activation caching")

    # Dataset sampling: 1/16 of ImageNet
    full_dataset = ImageNet(root=args.imagenet_root, split="train",
                            transform=transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                            ]))
    num_samples = len(full_dataset) // 16
    sampled_indices = torch.randperm(len(full_dataset))[:num_samples]
    dataset = Subset(full_dataset, sampled_indices)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FluxWrapper(device=device, use_half=args.fp16_storage).to(device).eval()

    # Dummy pass to get shapes
    dummy_input = torch.zeros(1, 3, 224, 224, device=device)
    dummy_features, dummy_timesteps = model.forward_features(dummy_input)
    NUM_TOKENS = dummy_features.shape[1]
    FEATURE_DIM = dummy_features.shape[2]

    # HDF5 cache setup
    start_index = check_or_resume_cache(args.cache_file, args.overwrite)
    if start_index == 0:
        createHDF5storage(args.cache_file, num_samples, NUM_TOKENS, FEATURE_DIM,
                          args.fp16_storage, args.model_name)

    save_run_config(args, num_samples, os.path.dirname(args.cache_file) or '.', NUM_TOKENS, FEATURE_DIM)

    # Activation extraction loop
    samples_processed = start_index
    start_batch = start_index // args.batch_size

    with h5py.File(args.cache_file, 'a') as f:
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx < start_batch:
                continue
            images = images.to(device, non_blocking=True)
            features, timesteps = model.forward_features(images)

            if args.fp16_storage:
                features = features.half()
            else:
                features = features.float()

            features_np = features.cpu().numpy()
            labels_np = labels.cpu().numpy()
            timesteps_np = timesteps.cpu().numpy()

            batch_size = features_np.shape[0]
            f['activations'][samples_processed:samples_processed+batch_size] = features_np
            f['labels'][samples_processed:samples_processed+batch_size] = labels_np
            f['timesteps'][samples_processed:samples_processed+batch_size] = timesteps_np

            samples_processed += batch_size

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {samples_processed}/{num_samples} samples")

    logger.info("Caching complete")
    logger.info(f"Saved to {args.cache_file}")


if __name__ == "__main__":
    main()
