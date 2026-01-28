"""
activation caching script for multiple models on imagenet in one file

Includes:
  - CLIP   -> saves shard_XXXXX.pt files (acts/labels/indices)
  - SigLIP -> saves one big .h5 cache (activations/labels) + logs + run_config.json
  - ViT    -> saves per-image .npz files (like imagenet folder structure)
"""

import argparse
import os
import sys



# CLIP


def clip_extract_features(model, processor, images, device):
    """
    Extract feature vectors from images using CLIP vision encoder.
    Returns [N, D] tensor where D is the feature dimension.
    """
    import torch

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=inputs.pixel_values)

    if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
        features = vision_outputs.pooler_output
    else:
        features = vision_outputs.last_hidden_state.mean(dim=1)

    return features


def run_clip(args):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageNet
    from transformers import CLIPProcessor, CLIPModel
    from tqdm import tqdm

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_name}")
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    print(f"Loading ImageNet {args.split} split from {args.imagenet_root}")
    try:
        dataset = ImageNet(root=args.imagenet_root, split=args.split)
    except Exception as e:
        print(f"Error loading ImageNet dataset: {e}")
        print("Make sure --imagenet_root points to a directory with train/val folders")
        return

    if args.max_images is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_images, len(dataset))))

    def collate_fn(batch):
        images, labels = zip(*batch)
        return list(images), torch.tensor(labels, dtype=torch.int64)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    all_features = []
    all_labels = []
    all_indices = []
    global_idx = 0
    shard_num = 0

    print(f"Processing {len(dataset)} images in batches of {args.batch_size}")

    for batch_images, batch_labels in tqdm(dataloader, desc="Processing batches"):
        try:
            batch_features = clip_extract_features(model, processor, batch_images, device)
            batch_features = batch_features.cpu().half()

            batch_size = len(batch_labels)
            batch_indices = torch.arange(global_idx, global_idx + batch_size, dtype=torch.int64)

            all_features.append(batch_features)
            all_labels.append(batch_labels)
            all_indices.append(batch_indices)

            global_idx += batch_size

            total_accumulated = sum(f.shape[0] for f in all_features)
            if total_accumulated >= args.shard_size:
                shard_features = torch.cat(all_features, dim=0)
                shard_labels = torch.cat(all_labels, dim=0)
                shard_indices = torch.cat(all_indices, dim=0)

                if len(shard_features) > args.shard_size:
                    all_features = [shard_features[args.shard_size:]]
                    all_labels = [shard_labels[args.shard_size:]]
                    all_indices = [shard_indices[args.shard_size:]]

                    shard_features = shard_features[:args.shard_size]
                    shard_labels = shard_labels[:args.shard_size]
                    shard_indices = shard_indices[:args.shard_size]
                else:
                    all_features, all_labels, all_indices = [], [], []

                shard_path = os.path.join(args.out_dir, f"shard_{shard_num:05d}.pt")
                shard_data = {
                    "acts": shard_features,
                    "labels": shard_labels,
                    "indices": shard_indices,
                    "model_name": args.model_name,
                    "split": args.split,
                }
                torch.save(shard_data, shard_path)
                print(
                    f"Saved shard {shard_num} to {shard_path} "
                    f"({len(shard_features)} images, feature dim: {shard_features.shape[1]})"
                )
                shard_num += 1

        except Exception as e:
            print(f"Error processing batch: {e}")
            continue

    if all_features:
        shard_features = torch.cat(all_features, dim=0)
        shard_labels = torch.cat(all_labels, dim=0)
        shard_indices = torch.cat(all_indices, dim=0)

        shard_path = os.path.join(args.out_dir, f"shard_{shard_num:05d}.pt")
        shard_data = {
            "acts": shard_features,
            "labels": shard_labels,
            "indices": shard_indices,
            "model_name": args.model_name,
            "split": args.split,
        }
        torch.save(shard_data, shard_path)
        print(f"Saved final shard {shard_num} to {shard_path} ({len(shard_features)} images)")
        shard_num += 1

    print(f"Done! Saved {shard_num} shard(s) to {args.out_dir}")



# SigLip


def siglip_parse_args(subparser):
    subparser.add_argument("--imagenet_root", type=str, required=True,
                           help="Path to ImageNet dataset")
    subparser.add_argument("--cache_file", type=str, required=True,
                           help="Output path for HDF5 cache file")

    subparser.add_argument("--batch_size", type=int, default=256,
                           help="Batch size for processing")
    subparser.add_argument("--num_workers", type=int, default=8,
                           help="Number of data loading workers")

    subparser.add_argument("--fp16_storage", action="store_true",
                           help="Store activations as fp16 to save space")

    subparser.add_argument("--model_name", type=str, default="SigLIP",
                           help="Model name for metadata")

    subparser.add_argument("--max_batches", type=int, default=None,
                           help="Max batches to process (for testing)")
    subparser.add_argument("--overwrite", action="store_true",
                           help="Overwrite existing cache without prompting")


def siglip_setup_logging(cachePath):
    import logging

    log_file = cachePath.replace(".h5", ".log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def siglip_check_or_resume_cache(cachePath, overwrite):
    import h5py
    import numpy as np

    if os.path.exists(cachePath):
        if overwrite:
            print(f"Overwriting existing cache: {cachePath}")
            os.remove(cachePath)
            return 0
        else:
            with h5py.File(cachePath, "r") as f:
                labels = f["labels"][:]
                # NOTE: resume assumes "empty" labels are -1
                num_cached = int(np.sum(labels >= 0))

            print(f"Found existing cache with {num_cached} samples")
            print("Resuming from where we left off...")
            return num_cached
    else:
        return 0


def siglip_save_run_config(args, dataset_size, output_dir, token_output, feature_dimension):
    import json
    from datetime import datetime

    config = {
        "imagenet_root": args.imagenet_root,
        "cache_file": args.cache_file,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "fp16_storage": args.fp16_storage,
        "model_name": args.model_name,
        "num_tokens": token_output,
        "feature_dim": feature_dimension,
        "dataset_size": dataset_size,
        "date_created": str(datetime.now()),
        "platform": "lambda_labs",
    }

    config_path = os.path.join(output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Run config saved to: {config_path}")
    return config


def siglip_create_hdf5_storage(cachePath, numberOfImages, fp16_storage, model_name, token_output, feature_dimension):
    import h5py
    import numpy as np

    datatype = "float16" if fp16_storage else "float32"

    with h5py.File(cachePath, "w") as f:
        f.create_dataset(
            "activations",
            shape=(numberOfImages, token_output, feature_dimension),
            dtype=datatype,
            chunks=(32, token_output, feature_dimension),
            compression="gzip",
            compression_opts=4,
        )

        f.create_dataset(
            "labels",
            shape=(numberOfImages,),
            dtype="int64",
            data=np.full((numberOfImages,), -1, dtype=np.int64),
        )

        f.attrs["model"] = model_name
        f.attrs["num_tokens"] = token_output
        f.attrs["output_dimension"] = feature_dimension
        f.attrs["storage_type"] = datatype

    print("Created an Empty HDF5 Storage file for the activations.")
    print("The activations will be added as they are extracted from SigLIP model")


def siglip_extract_activations(model, images):
    import torch
    with torch.no_grad():
        tokens = model.forward_features(images)
    return tokens


def run_siglip(args):
    import time
    import torch
    import h5py
    from torchvision.datasets import ImageNet
    from overcomplete.models import SigLIP

    logger = siglip_setup_logging(args.cache_file)
    logger.info("Starting SigLIP activation caching")
    logger.info(f"Arguments: {vars(args)}")

    current_index = siglip_check_or_resume_cache(args.cache_file, args.overwrite)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading {args.model_name} model...")
    model = SigLIP().to(device)
    model.eval()
    logger.info("SigLIP model has been loaded")

    with torch.no_grad():
        dummyInputs = torch.zeros(1, 3, 224, 224, device=device)
        dummyTokens = model.forward_features(dummyInputs)
    NUM_TOKENS = dummyTokens.shape[1]
    FEATURE_DIM = dummyTokens.shape[2]
    logger.info(f"SigLIP has {NUM_TOKENS} tokens and feature dim {FEATURE_DIM}")

    logger.info(f"Loading ImageNet from: {args.imagenet_root}")
    dataset = ImageNet(
        root=args.imagenet_root,
        split="train",
        transform=model.preprocess,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_batch = current_index // args.batch_size
    logger.info(f"ImageNet loaded: {len(dataset)} images, {len(dataloader)} batches")

    cache_dir = os.path.dirname(args.cache_file) or "."
    siglip_save_run_config(args, len(dataset), cache_dir, NUM_TOKENS, FEATURE_DIM)

    if current_index == 0:
        siglip_create_hdf5_storage(
            args.cache_file,
            len(dataset),
            args.fp16_storage,
            args.model_name,
            NUM_TOKENS,
            FEATURE_DIM,
        )

    logger.info("Starting activation extraction...")
    start_time = time.time()
    samples_processed = current_index

    with h5py.File(args.cache_file, "a") as f:
        for batch_index, (images, labels) in enumerate(dataloader):
            if batch_index < start_batch:
                continue

            if args.max_batches is not None and batch_index >= args.max_batches:
                logger.info(f"Reached max_batches limit: {args.max_batches}")
                break

            images = images.to(device, non_blocking=True)

            tokens = siglip_extract_activations(model, images)

            if args.fp16_storage:
                tokens = tokens.to(torch.float16)
            else:
                tokens = tokens.to(torch.float32)

            tokens_np = tokens.cpu().numpy()
            labels_np = labels.cpu().numpy()

            batch_size = tokens_np.shape[0]
            f["activations"][current_index:current_index + batch_size] = tokens_np
            f["labels"][current_index:current_index + batch_size] = labels_np

            current_index += batch_size
            samples_processed += batch_size

            if (batch_index + 1) % 100 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = samples_processed / elapsed
                eta_batches = len(dataloader) - (batch_index + 1)
                eta_seconds = eta_batches / ((batch_index + 1) / elapsed)

                logger.info(
                    f"Progress: {batch_index+1}/{len(dataloader)} batches, "
                    f"{samples_processed} samples, "
                    f"{samples_per_sec:.1f} samples/sec, "
                    f"ETA: {eta_seconds/60:.1f} min"
                )

    total_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info("Caching complete")
    logger.info(f"Total samples: {samples_processed}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average speed: {samples_processed/total_time:.1f} samples/sec")
    logger.info(f"Saved to: {args.cache_file}")
    logger.info("=" * 50)



# ViT


def vit_save_activation_npz(activation, label, path):
    import numpy as np
    np.savez_compressed(
        path,
        activation=activation.cpu().numpy(),
        label=label.cpu().numpy(),
    )


def run_vit(args):
    import torch
    import numpy as np
    from tqdm import tqdm
    from torchvision.datasets import ImageNet
    from overcomplete.models import ViT

    os.makedirs(args.cache_dir, exist_ok=True)

    print("Loading ViT model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT().to(device)
    model.eval()

    print(f"Loading ImageNet {args.split} split...")
    imagenet_data = ImageNet(
        root=args.imagenet_root,
        split=args.split,
        transform=model.preprocess,
    )
    print(f"Loaded {len(imagenet_data)} images")

    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Processing batches (resume={args.resume})...")
    skipped = 0
    saved = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            labels = labels.cpu()

            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + images.size(0)
            batch_samples = imagenet_data.samples[start_idx:end_idx]

            output_features = model.model.forward_features(images)

            for j in range(images.size(0)):
                image_path = batch_samples[j][0]
                label = labels[j]
                activation = output_features[j]

                relative_path = os.path.relpath(image_path, imagenet_data.root)
                activation_path = os.path.join(args.cache_dir, relative_path)
                activation_filename = os.path.splitext(activation_path)[0] + "_ViT.npz"

                if args.resume and os.path.exists(activation_filename):
                    skipped += 1
                    continue

                os.makedirs(os.path.dirname(activation_filename), exist_ok=True)
                vit_save_activation_npz(activation, label, activation_filename)
                saved += 1

    print(f"\nDone! Saved: {saved}, Skipped: {skipped}")



# argparse


def build_parser():
    parser = argparse.ArgumentParser(
        description="Cache ImageNet activations for multiple models (CLIP / SigLIP / ViT) from ONE file"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- CLIP ----
    clip_p = subparsers.add_parser("clip", help="Cache CLIP vision encoder activations to .pt shards")
    clip_p.add_argument("--imagenet_root", type=str, required=True,
                        help="Path to ImageNet root directory (contains train/val folders)")
    clip_p.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Dataset split to use (default: val)")
    clip_p.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16",
                        help="Hugging Face model name (default: openai/clip-vit-base-patch16)")
    clip_p.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing (default: 32)")
    clip_p.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers (default: 4)")
    clip_p.add_argument("--shard_size", type=int, default=1024,
                        help="Number of images per shard (default: 1024)")
    clip_p.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for shard files")
    clip_p.add_argument("--max_images", type=int, default=None,
                        help="Optional: maximum number of images to process (for quick testing)")
    clip_p.set_defaults(func=run_clip)

    # ---- SigLIP ----
    siglip_p = subparsers.add_parser("siglip", help="Cache SigLIP activations to ONE .h5 file")
    siglip_parse_args(siglip_p)
    siglip_p.set_defaults(func=run_siglip)

    # ---- ViT ----
    vit_p = subparsers.add_parser("vit", help="Cache ViT activations to per-image .npz files")
    vit_p.add_argument("--imagenet_root", type=str, required=True,
                       help="Path to ImageNet root directory")
    vit_p.add_argument("--cache_dir", type=str, required=True,
                       help="Output directory for cached activations")
    vit_p.add_argument("--batch_size", type=int, default=512,
                       help="Batch size (default: 512)")
    vit_p.add_argument("--num_workers", type=int, default=8,
                       help="Number of data loading workers (default: 8)")
    vit_p.add_argument("--split", type=str, default="train", choices=["train", "val"],
                       help="Dataset split (default: train)")
    vit_p.add_argument("--resume", action="store_true",
                       help="Skip already cached files")
    vit_p.set_defaults(func=run_vit)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
