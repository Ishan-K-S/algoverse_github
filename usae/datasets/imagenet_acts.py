"""
ImageNet activation dataset loader for combined NPZ files.

Combined NPZ format:
  - vision source key S: array (N, D)
  - diffusion source key S: array (T, N, D)
    plus S__timesteps: (T,), S__sigmas: (T,)
  - label: int

__getitem__ returns:
  (activations_dict, label, metadata_dict)

For vision models, activations are torch.FloatTensor (N, D).
For diffusion models, activations are torch.FloatTensor (T, N, D).

metadata_dict:
  - for each diffusion source S:
        metadata['sigmas'][S] = torch.FloatTensor (T,)
        metadata['timesteps'][S] = torch.LongTensor (T,)
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from torchvision.datasets import ImageNet
from typing import Dict, Tuple, Union, Any


MODELS_WITH_CLS_TOKEN = {"DinoV2", "ViT", "CLIP"}


class ImageNetCombinedActivationDataset(ImageNet):
    def __init__(
        self,
        root: str,
        activation_root: str,
        sources: list,
        split: str = "train",
        target_class: Union[str, int, None] = None,
        use_class_tokens: bool = True,
        standardize: bool = False,
        divide_norm: bool = False,
        **kwargs,
    ):
        super().__init__(root=root, split=split, **kwargs)

        if target_class is not None and target_class != "ALL":
            if isinstance(target_class, str):
                if target_class not in self.class_to_idx:
                    raise ValueError(f"Invalid WordNet ID: {target_class}")
                target_class = self.class_to_idx[target_class]
            self.samples = [(p, idx) for p, idx in self.samples if idx == target_class]
            print(f"Filtered to {len(self.samples)} samples for class {target_class}")
        else:
            print("Training on all classes")

        self.activation_root = activation_root
        self.sources = sources
        self.split = split
        self.use_class_tokens = use_class_tokens
        self.standardize = standardize
        self.divide_norm = divide_norm

        self.samples_used = []
        for img_path, target in tqdm(self.samples, total=len(self.samples), desc="Indexing activation files"):
            rel_path = os.path.relpath(img_path, os.path.join(self.root, self.split))
            class_dir = os.path.dirname(rel_path)
            act_path = os.path.join(
                self.activation_root,
                self.split,
                class_dir,
                os.path.basename(img_path).replace(".JPEG", "_combined.npz"),
            )
            self.samples_used.append((act_path, target))

        self.standardization_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        if self.standardize:
            self._compute_standardization_stats()

    def _compute_standardization_stats(self, sample_size: int = 2000):
        # crude but effective: sample images, flatten all tokens (and time if present)
        sample_size = min(sample_size, len(self.samples_used))
        idxs = np.random.choice(len(self.samples_used), sample_size, replace=False)

        for source in self.sources:
            print(f"Computing mean/std for {source}")
            xs = []
            for idx in tqdm(idxs, desc=f"stats {source}"):
                path, _ = self.samples_used[idx]
                npz = np.load(path, mmap_mode="r")
                if source not in npz.files:
                    continue
                arr = npz[source]
                x = torch.from_numpy(arr).float()
                # remove CLS if requested (vision only)
                if x.dim() == 2 and (source in MODELS_WITH_CLS_TOKEN) and (not self.use_class_tokens):
                    x = x[1:, :]
                # flatten tokens/time -> (M, D)
                if x.dim() == 3:  # (T,N,D)
                    x = x.reshape(-1, x.shape[-1])
                else:  # (N,D)
                    x = x.reshape(-1, x.shape[-1])
                xs.append(x)

            if not xs:
                continue
            X = torch.cat(xs, dim=0)
            self.standardization_stats[source] = {
                "mean": X.mean(dim=0),
                "std": X.std(dim=0).clamp_min(1e-6),
            }

    def __len__(self):
        return len(self.samples_used)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, Any]]:
        act_path, target = self.samples_used[index]
        npz = np.load(act_path, mmap_mode="r")

        acts: Dict[str, torch.Tensor] = {}
        metadata: Dict[str, Any] = {"sigmas": {}, "timesteps": {}}

        for source in self.sources:
            if source not in npz.files:
                continue
            x = torch.from_numpy(npz[source]).float()

            # remove CLS token if requested (vision only)
            if x.dim() == 2 and (source in MODELS_WITH_CLS_TOKEN) and (not self.use_class_tokens):
                x = x[1:, :]

            # standardize / norm
            if self.standardize and source in self.standardization_stats:
                mean = self.standardization_stats[source]["mean"]
                std = self.standardization_stats[source]["std"]
                x = (x - mean) / std

            if self.divide_norm:
                # normalize over feature dim
                denom = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-6)
                x = x / denom

            acts[source] = x

            # diffusion metadata if present
            t_key = f"{source}__timesteps"
            s_key = f"{source}__sigmas"
            if t_key in npz.files:
                metadata["timesteps"][source] = torch.from_numpy(npz[t_key]).long()
            if s_key in npz.files:
                metadata["sigmas"][source] = torch.from_numpy(npz[s_key]).float()

        return acts, int(target), metadata
