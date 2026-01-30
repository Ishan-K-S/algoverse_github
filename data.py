"""
Module for loading ImageNet Activations Dataset

Patched to support BOTH:
  (A) Vision encoders cached as (N, D) per image
  (B) Diffusion models cached as (T, N, D) per image, plus sigma/timestep metadata

Key features:
- Still supports ImageNet directory structure.
- Still supports combined_npz ("*_combined.npz") or per-model npz files.
- Adds optional return_metadata flag:
    - return_metadata=False (default): returns (activations_dict, target)
    - return_metadata=True: returns ((activations_dict, metadata), target)

- Standardization stats work for both 2D (N,D) and 3D (T,N,D) cached activations.
"""

import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from torchvision.datasets import ImageNet
from typing import Tuple, Dict, Union, Optional, Any


# Models that have a CLS token prepended to patch tokens
# ViT, DinoV2, and CLIP all have CLS tokens at position 0
# SigLIP does NOT have a CLS token
MODELS_WITH_CLS_TOKEN = {"DinoV2", "ViT", "CLIP"}


def _is_diffusion_activation(act: torch.Tensor) -> bool:
    """
    Heuristic: diffusion cached activations are typically (T, N, D).
    Vision activations are typically (N, D).
    """
    return act.dim() == 3


def _flatten_tokens_for_stats(act: torch.Tensor) -> torch.Tensor:
    """
    Convert (N,D) -> (N,D)
            (T,N,D) -> (T*N, D)
    so we can compute mean/std over the token axis.
    """
    if act.dim() == 2:
        return act
    if act.dim() == 3:
        t, n, d = act.shape
        return act.reshape(t * n, d)
    raise ValueError(f"Unsupported activation rank {act.dim()} with shape {tuple(act.shape)}")


def _maybe_strip_cls(act: torch.Tensor, source: str, use_class_tokens: bool) -> torch.Tensor:
    """
    Only strip CLS for 2D (N,D) activations; diffusion is (T,N,D) and should not be touched here.
    """
    if act.dim() == 2 and (source in MODELS_WITH_CLS_TOKEN) and (not use_class_tokens):
        # CLS token is at position 0
        return act[1:, :]
    return act


def _read_npz_key(npz: np.lib.npyio.NpzFile, key: str) -> Optional[np.ndarray]:
    if key in npz.files:
        return npz[key]
    return None


class ImageNetActivationDataset(ImageNet):
    def __init__(
        self,
        root: str,
        activation_root: str,
        sources: list,
        combined_npz: bool = False,
        split: str = "train",
        target_class: Union[str, int, None] = None,
        standardize: bool = False,
        divide_norm: bool = False,
        use_class_tokens: bool = True,
        return_metadata: bool = False,
        diffusion_models: Optional[list] = None,
        **kwargs,
    ):
        """
        Dataset for loading ImageNet activations with optional class filtering.

        Args:
            root: Root directory of the ImageNet dataset
            activation_root: Root directory containing activation files
            sources: List of model names to load from cached files
            combined_npz: if True, expect a single per-image *_combined.npz containing all sources
            split: 'train' or 'val'
            target_class: int class index, str class name, "ALL", or None
            standardize: standardize activations using computed mean/std per source
            divide_norm: normalize activations by their L2 norm (token-wise)
            use_class_tokens: include CLS tokens for CLS-bearing vision models
            return_metadata: if True, __getitem__ returns ((acts_dict, metadata), target)
            diffusion_models: optional list of which sources should be treated as diffusion
                             (only used to decide which metadata keys to look for / return)
        """
        super().__init__(root=root, split=split, **kwargs)

        # Filter to target class if specified
        if target_class is not None and target_class != "ALL":
            if isinstance(target_class, str):
                if target_class not in self.class_to_idx:
                    raise ValueError(f"Invalid ImageNet class name: {target_class}")
                target_class = self.class_to_idx[target_class]
            self.samples = [(path, idx) for path, idx in self.samples if idx == target_class]
            print(f"Filtered to {len(self.samples)} samples for class {target_class}")
        else:
            print("Training on all classes")

        self.activation_root = activation_root
        self.sources = sources
        self.combined_npz = combined_npz
        self.standardize = standardize
        self.divide_norm = divide_norm
        self.split = split
        self.use_class_tokens = use_class_tokens
        self.return_metadata = return_metadata
        self.diffusion_models = set(diffusion_models or [])

        # Precompute activation file paths aligned with ImageNet samples
        if self.combined_npz:
            self.samples_used = []
            for img_path, target in tqdm(self.samples, total=len(self.samples), desc="Indexing combined npz"):
                rel_path = os.path.relpath(img_path, os.path.join(self.root, self.split))
                class_dir = os.path.dirname(rel_path)
                act_path = os.path.join(
                    self.activation_root,
                    self.split,
                    class_dir,
                    os.path.basename(img_path).replace(".JPEG", "_combined.npz"),
                )
                self.samples_used.append((act_path, target))
        else:
            self.samples_used = {}
            for source in self.sources:
                self.samples_used[source] = []
                for img_path, target in tqdm(self.samples, total=len(self.samples), desc=f"Indexing {source} npz"):
                    rel_path = os.path.relpath(img_path, os.path.join(self.root, self.split))
                    class_dir = os.path.dirname(rel_path)
                    act_path = os.path.join(
                        self.activation_root,
                        self.split,
                        class_dir,
                        os.path.basename(img_path).replace(".JPEG", "_" + source + ".npz"),
                    )
                    self.samples_used[source].append((act_path, target))

        # Calculate standardization stats if needed
        if self.standardize:
            self._compute_standardization_stats()

    def _compute_standardization_stats(self, sample_size: int = 1000):
        """
        Compute sample mean and std of activations for standardization.

        Works for both:
          - vision activations (N,D)
          - diffusion activations (T,N,D) (flattened to tokens for stats)
        """
        self.standardization_stats = {}

        sample_size = min(sample_size, len(self.samples))
        sample_indices = np.random.choice(len(self.samples), sample_size, replace=False)

        for source in self.sources:
            print(f"Computing standardization stats for {source}")

            token_rows = []
            for idx in tqdm(sample_indices, desc=f"Stats {source}"):
                if self.combined_npz:
                    act_path, _ = self.samples_used[idx]
                    npz = np.load(act_path, mmap_mode="r")
                    raw = _read_npz_key(npz, source)
                    if raw is None:
                        raise KeyError(f"Missing key '{source}' in combined npz: {act_path}")
                    act = torch.from_numpy(raw)
                else:
                    act_path, _ = self.samples_used[source][idx]
                    npz = np.load(act_path, mmap_mode="r")
                    raw = _read_npz_key(npz, "activation")
                    if raw is None:
                        raise KeyError(f"Missing key 'activation' in npz: {act_path}")
                    act = torch.from_numpy(raw)

                act = _maybe_strip_cls(act, source, self.use_class_tokens)
                token_rows.append(_flatten_tokens_for_stats(act))

            token_rows = torch.cat(token_rows, dim=0)  # (total_tokens, D)
            mean = token_rows.mean(dim=0)
            std = token_rows.std(dim=0)
            self.standardization_stats[source] = {"mean": mean, "std": std}

            print(f"[{source}] mean.shape={tuple(mean.shape)} std.shape={tuple(std.shape)}")

    def _load_combined_npz(self, act_path: str) -> Dict[str, torch.Tensor]:
        npz_file = np.load(act_path, mmap_mode="r")
        return {key: torch.from_numpy(npz_file[key]) for key in npz_file.files}

    def _extract_metadata_from_combined(
        self,
        act_dict: Dict[str, torch.Tensor],
        index: int,
    ) -> Dict[str, Any]:
        """
        Build a metadata dict usable by temporal trainers.

        We support:
          - per-model sigmas:  "<MODEL>__sigmas" (T,)
          - per-model timesteps: "<MODEL>__timesteps" (T,)
          - global sigmas: "sigmas" (T,)
          - global timesteps: "timesteps" (T,)

        Returned metadata:
          - "sigmas": Tensor(B,T)   (picked from the first diffusion model found, if possible)
          - "timesteps": Tensor(B,T) optional, same rule
          - "sigmas_by_model": Dict[str, Tensor(B,T)]
          - "timesteps_by_model": Dict[str, Tensor(B,T)]
        """
        metadata: Dict[str, Any] = {}

        sigmas_by_model: Dict[str, torch.Tensor] = {}
        timesteps_by_model: Dict[str, torch.Tensor] = {}

        # Identify diffusion models by either:
        #   (1) explicit diffusion_models list
        #   (2) activation rank == 3 (T,N,D)
        diffusion_sources = []
        for s in self.sources:
            if s in act_dict and torch.is_tensor(act_dict[s]) and _is_diffusion_activation(act_dict[s]):
                diffusion_sources.append(s)
            elif s in self.diffusion_models:
                diffusion_sources.append(s)

        diffusion_sources = list(dict.fromkeys(diffusion_sources))  # unique preserve order

        # Helper to broadcast (T,) -> (1,T)
        def _to_bt(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 1:
                return x.unsqueeze(0)
            return x

        # Extract per-model
        for s in diffusion_sources:
            k_sig = f"{s}__sigmas"
            k_t   = f"{s}__timesteps"
            if k_sig in act_dict:
                sigmas_by_model[s] = _to_bt(act_dict[k_sig].float().cpu())
            if k_t in act_dict:
                timesteps_by_model[s] = _to_bt(act_dict[k_t].float().cpu())

        # Extract global if present
        if "sigmas" in act_dict:
            metadata["sigmas"] = _to_bt(act_dict["sigmas"].float().cpu())
        if "timesteps" in act_dict:
            metadata["timesteps"] = _to_bt(act_dict["timesteps"].float().cpu())

        # If no global sigmas, pick first available per-model
        if "sigmas" not in metadata and len(sigmas_by_model) > 0:
            first = next(iter(sigmas_by_model.keys()))
            metadata["sigmas"] = sigmas_by_model[first]

        if "timesteps" not in metadata and len(timesteps_by_model) > 0:
            first = next(iter(timesteps_by_model.keys()))
            metadata["timesteps"] = timesteps_by_model[first]

        metadata["sigmas_by_model"] = sigmas_by_model
        metadata["timesteps_by_model"] = timesteps_by_model
        metadata["diffusion_sources"] = diffusion_sources
        metadata["index"] = index

        return metadata

    def __getitem__(self, index: int):
        """
        Returns:
          if return_metadata == False:
              (activations_dict, target)
          else:
              ((activations_dict, metadata), target)

        activations_dict values are either:
          - vision: (N, D)
          - diffusion: (T, N, D)
        """
        if self.combined_npz:
            act_path, target = self.samples_used[index]
            act_dict = self._load_combined_npz(act_path)
        else:
            target = None
            act_dict = {}

        activations: Dict[str, torch.Tensor] = {}

        # Load per-source activations
        for source in self.sources:
            if self.combined_npz:
                if source not in act_dict:
                    raise KeyError(f"Missing key '{source}' in combined npz: {self.samples_used[index][0]}")
                act = act_dict[source]
            else:
                act_path, target = self.samples_used[source][index]
                npz = np.load(act_path, mmap_mode="r")
                raw = _read_npz_key(npz, "activation")
                if raw is None:
                    raise KeyError(f"Missing key 'activation' in npz: {act_path}")
                act = torch.from_numpy(raw)

            act = _maybe_strip_cls(act, source, self.use_class_tokens)

            # standardize/normalize
            if self.standardize:
                mean = self.standardization_stats[source]["mean"]
                std = self.standardization_stats[source]["std"]

                # broadcast mean/std over tokens (and timesteps if present)
                # act: (N,D) or (T,N,D)
                act = (act - mean) / (std + 1e-5)
            elif self.divide_norm:
                # token-wise norm along last dim; works for both (N,D) and (T,N,D)
                act = act / (act.norm(dim=-1, keepdim=True) + 1e-9)

            activations[source] = act

        if not self.return_metadata:
            return activations, int(target)

        if not self.combined_npz:
            raise ValueError("return_metadata=True is only supported with combined_npz=True (per-image combined files).")

        metadata = self._extract_metadata_from_combined(act_dict, index)

        # Make sure sigmas is present if there are diffusion sources
        if len(metadata.get("diffusion_sources", [])) > 0 and "sigmas" not in metadata:
            raise KeyError(
                "Detected diffusion activations but could not find sigmas in combined npz. "
                "Expected 'sigmas' or '<MODEL>__sigmas' keys."
            )

        return (activations, metadata), int(target)

    def __len__(self) -> int:
        return len(self.samples)


def load_directory(directory):
    """
    Load all images from a directory.

    Returns:
        list[PIL.Image]
    """
    paths = os.listdir(directory)
    paths = [path for path in paths if path.endswith((".jpg", ".jpeg", ".png"))]
    paths = sorted(paths)

    images = []
    for path in paths:
        try:
            img = Image.open(os.path.join(directory, path)).convert("RGB")
            images.append(img)
        except OSError:
            continue
    return images
