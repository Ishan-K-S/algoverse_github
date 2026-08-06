"""
Module for loading COCO Activations Dataset.

Supports BOTH:
  (A) Vision encoders cached as (N, D) per image
  (B) Diffusion models cached as (T, N, D) per image, plus sigma/timestep metadata

Cache directory layout (flat — no class subdirectories):
  <cache_root>/<img_stem>_<MODEL>.npz          (non-combined mode)
  <cache_root>/<img_stem>_combined.npz         (combined_npz mode)

Vision npz keys:   activation (N, D)
Diffusion npz keys: activation (T, N, D), sigmas (T,), timesteps (T,)
Combined npz keys:  <MODEL> (N,D or T,N,D),  <MODEL>__sigmas (T,),  <MODEL>__timesteps (T,)

Key differences from the old ImageNet version:
  - Inherits from torch.utils.data.Dataset (not torchvision ImageNet)
  - No class-label filtering (COCO has no class index)
  - Flat directory: paths are <cache_root>/<filename_stem>_<source>.npz
  - Image filenames discovered from the first source's cached npz files
"""

import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import Dataset
from typing import Tuple, Dict, Union, Optional, Any, List


# Models that have a CLS token prepended to patch tokens.
# ViT, DinoV2, and CLIP all have CLS tokens at position 0.
# SigLIP does NOT have a CLS token.
MODELS_WITH_CLS_TOKEN = {"DinoV2", "ViT", "CLIP"}

DIFF_META_FIELDS = ("sigmas", "timesteps")


def _is_diffusion_activation(act: torch.Tensor) -> bool:
    """Diffusion cached activations are (T, N, D); vision are (N, D)."""
    return act.dim() == 3


def _flatten_tokens_for_stats(act: torch.Tensor) -> torch.Tensor:
    """
    (N, D) -> (N, D)
    (T, N, D) -> (T*N, D)
    """
    if act.dim() == 2:
        return act
    if act.dim() == 3:
        t, n, d = act.shape
        return act.reshape(t * n, d)
    raise ValueError(f"Unsupported activation rank {act.dim()} with shape {tuple(act.shape)}")


def _maybe_strip_cls(act: torch.Tensor, source: str, use_class_tokens: bool) -> torch.Tensor:
    """Strip CLS token for 2-D (N, D) vision activations only."""
    if act.dim() == 2 and (source in MODELS_WITH_CLS_TOKEN) and (not use_class_tokens):
        return act[1:, :]
    return act


def _read_npz_key(npz: np.lib.npyio.NpzFile, key: str) -> Optional[np.ndarray]:
    if key in npz.files:
        return npz[key]
    return None


def _img_stem(filename: str) -> str:
    """Return filename without extension, e.g. '000000001234.jpg' -> '000000001234'."""
    return os.path.splitext(filename)[0]


class CocoActivationDataset(Dataset):
    """
    Dataset for loading cached COCO activations from a flat directory.

    Parameters
    ----------
    cache_root : str
        Root directory containing cached .npz activation files.
    sources : list[str]
        List of model names to load (e.g. ["ViT", "DinoV2", "SD3"]).
    combined_npz : bool
        If True, expect one ``*_combined.npz`` per image containing all sources.
        If False, expect separate ``*_<source>.npz`` per image per source.
    standardize : bool
        Standardise activations using mean/std computed over a sample.
    divide_norm : bool
        Normalise each token vector by its L2 norm (mutually exclusive with standardize).
    use_class_tokens : bool
        If True, keep the CLS token for models that have one (ViT, CLIP, DinoV2).
    return_metadata : bool
        If True, __getitem__ returns ((acts_dict, metadata), dummy_target).
        Only supported when combined_npz=True.
    diffusion_models : list[str] | None
        Names of sources that are diffusion/flow models.  Used to decide which
        metadata keys (sigmas, timesteps) to extract.
    """

    def __init__(
        self,
        cache_root: str,
        sources: List[str],
        combined_npz: bool = False,
        standardize: bool = False,
        divide_norm: bool = False,
        use_class_tokens: bool = True,
        return_metadata: bool = False,
        diffusion_models: Optional[List[str]] = None,
        standardization_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        stats_seed: Optional[int] = None,
        spatial_aligner: Optional[Any] = None,
    ):
        super().__init__()

        self.cache_root = cache_root
        self.sources = sources
        self.combined_npz = combined_npz
        self.standardize = standardize
        self.divide_norm = divide_norm
        self.use_class_tokens = use_class_tokens
        self.return_metadata = return_metadata
        self.diffusion_models = set(diffusion_models or [])
        self.stats_seed = stats_seed
        # Used ONLY to compute standardization stats on the post-alignment
        # distribution (REPAIR_PLAN.md V6) -- NOT applied to the activations
        # __getitem__ returns. Per-channel affine (standardize) and spatial
        # average-pooling (align) are both linear in the token/spatial axis,
        # so they commute exactly: (avgpool(x) - mean) / std == avgpool((x -
        # mean) / std) for the same per-channel mean/std constants. The actual
        # bug was never *where* standardization runs relative to alignment --
        # it was that the mean/std were fit to individual tokens' variance
        # instead of the pooled token's variance, which is lower (averaging
        # correlated neighbours reduces variance). Computing stats here on
        # aligned samples fixes the constants that get used; leaving
        # __getitem__'s and train.py's existing standardize-then-align
        # sequence untouched is equivalent, not a shortcut.
        self.spatial_aligner = spatial_aligner

        # ------------------------------------------------------------------ #
        # Discover image stems from existing cache files.
        # We anchor on the first source (or the combined suffix) so every
        # returned sample is guaranteed to have at least that file present.
        # ------------------------------------------------------------------ #
        if combined_npz:
            anchor_suffix = "_combined.npz"
        else:
            anchor_suffix = f"_{sources[0]}.npz"

        all_files = os.listdir(cache_root)
        self.stems: List[str] = sorted(
            f[: -len(anchor_suffix)]
            for f in all_files
            if f.endswith(anchor_suffix)
        )

        if len(self.stems) == 0:
            raise RuntimeError(
                f"No cached files found in '{cache_root}' with suffix '{anchor_suffix}'. "
                "Make sure you have run the caching scripts first."
            )

        print(f"[CocoActivationDataset] Found {len(self.stems)} images "
              f"(combined_npz={combined_npz}, sources={sources})")

        # ------------------------------------------------------------------ #
        # Pre-build full paths for each stem × source (non-combined mode)
        # ------------------------------------------------------------------ #
        if not combined_npz:
            # self.source_paths[source][i] = path to npz for stem i
            self.source_paths: Dict[str, List[str]] = {}
            for source in sources:
                self.source_paths[source] = [
                    os.path.join(cache_root, f"{stem}_{source}.npz")
                    for stem in self.stems
                ]

        # ------------------------------------------------------------------ #
        # Standardisation stats
        # ------------------------------------------------------------------ #
        if standardize:
            if standardization_stats is None:
                self._compute_standardization_stats()
            else:
                self.standardization_stats = self._validate_standardization_stats(
                    standardization_stats
                )

    # ---------------------------------------------------------------------- #
    # Standardisation helpers
    # ---------------------------------------------------------------------- #

    def _apply_spatial_align_for_stats(self, act: torch.Tensor, source: str) -> torch.Tensor:
        """
        Apply self.spatial_aligner (if any) to a per-image activation before
        it feeds the Welford accumulator below, so the computed std reflects
        the POOLED token's variance, not each individual native token's
        variance (REPAIR_PLAN.md V6). Averaging correlated neighbouring
        tokens together lowers variance below 1 even when every input token
        is already unit-variance, so a std fit to un-pooled tokens
        under-normalizes whatever this source looks like after alignment.

        SpatialAligner.align expects (B, N, D). A 2-D vision activation
        (N, D) gets a throwaway batch dim; a 3-D diffusion activation
        (T, N, D) already has a leading axis that plays that role exactly
        (pooling only touches the N/D axes, so this is correct regardless of
        which single timestep training eventually slices out of T).
        """
        if self.spatial_aligner is None or source not in self.spatial_aligner.native_grid_sizes:
            return act
        if act.dim() == 2:
            return self.spatial_aligner.align(act.unsqueeze(0), source=source).squeeze(0)
        if act.dim() == 3:
            return self.spatial_aligner.align(act, source=source)
        raise ValueError(
            f"Unsupported activation rank {act.dim()} for spatial alignment: {tuple(act.shape)}"
        )

    def _compute_standardization_stats(self, sample_size: int = 1000):
        """
        Compute per-source mean and std from a random sample of the cache.

        Uses Welford's online algorithm to avoid accumulating all tensors
        in memory — critical for large diffusion activations like SD3/FLUX
        which can be (T, 1024, 1536) or (T, 4096, 3072) per image.

        Works for both vision (N, D) and diffusion (T, N, D) activations.
        """
        self.standardization_stats: Dict[str, Dict[str, torch.Tensor]] = {}

        sample_size = min(sample_size, len(self.stems))
        rng = np.random.default_rng(self.stats_seed)
        sample_indices = rng.choice(len(self.stems), sample_size, replace=False)

        for source in self.sources:
            print(f"[stats] Computing standardisation stats for {source} …")

            # Welford's online mean/variance accumulators
            # Initialised lazily on first batch so we don't need to know D upfront.
            n_seen: int = 0
            welford_mean: Optional[torch.Tensor] = None
            welford_M2:   Optional[torch.Tensor] = None   # sum of squared deviations

            for idx in tqdm(sample_indices, desc=f"Stats {source}"):
                act = self._load_activation(source, int(idx))
                act = _maybe_strip_cls(act, source, self.use_class_tokens)
                act = self._apply_spatial_align_for_stats(act, source)
                tokens = _flatten_tokens_for_stats(act).float()  # (K, D)

                if welford_mean is None:
                    D = tokens.shape[-1]
                    welford_mean = torch.zeros(D, dtype=torch.float32)
                    welford_M2   = torch.zeros(D, dtype=torch.float32)

                # Update accumulators one token at a time via batch Welford update
                # Batch update: process all K tokens from this image at once.
                K = tokens.shape[0]
                batch_mean = tokens.mean(dim=0)   # (D,)
                batch_var  = tokens.var(dim=0, unbiased=False)  # (D,)

                # Merge running stats with this batch (Chan et al. parallel formula)
                n_new   = n_seen + K
                delta   = batch_mean - welford_mean
                welford_mean = welford_mean + delta * (K / n_new)
                welford_M2   = (
                    welford_M2
                    + batch_var * K
                    + delta ** 2 * (n_seen * K / n_new)
                )
                n_seen = n_new

                # Explicitly free the tensor to avoid memory accumulation
                del tokens, act

            if n_seen < 2 or welford_mean is None:
                raise RuntimeError(f"[stats] Not enough samples to compute stats for {source}")

            mean = welford_mean
            std  = (welford_M2 / n_seen).sqrt()   # population std
            std  = torch.clamp(std, min=1e-5)      # guard against zero-std features

            self.standardization_stats[source] = {"mean": mean, "std": std}
            print(f"  [{source}] mean.shape={tuple(mean.shape)}  std.shape={tuple(std.shape)}")

    def _validate_standardization_stats(
        self,
        stats: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Validate and normalize persisted training statistics.

        Evaluation must use the exact mean/std used during training.  Recomputing
        a random cache sample at inference changes the SAE's input coordinate
        system and can make a few global features dominate every image.
        """
        normalized: Dict[str, Dict[str, torch.Tensor]] = {}
        for source in self.sources:
            if source not in stats:
                raise KeyError(
                    f"Persisted standardization stats are missing source {source!r}."
                )
            source_stats = stats[source]
            if "mean" not in source_stats or "std" not in source_stats:
                raise KeyError(
                    f"Persisted standardization stats for {source!r} require 'mean' and 'std'."
                )
            mean = torch.as_tensor(source_stats["mean"], dtype=torch.float32).cpu()
            std = torch.as_tensor(source_stats["std"], dtype=torch.float32).cpu()
            if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
                raise ValueError(
                    f"Invalid persisted standardization stats for {source!r}: "
                    f"mean={tuple(mean.shape)}, std={tuple(std.shape)}"
                )
            normalized[source] = {"mean": mean, "std": std.clamp_min(1e-5)}
        print("[stats] Using standardization statistics persisted in the checkpoint.")
        return normalized

    # ---------------------------------------------------------------------- #
    # Low-level loaders
    # ---------------------------------------------------------------------- #

    def _load_activation(self, source: str, index: int) -> torch.Tensor:
        """
        Load the raw activation tensor for a single (source, index) pair.
        Returns (N, D) for vision models or (T, N, D) for diffusion models.
        Handles both combined_npz and per-source npz modes.
        """
        if self.combined_npz:
            path = os.path.join(self.cache_root, f"{self.stems[index]}_combined.npz")
            try:
                npz = np.load(path, mmap_mode="r")
            except Exception as e:
                raise RuntimeError(f"Failed to load combined npz: {path}. The file may be corrupted or was interrupted during saving.") from e
            raw = _read_npz_key(npz, source)
            if raw is None:
                raise KeyError(f"Key '{source}' missing in combined npz: {path}")
        else:
            path = self.source_paths[source][index]
            try:
                npz = np.load(path, mmap_mode="r")
            except Exception as e:
                raise RuntimeError(f"Failed to load source npz: {path}. The file may be corrupted or was interrupted during saving.") from e
            raw = _read_npz_key(npz, "activation")
            if raw is None:
                raise KeyError(f"Key 'activation' missing in npz: {path}")

        return torch.from_numpy(raw.copy())

    def _load_combined_npz_full(self, index: int) -> Dict[str, torch.Tensor]:
        """Load every key from the combined npz as a dict of tensors."""
        path = os.path.join(self.cache_root, f"{self.stems[index]}_combined.npz")
        try:
            npz = np.load(path, mmap_mode="r")
        except Exception as e:
            raise RuntimeError(f"Failed to load combined npz: {path}. The file may be corrupted or was interrupted during saving.") from e
        return {key: torch.from_numpy(npz[key].copy()) for key in npz.files}

    # ---------------------------------------------------------------------- #
    # Metadata extraction
    # ---------------------------------------------------------------------- #

    def _extract_metadata(
        self,
        act_dict: Dict[str, torch.Tensor],
        index: int,
    ) -> Dict[str, Any]:
        """
        Build a metadata dict compatible with train.py.

        Supported combined-npz keys:
          <MODEL>__sigmas    (T,)
          <MODEL>__timesteps (T,)

        Returned keys:
          sigmas             (1, T)  – from first diffusion model found
          timesteps          (1, T)  – optional
          sigmas_by_model    Dict[str, Tensor(1, T)]
          timesteps_by_model Dict[str, Tensor(1, T)]
          diffusion_sources  List[str]
          index              int
        """
        metadata: Dict[str, Any] = {}
        sigmas_by_model: Dict[str, torch.Tensor] = {}
        timesteps_by_model: Dict[str, torch.Tensor] = {}

        # Determine which sources are diffusion
        diffusion_sources = []
        for s in self.sources:
            if s in self.diffusion_models:
                diffusion_sources.append(s)
            elif s in act_dict and _is_diffusion_activation(act_dict[s]):
                diffusion_sources.append(s)
        diffusion_sources = list(dict.fromkeys(diffusion_sources))

        def _to_bt(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(0) if x.dim() == 1 else x

        for s in diffusion_sources:
            k_sig = f"{s}__sigmas"
            k_ts  = f"{s}__timesteps"
            if k_sig in act_dict:
                sigmas_by_model[s] = _to_bt(act_dict[k_sig].float().cpu())
            if k_ts in act_dict:
                timesteps_by_model[s] = _to_bt(act_dict[k_ts].float().cpu())

        # Global fallbacks (written by old combine scripts)
        if "sigmas" in act_dict:
            metadata["sigmas"] = _to_bt(act_dict["sigmas"].float().cpu())
        if "timesteps" in act_dict:
            metadata["timesteps"] = _to_bt(act_dict["timesteps"].float().cpu())

        # Prefer per-model sigmas as global if global not present
        if "sigmas" not in metadata and sigmas_by_model:
            first = next(iter(sigmas_by_model))
            metadata["sigmas"] = sigmas_by_model[first]
        if "timesteps" not in metadata and timesteps_by_model:
            first = next(iter(timesteps_by_model))
            metadata["timesteps"] = timesteps_by_model[first]

        metadata["sigmas_by_model"]    = sigmas_by_model
        metadata["timesteps_by_model"] = timesteps_by_model
        metadata["diffusion_sources"]  = diffusion_sources
        metadata["index"]              = index

        return metadata

    # ---------------------------------------------------------------------- #
    # Dataset interface
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int):
        """
        Returns:
          if return_metadata is False:
              (activations_dict, 0)          # dummy label; COCO has no single label
          if return_metadata is True:
              ((activations_dict, metadata), 0)

        activations_dict values:
          vision:    (N, D)
          diffusion: (T, N, D)
        """
        act_dict_full: Dict[str, torch.Tensor] = {}
        if self.combined_npz and self.return_metadata:
            act_dict_full = self._load_combined_npz_full(index)

        activations: Dict[str, torch.Tensor] = {}

        for source in self.sources:
            if self.combined_npz and self.return_metadata:
                # Re-use already-loaded dict
                if source not in act_dict_full:
                    raise KeyError(
                        f"Key '{source}' missing in combined npz: "
                        f"{self.stems[index]}_combined.npz"
                    )
                act = act_dict_full[source]
            else:
                act = self._load_activation(source, index)

            act = _maybe_strip_cls(act, source, self.use_class_tokens)

            if self.standardize:
                mean = self.standardization_stats[source]["mean"]
                std  = self.standardization_stats[source]["std"]
                act  = (act - mean) / (std + 1e-5)
            elif self.divide_norm:
                act = act / (act.norm(dim=-1, keepdim=True) + 1e-9)

            activations[source] = act

        if not self.return_metadata:
            return activations, 0  # dummy label

        if not self.combined_npz:
            raise ValueError(
                "return_metadata=True requires combined_npz=True "
                "(metadata such as sigmas/timesteps is only stored in combined files)."
            )

        metadata = self._extract_metadata(act_dict_full, index)

        if metadata["diffusion_sources"] and "sigmas" not in metadata:
            raise KeyError(
                "Detected diffusion activations but could not find sigmas in combined npz. "
                "Expected '<MODEL>__sigmas' keys written by combine_cached_acts.py."
            )

        return (activations, metadata), 0


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #

def load_directory(directory: str):
    """Load all JPEG/PNG images from a directory. Returns list[PIL.Image]."""
    paths = sorted(
        p for p in os.listdir(directory)
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    images = []
    for p in paths:
        try:
            images.append(Image.open(os.path.join(directory, p)).convert("RGB"))
        except OSError:
            continue
    return images
