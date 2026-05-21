"""
spatial_align.py

Spatially-pools per-model activation token grids to a common grid size,
preserving image-space correspondence between models with different
patch resolutions (e.g. DinoV2's 16x16 grid and PixArt's 32x32 grid).

This is Option B1 in the cross-model alignment discussion: instead of mean-
pooling all tokens to a single CLS-like vector, we leverage the fact that
both encoders see the same image and produce square patch grids over it.
A 2x2 average-pool of PixArt's tokens collapses each 2x2 PixArt patch block
into the DinoV2 patch it spatially overlaps with, giving both models the
same N tokens with genuine per-position correspondence.

Usage in train.py / inference.py:

    from spatial_align import SpatialAligner, infer_grid_size

    aligner = SpatialAligner(
        native_grid_sizes={"DinoV2": 16, "PixArt": 32},
        target_grid_size=16,
    )

    # x: (B, N_src, D) from the dataloader
    x_aligned = aligner.align(x, source="PixArt")   # -> (B, 256, D)

The target grid size must divide every model's native grid size; otherwise
this raises a ValueError. For DinoV2 (16) + PixArt (32) -> target 16: valid.
For SD3 (32) + PixArt (32) + DinoV2 (16) -> target must be 16, or you can
exclude DinoV2 by setting target_grid_size=32 (PixArt and SD3 unchanged,
DinoV2 would need upsampling which we don't support — keep DinoV2 out of
the run, or set target to its grid size).

Notes:
- Operates on square grids only (N = G*G). If your activations include a
  CLS token (N = G*G + 1), strip it before calling .align(). With
  use_class_tokens=false in the config this is already the case.
- The pooled features are model-native (we do NOT change D). Each model's
  encoder still maps from its own D to the shared latent_dim.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def infer_grid_size(n_tokens: int) -> int:
    """
    Return G such that G*G == n_tokens. Raises if n_tokens is not a perfect square.
    """
    g = int(round(math.sqrt(n_tokens)))
    if g * g != n_tokens:
        raise ValueError(
            f"Cannot infer square grid: n_tokens={n_tokens} is not a perfect square. "
            "If your activations include a CLS or register tokens, strip them before "
            "calling spatial_align (set use_class_tokens=false in config)."
        )
    return g


class SpatialAligner:
    """
    Pools per-model activations from their native G_src x G_src token grid down
    to a common target G_tgt x G_tgt grid via 2D average pooling.

    Args:
        native_grid_sizes: model_name -> native square grid size G (e.g. {"DinoV2": 16, "PixArt": 32})
        target_grid_size: target square grid size G_tgt (e.g. 16). Must divide every native grid.
    """

    def __init__(
        self,
        native_grid_sizes: Dict[str, int],
        target_grid_size: int,
    ):
        if target_grid_size <= 0:
            raise ValueError(f"target_grid_size must be positive, got {target_grid_size}")

        self.native_grid_sizes = dict(native_grid_sizes)
        self.target_grid_size = int(target_grid_size)

        for name, g in self.native_grid_sizes.items():
            if g < self.target_grid_size:
                raise ValueError(
                    f"Model {name!r} has grid size {g} < target {self.target_grid_size}. "
                    "Upsampling is not supported — pick a target grid size that is "
                    "no larger than the smallest model's native grid."
                )
            if g % self.target_grid_size != 0:
                raise ValueError(
                    f"Model {name!r}'s native grid {g} is not an integer multiple of "
                    f"target {self.target_grid_size}. Pick a target that divides all natives."
                )

    @property
    def target_n_tokens(self) -> int:
        return self.target_grid_size * self.target_grid_size

    def pool_factor(self, model: str) -> int:
        """How many native tokens collapse into one target token along each axis."""
        return self.native_grid_sizes[model] // self.target_grid_size

    def align(self, x: torch.Tensor, source: str) -> torch.Tensor:
        """
        x: (B, N_src, D) where N_src = G_src * G_src.
        Returns:
            (B, N_tgt, D) where N_tgt = target_grid_size ** 2.

        If the model is already at the target grid size, returns x unchanged.
        """
        if x.dim() != 3:
            raise ValueError(f"align expected (B, N, D), got {tuple(x.shape)}")

        if source not in self.native_grid_sizes:
            raise KeyError(
                f"No native grid size registered for source={source!r}. "
                f"Known: {sorted(self.native_grid_sizes.keys())}"
            )

        g_src = self.native_grid_sizes[source]
        g_tgt = self.target_grid_size

        # Sanity check that the activation tensor matches the registered grid
        expected = g_src * g_src
        if x.shape[1] != expected:
            raise ValueError(
                f"Source {source!r} expected N={expected} (grid {g_src}x{g_src}), "
                f"got N={x.shape[1]}. If your activations include a CLS/register "
                "token, strip it before calling align()."
            )

        if g_src == g_tgt:
            return x

        B, N, D = x.shape
        k = g_src // g_tgt  # pool factor per axis

        # (B, N, D) -> (B, G, G, D) -> (B, D, G, G) -> avg_pool -> (B, D, g_tgt, g_tgt) -> (B, N_tgt, D)
        x_grid = x.reshape(B, g_src, g_src, D).permute(0, 3, 1, 2)  # (B, D, G, G)
        x_pooled = F.avg_pool2d(x_grid, kernel_size=k, stride=k)     # (B, D, g_tgt, g_tgt)
        x_out = x_pooled.permute(0, 2, 3, 1).reshape(B, g_tgt * g_tgt, D)
        return x_out

    def effective_token_counts(self) -> Dict[str, int]:
        """Post-alignment token count for each model (all equal to target_n_tokens)."""
        return {name: self.target_n_tokens for name in self.native_grid_sizes}


def build_spatial_aligner_from_config(
    config_global: dict,
    model_tokens: Dict[str, int],
) -> Optional[SpatialAligner]:
    """
    Construct a SpatialAligner from a config dict, or None if alignment is disabled.

    Reads from config_global:
        spatial_align_to: model name to whose grid we pool, e.g. "DinoV2".
            If absent / null / empty, returns None.
        spatial_align_grid_overrides (optional): explicit overrides for
            native grid sizes (model_name -> grid_size). Otherwise grid sizes
            are inferred from model_tokens via sqrt.

    Returns:
        SpatialAligner or None.
    """
    target_model = config_global.get("spatial_align_to", None)
    if target_model is None or (isinstance(target_model, str) and not target_model.strip()):
        return None

    overrides = config_global.get("spatial_align_grid_overrides", {}) or {}

    native = {}
    for name, n in model_tokens.items():
        if name in overrides:
            native[name] = int(overrides[name])
        else:
            native[name] = infer_grid_size(int(n))

    if target_model not in native:
        raise KeyError(
            f"spatial_align_to={target_model!r} is not in model_tokens "
            f"(have {sorted(native.keys())})"
        )

    return SpatialAligner(
        native_grid_sizes=native,
        target_grid_size=native[target_model],
    )
