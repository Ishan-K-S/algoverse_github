"""
spatial_align.py

Spatially resamples per-model activation token grids to a common grid size,
preserving image-space correspondence between models with different patch
resolutions (for example, DinoV2's 16x16 grid and PixArt's 32x32 grid).

This version supports upsizing smaller grids instead of only downsampling
larger grids. To align DinoV2 to PixArt, set the target grid to PixArt's
native grid:

    from spatial_align import SpatialAligner, infer_grid_size

    aligner = SpatialAligner(
        native_grid_sizes={"DinoV2": 16, "PixArt": 32},
        target_grid_size=32,
    )

    dino = aligner.align(dino, source="DinoV2")    # -> (B, 1024, D_dino)
    pixart = aligner.align(pixart, source="PixArt")  # -> unchanged

By default, upsampling uses nearest-neighbor expansion, which repeats each
DinoV2 token into the 2x2 PixArt patch block it spatially covers. This keeps
the alignment literal rather than inventing blended intermediate features.

The target grid size must be commensurate with every model's native grid:
- if native > target, native must be an integer multiple of target
- if native < target, target must be an integer multiple of native

Notes:
- Operates on square grids only (N = G*G). If your activations include a
  CLS token (N = G*G + 1), strip it before calling .align(). With
  use_class_tokens=false in the config this is already the case.
- The resampled features are model-native (we do NOT change D). Each model's
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
    Resamples per-model activations from their native G_src x G_src token grid
    to a common target G_tgt x G_tgt grid.

    Larger native grids are downsampled with 2D average pooling. Smaller native
    grids are upsampled, defaulting to nearest-neighbor expansion so one source
    token fills the corresponding block of target tokens.

    Args:
        native_grid_sizes: model_name -> native square grid size G
            (for example, {"DinoV2": 16, "PixArt": 32})
        target_grid_size: target square grid size G_tgt. Use 32 to upsize
            DinoV2 to PixArt's grid.
        upsample_mode: "nearest", "bilinear", or "bicubic". "nearest" repeats
            tokens exactly for integer scale factors.
    """

    _UPSAMPLE_MODES = {"nearest", "bilinear", "bicubic"}

    def __init__(
        self,
        native_grid_sizes: Dict[str, int],
        target_grid_size: int,
        upsample_mode: str = "nearest",
    ):
        if target_grid_size <= 0:
            raise ValueError(f"target_grid_size must be positive, got {target_grid_size}")
        if upsample_mode not in self._UPSAMPLE_MODES:
            raise ValueError(
                f"upsample_mode must be one of {sorted(self._UPSAMPLE_MODES)}, "
                f"got {upsample_mode!r}"
            )

        self.native_grid_sizes = {name: int(g) for name, g in native_grid_sizes.items()}
        self.target_grid_size = int(target_grid_size)
        self.upsample_mode = upsample_mode

        for name, g in self.native_grid_sizes.items():
            if g <= 0:
                raise ValueError(f"Model {name!r} has non-positive grid size {g}.")
            if g == self.target_grid_size:
                continue
            if g > self.target_grid_size and g % self.target_grid_size != 0:
                raise ValueError(
                    f"Model {name!r}'s native grid {g} is not an integer multiple of "
                    f"target {self.target_grid_size}. Cannot average-pool cleanly."
                )
            if g < self.target_grid_size and self.target_grid_size % g != 0:
                raise ValueError(
                    f"Target grid {self.target_grid_size} is not an integer multiple of "
                    f"model {name!r}'s native grid {g}. Cannot upsample cleanly."
                )

    @property
    def target_n_tokens(self) -> int:
        return self.target_grid_size * self.target_grid_size

    def resample_direction(self, model: str) -> str:
        """Return 'identity', 'downsample', or 'upsample' for a registered model."""
        g = self.native_grid_sizes[model]
        if g == self.target_grid_size:
            return "identity"
        if g > self.target_grid_size:
            return "downsample"
        return "upsample"

    def resample_factor(self, model: str) -> int:
        """Integer factor between native and target grid sizes along each axis."""
        g = self.native_grid_sizes[model]
        return max(g, self.target_grid_size) // min(g, self.target_grid_size)

    def pool_factor(self, model: str) -> int:
        """How many native tokens collapse into one target token along each axis."""
        if self.resample_direction(model) == "upsample":
            raise ValueError(
                f"Model {model!r} is upsampled from {self.native_grid_sizes[model]} "
                f"to {self.target_grid_size}; use resample_factor() instead."
            )
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

        expected = g_src * g_src
        if x.shape[1] != expected:
            raise ValueError(
                f"Source {source!r} expected N={expected} (grid {g_src}x{g_src}), "
                f"got N={x.shape[1]}. If your activations include a CLS/register "
                "token, strip it before calling align()."
            )

        if g_src == g_tgt:
            return x

        batch_size, _n_tokens, dim = x.shape
        x_grid = x.reshape(batch_size, g_src, g_src, dim).permute(0, 3, 1, 2)

        if g_src > g_tgt:
            k = g_src // g_tgt
            x_resampled = F.avg_pool2d(x_grid, kernel_size=k, stride=k)
        else:
            k = g_tgt // g_src
            if self.upsample_mode == "nearest":
                x_resampled = x_grid.repeat_interleave(k, dim=2).repeat_interleave(k, dim=3)
            else:
                x_resampled = F.interpolate(
                    x_grid,
                    size=(g_tgt, g_tgt),
                    mode=self.upsample_mode,
                    align_corners=False,
                )

        return x_resampled.permute(0, 2, 3, 1).reshape(batch_size, g_tgt * g_tgt, dim)

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
        spatial_align_to: model name to whose grid we resample, e.g. "PixArt".
            If absent / null / empty, returns None.
        spatial_align_grid_overrides (optional): explicit overrides for
            native grid sizes (model_name -> grid_size). Otherwise grid sizes
            are inferred from model_tokens via sqrt.
        spatial_align_upsample_mode (optional): "nearest", "bilinear", or
            "bicubic". Defaults to "nearest".

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
        upsample_mode=str(config_global.get("spatial_align_upsample_mode", "nearest")),
    )
