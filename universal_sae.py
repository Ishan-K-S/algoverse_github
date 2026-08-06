"""
UniversalSAE — stripped-down rewrite following:
  - Gao et al. "Scaling and evaluating sparse autoencoders" (TopK SAE)
  - Thasarathan et al. "Universal Sparse Autoencoders" (USAE), arXiv:2502.03714

Key changes vs. the previous architecture:
  - NO token pooling / unpooling. Encoder and decoder are pure linear maps R^d <-> R^m.
  - TopK is applied directly to the encoder pre-activation (no ReLU before TopK).
  - Per-model learnable mean b_pre is subtracted on encode and added back on decode.
  - Cross-model decoding (source != target) is only allowed when token counts match.
    For mismatched token counts (e.g. DinoV2:256 vs PixArt:1024), set
    cls_pool_mode='mean' to pool each model's tokens to a single CLS-like vector
    per image. This matches the paper's per-sample formulation.
  - Diffusion timestep modulation is still available for diffusion sources but is
    disabled-by-default since with a single-timestep training run it learns a
    constant affine and does no useful work.

Public interface (unchanged from previous version, so train.py / inference.py /
cross_model_overlap.py keep working):
    model.encode(x, source, sigma=None) -> (z_pre, z)
    model.decode(z, target, sigma=None) -> x_hat
    model.forward(x, source, target, sigma_src=None, sigma_tgt=None) -> (z_pre, z, x_hat)
    model.normalize_decoder_dictionaries_()
    model.diffusion_models : Set[str]
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Optional diffusion timestep modulation (TIDE-style). Kept for compatibility
# but its use is gated by `use_tide=False` by default — see UniversalSAE init.
# ---------------------------------------------------------------------------


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -scale)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TIDETemporalModulation(nn.Module):
    """Per-channel affine conditioned on diffusion timestep."""

    def __init__(self, dim: int, tdim: int = 256):
        super().__init__()
        self.t_embed = TimestepEmbedding(tdim)
        self.mlp = nn.Sequential(
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, 2 * dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _flatten_timestep(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            return t.unsqueeze(0).float()
        if t.dim() == 1:
            return t.float()
        return t.flatten(start_dim=1)[:, 0].float()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        t = self._flatten_timestep(t)
        if t.shape[0] != B:
            raise ValueError(f"Timestep batch mismatch: {tuple(t.shape)} for B={B}")
        h = self.t_embed(t)
        scale, shift = self.mlp(h).chunk(2, dim=-1)  # (B, D), (B, D)
        if x.dim() == 3:
            return x * (1 + scale[:, None, :]) + shift[:, None, :]
        return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Per-model TopK SAE (Gao et al. style)
# ---------------------------------------------------------------------------


class PerModelTopKSAE(nn.Module):
    """
    Encoder/decoder pair for one source model, following Gao et al.:

        z_pre = W_enc (x - b_pre)
        z     = TopK(z_pre)        # NO ReLU before TopK; TopK selects largest signed pre-acts
        x_hat = W_dec z + b_pre

    The same b_pre is shared between encode and decode (paper's exact formulation).
    Decoder columns are unit-normalized after each step via
    `normalize_decoder_columns_`, called from UniversalSAE.
    """

    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        self.b_pre = nn.Parameter(torch.zeros(in_dim))
        self.W_enc = nn.Linear(in_dim, latent_dim, bias=False)
        self.W_dec = nn.Linear(latent_dim, in_dim, bias=False)

        # Tie decoder to encoder transpose at init (Gao et al. suggestion)
        with torch.no_grad():
            self.W_dec.weight.copy_(self.W_enc.weight.t())
        self.normalize_decoder_columns_()

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_dim) -> (..., latent_dim). Subtracts per-model mean first."""
        return self.W_enc(x - self.b_pre)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., latent_dim) -> (..., in_dim). Adds per-model mean back."""
        return self.W_dec(z) + self.b_pre

    @torch.no_grad()
    def normalize_decoder_columns_(self, eps: float = 1e-8) -> None:
        # For nn.Linear(latent_dim -> in_dim), each latent feature is one column
        # of self.W_dec.weight, shaped (in_dim, latent_dim). Columns = dim=0.
        w = self.W_dec.weight.data
        norms = w.norm(dim=0, keepdim=True).clamp_min(eps)
        self.W_dec.weight.data = w / norms


# ---------------------------------------------------------------------------
# UniversalSAE
# ---------------------------------------------------------------------------


CLSPoolMode = Literal["none", "mean", "first"]


class UniversalSAE(nn.Module):
    """
    Per-model linear encoder/decoder pairs sharing a feature index space.

    Cross-model concept alignment emerges from:
      (a) the shared dictionary index k (feature k means the same concept across
          all decoders), enforced by joint training, and
      (b) cross-model reconstruction when token counts match (or when CLS-pooling
          is enabled).

    There is no token reshaping. Each model operates in its own native token grid.

    Args:
        model_dims: model_name -> feature dim D_i
        latent_dim: shared dictionary size m (same for all models)
        diffusion_models: set of model names that are diffusion/flow models
        model_tokens: model_name -> token count N_i (kept for diagnostics; not
            used to reshape activations any more)
        top_k: K for TopK sparsity. Operates along the feature axis.
        cls_pool_mode:
            'none'  -> per-token operation. Cross-model decode is only allowed
                       when source and target have the same N. (Default.)
            'mean'  -> mean-pool tokens to a single (B, D) vector per model
                       before encoding. Use this when token counts differ but
                       you want full cross-model reconstruction (matches the
                       USAE paper's per-sample formulation).
            'first' -> take the first token (CLS) only.
        use_tide: whether to apply a TIDE-style timestep affine on diffusion
            sources/targets. Disabled by default — with a single-timestep
            training run it learns a constant affine and does no useful work.
        tide_modulation_location: 'pre' | 'post' | 'both' (only used if use_tide).
        timestep_dim: TIDE embedding dim.
    """

    def __init__(
        self,
        model_dims: Dict[str, int],
        latent_dim: int,
        diffusion_models: Set[str],
        model_tokens: Optional[Dict[str, int]] = None,
        top_k: Optional[int] = None,
        cls_pool_mode: CLSPoolMode = "none",
        use_tide: bool = False,
        tide_modulation_location: Literal["pre", "post", "both"] = "pre",
        timestep_dim: int = 256,
        # Legacy kwargs accepted for backward compatibility but ignored:
        shared_latent_tokens: Optional[int] = None,
        topk_temperature: float = 0.1,
        use_soft_topk: bool = False,
        interpolation_mode: str = "bilinear",
        token_reshape_mode: str = "none",
        attention_heads: int = 8,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        # Soft-topk was always a hack around hard-topk gradient flow; we just
        # use hard TopK now. If the caller asked for soft, warn once.
        if use_soft_topk:
            import warnings
            warnings.warn(
                "use_soft_topk is ignored in the stripped-down UniversalSAE; "
                "hard TopK is used (matches Gao et al. and the USAE paper).",
                stacklevel=2,
            )
        if token_reshape_mode not in ("none",) and shared_latent_tokens is not None:
            import warnings
            warnings.warn(
                f"token_reshape_mode={token_reshape_mode!r} and "
                f"shared_latent_tokens={shared_latent_tokens} are ignored — "
                "this UniversalSAE does not pool tokens. Set "
                "cls_pool_mode='mean' if you need to align different token counts.",
                stacklevel=2,
            )

        self.model_names = list(model_dims.keys())
        self.model_dims = dict(model_dims)
        self.model_tokens = dict(model_tokens or {})
        self.diffusion_models = set(diffusion_models)
        self.latent_dim = latent_dim
        self.top_k = top_k
        self.cls_pool_mode = cls_pool_mode
        self.use_tide = bool(use_tide)
        self.tide_modulation_location = tide_modulation_location

        # Per-model SAEs (linear encoder + linear decoder + shared b_pre)
        self.saes = nn.ModuleDict({
            name: PerModelTopKSAE(in_dim=dim, latent_dim=latent_dim)
            for name, dim in model_dims.items()
        })

        # Optional diffusion timestep adapters
        self.pre = nn.ModuleDict()
        self.post = nn.ModuleDict()
        for name, dim in model_dims.items():
            if self.use_tide and name in self.diffusion_models:
                self.pre[name] = TIDETemporalModulation(dim, tdim=timestep_dim)
                self.post[name] = TIDETemporalModulation(dim, tdim=timestep_dim)
            else:
                self.pre[name] = nn.Identity()
                self.post[name] = nn.Identity()

    # -----------------------------------------------------------------------
    # TopK sparsity
    # -----------------------------------------------------------------------

    def apply_topk(self, z_pre: torch.Tensor) -> torch.Tensor:
        """
        Hard TopK on the last dim. Operates on the *signed* pre-activation,
        following Gao et al. — no ReLU before TopK.

        This is plain hard TopK, NOT a straight-through estimator (the docstring
        and the comment below used to claim otherwise — REPAIR_PLAN.md §2). `mask`
        is built from torch.zeros_like and carries no grad, so d(out)/d(z_pre) is
        exactly `mask`: gradient reaches the selected indices only, and is exactly
        zero everywhere else. A straight-through estimator would deliberately pass
        gradient to the UNSELECTED entries too; this does not. That is precisely
        why `pre_topk_align_weight` exists — the pre-TopK alignment term is the
        only thing giving unselected features any learning signal.
        """
        if self.top_k is None:
            return z_pre
        K = min(self.top_k, z_pre.shape[-1])

        # Top-K on signed pre-activations
        _, idx = torch.topk(z_pre, k=K, dim=-1)
        mask = torch.zeros_like(z_pre)
        mask.scatter_(-1, idx, 1.0)
        # forward = z_pre * mask; backward grad reaches z_pre at the selected
        # positions ONLY (mask has no grad). Not straight-through -- see docstring.
        return z_pre * mask

    # -----------------------------------------------------------------------
    # CLS-pool helper (only used if cls_pool_mode != "none")
    # -----------------------------------------------------------------------

    def _maybe_cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D). Returns (B, 1, D) if pooling is enabled, else x unchanged.
        """
        if self.cls_pool_mode == "none":
            return x
        if x.dim() != 3:
            raise ValueError(
                f"_maybe_cls_pool expected (B, N, D), got {tuple(x.shape)}"
            )
        if self.cls_pool_mode == "mean":
            return x.mean(dim=1, keepdim=True)
        if self.cls_pool_mode == "first":
            return x[:, :1]
        raise ValueError(f"Unknown cls_pool_mode={self.cls_pool_mode!r}")

    # -----------------------------------------------------------------------
    # Encode / decode
    # -----------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        source: str,
        sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N_src, D_src)
        Returns:
            z_pre: (B, N_eff, latent_dim)
            z:     (B, N_eff, latent_dim)  — TopK applied along the last dim
        where N_eff = N_src if cls_pool_mode='none' else 1.
        """
        if x.dim() != 3:
            raise ValueError(
                f"encode expects (B, N, D), got {tuple(x.shape)} for source={source!r}"
            )

        # Optional diffusion pre-modulation
        if (
            self.use_tide
            and source in self.diffusion_models
            and self.tide_modulation_location in {"pre", "both"}
        ):
            if sigma is None:
                raise ValueError(f"sigma required for diffusion source {source!r}")
            x = self.pre[source](x, sigma)

        # Optional CLS-pool (only when token counts differ between models)
        x = self._maybe_cls_pool(x)

        # Linear encode (per-token if not pooled, per-image if pooled)
        z_pre = self.saes[source].encode_pre(x)
        z = self.apply_topk(z_pre)
        return z_pre, z

    def decode(
        self,
        z: torch.Tensor,
        target: str,
        sigma: Optional[torch.Tensor] = None,
        source: Optional[str] = None,  # accepted for backward compat; unused
    ) -> torch.Tensor:
        """
        z: (B, N_eff, latent_dim)
        Returns:
            x_hat: (B, N_eff, D_target) if cls_pool_mode='none'
                   (B, 1, D_target) if pooling is enabled
        Caller is responsible for ensuring N_eff matches the target's expected
        token count when cross-decoding.
        """
        del source

        x_hat = self.saes[target].decode(z)

        if (
            self.use_tide
            and target in self.diffusion_models
            and self.tide_modulation_location in {"post", "both"}
        ):
            if sigma is None:
                raise ValueError(f"sigma required for diffusion target {target!r}")
            x_hat = self.post[target](x_hat, sigma)

        return x_hat

    def forward(
        self,
        x: torch.Tensor,
        source: str,
        target: str,
        sigma_src: Optional[torch.Tensor] = None,
        sigma_tgt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_pre, z = self.encode(x, source, sigma_src)
        x_hat = self.decode(z, target, sigma_tgt)
        return z_pre, z, x_hat

    @torch.no_grad()
    def normalize_decoder_dictionaries_(self) -> None:
        for sae in self.saes.values():
            sae.normalize_decoder_columns_()

    # -----------------------------------------------------------------------
    # Helper for train.py: can we compute a cross-model reconstruction loss?
    # -----------------------------------------------------------------------

    def can_cross_reconstruct(self, source: str, target: str) -> bool:
        """
        True when source's z can be decoded into target's space. With CLS-pooling
        enabled, always True. Without pooling, requires matching token counts.
        """
        if source == target:
            return True
        if self.cls_pool_mode != "none":
            return True
        n_src = self.model_tokens.get(source)
        n_tgt = self.model_tokens.get(target)
        return (n_src is not None) and (n_tgt is not None) and (n_src == n_tgt)
