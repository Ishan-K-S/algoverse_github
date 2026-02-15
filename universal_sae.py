# universal_sae.py
import math
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        half = self.dim // 2
        scale = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -scale)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TemporalAffine(nn.Module):
    """
    Sigma-conditioned per-channel affine:
      x' = x * (1 + scale(sigma)) + shift(sigma)

    Works for x shaped (B,D) or (B,N,D).
    """
    def __init__(self, dim: int, tdim: int = 256):
        super().__init__()
        self.t_embed = TimestepEmbedding(tdim)
        self.mlp = nn.Sequential(
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, 2 * dim),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (B,)
        h = self.t_embed(sigma)
        scale, shift = self.mlp(h).chunk(2, dim=-1)  # (B,D), (B,D)
        if x.dim() == 3:
            return x * (1 + scale[:, None, :]) + shift[:, None, :]
        return x * (1 + scale) + shift


def interpolate_tokens(
    x: torch.Tensor,
    src_tokens: int,
    tgt_tokens: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Spatially interpolate tokens from src_tokens to tgt_tokens.

    Assumes tokens are arranged in a square grid (sqrt(N) x sqrt(N)).
    
    Args:
        x: (B, N_src, D) tensor
        src_tokens: number of source tokens (must be a perfect square)
        tgt_tokens: number of target tokens (must be a perfect square)
        mode: interpolation mode ("bilinear", "nearest", "bicubic")
    
    Returns:
        (B, N_tgt, D) tensor
    """
    if src_tokens == tgt_tokens:
        return x
    
    src_h = int(math.sqrt(src_tokens))
    tgt_h = int(math.sqrt(tgt_tokens))
    
    assert src_h * src_h == src_tokens, f"src_tokens={src_tokens} is not a perfect square"
    assert tgt_h * tgt_h == tgt_tokens, f"tgt_tokens={tgt_tokens} is not a perfect square"
    
    # (B, N, D) -> (B, D, H, W)
    x = rearrange(x, "b (h w) d -> b d h w", h=src_h, w=src_h)
    
    # Interpolate spatially
    x = F.interpolate(x, size=(tgt_h, tgt_h), mode=mode, align_corners=False if mode != "nearest" else None)
    
    # (B, D, H, W) -> (B, N, D)
    x = rearrange(x, "b d h w -> b (h w) d")
    
    return x


class PerModelSAE(nn.Module):
    """
    Just the linear encoder/decoder pair for one source.
    """
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.enc = nn.Linear(in_dim, latent_dim)
        self.dec = nn.Linear(latent_dim, in_dim)
        nn.init.zeros_(self.enc.bias)
        nn.init.zeros_(self.dec.bias)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)


class UniversalSAE(nn.Module):
    """
    Separate encoder/decoder per model -> shared latent space.
    
    Now uses a FIXED-SIZE shared latent space where all models are pooled/unpooled
    to a canonical token count. This ensures:
    - Consistent sparsity: top-k operates on the same grid for all models
    - Shared semantics: latent features have the same spatial meaning across models

    Architecture:
      Input (variable tokens) → Pool to canonical size → Encode → Sparse latent (fixed size) 
      → Decode → Unpool to target size → Output (variable tokens)

    Diffusion-only:
      x --pre_affine(sigma)--> Pool --> SAE encoder --> z (sparse, fixed size)
      z --> SAE decoder --> Unpool --> x_hat --post_affine(sigma)--> x_hat_final
    """
    def __init__(
        self,
        model_dims: Dict[str, int],
        latent_dim: int,
        diffusion_models: Set[str],
        model_tokens: Optional[Dict[str, int]] = None,
        shared_latent_tokens: int = 256,
        timestep_dim: int = 256,
        top_k: Optional[int] = None,
        topk_temperature: float = 0.1,
        use_soft_topk: bool = True,
        interpolation_mode: str = "bilinear",
    ):
        """
        Args:
            model_dims: Dict mapping model name -> feature dimension (D)
            latent_dim: Shared latent dimension for all models
            diffusion_models: Set of model names that are diffusion/flow models
            model_tokens: Dict mapping model name -> number of tokens (N).
                          Required for pooling/unpooling operations.
            shared_latent_tokens: Canonical number of tokens in the shared latent space.
                                  All models are pooled to this size before encoding.
            timestep_dim: Dimension for timestep embeddings (diffusion only)
            top_k: If set, apply top-k sparsity to latent z
            topk_temperature: Temperature for soft top-k
            use_soft_topk: Use soft (differentiable) top-k vs hard top-k
            interpolation_mode: Mode for spatial interpolation ("bilinear", "nearest", "bicubic")
        """
        super().__init__()
        self.model_names = list(model_dims.keys())
        self.diffusion_models = set(diffusion_models)
        self.model_dims = model_dims
        self.model_tokens = model_tokens or {}
        self.shared_latent_tokens = shared_latent_tokens
        self.interpolation_mode = interpolation_mode

        self.latent_dim = latent_dim
        self.top_k = top_k
        self.topk_temperature = topk_temperature
        self.use_soft_topk = use_soft_topk

        # Validate that all models have token counts specified
        for name in self.model_names:
            if name not in self.model_tokens:
                raise ValueError(
                    f"Model '{name}' missing from model_tokens. "
                    f"All models must have token counts specified for fixed-size shared latent."
                )

        self.saes = nn.ModuleDict({k: PerModelSAE(v, latent_dim) for k, v in model_dims.items()})

        # diffusion-only adapters
        self.pre = nn.ModuleDict()
        self.post = nn.ModuleDict()
        for name, dim in model_dims.items():
            if name in self.diffusion_models:
                self.pre[name] = TemporalAffine(dim, tdim=timestep_dim)
                self.post[name] = TemporalAffine(dim, tdim=timestep_dim)
            else:
                self.pre[name] = nn.Identity()
                self.post[name] = nn.Identity()

    def apply_topk(self, z: torch.Tensor) -> torch.Tensor:
        if self.top_k is None:
            return z

        if self.use_soft_topk:
            # soft mask around kth value threshold
            k_index = z.shape[-1] - self.top_k
            thr = torch.kthvalue(z, k_index + 1, dim=-1, keepdim=True).values
            mask = torch.sigmoid((z - thr) / self.topk_temperature)
            return z * mask

        # hard topk
        mask = torch.zeros_like(z)
        _, idx = torch.topk(z, self.top_k, dim=-1)
        mask.scatter_(-1, idx, 1.0)
        return z * mask + (z * mask - z).detach()

    def encode(
        self,
        x: torch.Tensor,
        source: str,
        sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode activations from a source model into the shared latent space.
        
        Now pools input to canonical token count BEFORE encoding to ensure
        consistent sparse latent representation across all models.
        
        Args:
            x: (B, N_src, D) activations from the source model
            source: Name of the source model
            sigma: (B,) noise levels (required for diffusion models)
        
        Returns:
            z_pre: Pre-activation latent (before ReLU and top-k) - (B, shared_latent_tokens, latent_dim)
            z: Final sparse latent representation - (B, shared_latent_tokens, latent_dim)
        """
        # diffusion-only: adapt before pooling
        if source in self.diffusion_models:
            assert sigma is not None, f"sigma required for diffusion source '{source}'"
            x = self.pre[source](x, sigma)

        # Pool to canonical token count
        src_tokens = self.model_tokens[source]
        if src_tokens != self.shared_latent_tokens:
            x = interpolate_tokens(
                x,
                src_tokens=src_tokens,
                tgt_tokens=self.shared_latent_tokens,
                mode=self.interpolation_mode,
            )
        # x is now (B, shared_latent_tokens, D_src)

        z_pre = self.saes[source].encode_pre(x)  # (B, shared_latent_tokens, latent_dim)
        z = F.relu(z_pre)
        z = self.apply_topk(z)  # Top-k now operates on consistent grid
        return z_pre, z

    def decode(
        self,
        z: torch.Tensor,
        target: str,
        sigma: Optional[torch.Tensor] = None,
        source: Optional[str] = None,  # No longer needed but kept for API compatibility
    ) -> torch.Tensor:
        """
        Decode from shared latent space to a target model's activation space.
        
        Now unpools from canonical token count AFTER decoding.
        
        Args:
            z: (B, shared_latent_tokens, latent_dim) latent representation
            target: Name of the target model
            sigma: (B,) noise levels (required for diffusion targets)
            source: DEPRECATED - no longer used since z is always canonical size
        
        Returns:
            x_hat: (B, N_tgt, D_tgt) reconstructed activations
        """
        # z is always (B, shared_latent_tokens, latent_dim)
        x_hat = self.saes[target].decode(z)  # (B, shared_latent_tokens, D_tgt)
        
        # Unpool to target token count
        tgt_tokens = self.model_tokens[target]
        if tgt_tokens != self.shared_latent_tokens:
            x_hat = interpolate_tokens(
                x_hat,
                src_tokens=self.shared_latent_tokens,
                tgt_tokens=tgt_tokens,
                mode=self.interpolation_mode,
            )
        # x_hat is now (B, N_tgt, D_tgt)
        
        # diffusion-only: adapt after unpooling
        if target in self.diffusion_models:
            assert sigma is not None, f"sigma required for diffusion target '{target}'"
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
        """
        Full encode-decode pass from source to target model space.
        
        Args:
            x: (B, N_src, D_src) source activations
            source: Source model name
            target: Target model name
            sigma_src: Source sigma (for diffusion sources)
            sigma_tgt: Target sigma (for diffusion targets)
        
        Returns:
            z_pre: Pre-activation latent
            z: Sparse latent
            x_hat: Reconstructed activations in target space
        """
        z_pre, z = self.encode(x, source, sigma_src)
        x_hat = self.decode(z, target, sigma_tgt, source=source)
        return z_pre, z, x_hat
