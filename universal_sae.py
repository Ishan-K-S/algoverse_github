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
    
    Now with token interpolation support for cross-model translation
    when models have different token counts.

    Diffusion-only:
      x --pre_affine(sigma)--> SAE encoder --> z
      z --> SAE decoder --> x_hat --post_affine(sigma)--> x_hat_final
    """
    def __init__(
        self,
        model_dims: Dict[str, int],
        latent_dim: int,
        diffusion_models: Set[str],
        model_tokens: Optional[Dict[str, int]] = None,
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
                          Required for cross-model token interpolation.
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
        self.interpolation_mode = interpolation_mode

        self.latent_dim = latent_dim
        self.top_k = top_k
        self.topk_temperature = topk_temperature
        self.use_soft_topk = use_soft_topk

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

    def get_num_tokens(self, model_name: str) -> Optional[int]:
        """Get the number of tokens for a model, if known."""
        return self.model_tokens.get(model_name, None)

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
        
        Args:
            x: (B, N, D) activations from the source model
            source: Name of the source model
            sigma: (B,) noise levels (required for diffusion models)
        
        Returns:
            z_pre: Pre-activation latent (before ReLU and top-k)
            z: Final sparse latent representation
        """
        # diffusion-only: adapt before encoder
        if source in self.diffusion_models:
            assert sigma is not None, f"sigma required for diffusion source '{source}'"
            x = self.pre[source](x, sigma)

        z_pre = self.saes[source].encode_pre(x)
        z = F.relu(z_pre)
        z = self.apply_topk(z)
        return z_pre, z

    def decode(
        self,
        z: torch.Tensor,
        target: str,
        sigma: Optional[torch.Tensor] = None,
        source: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Decode from shared latent space to a target model's activation space.
        
        Handles token interpolation when source and target have different token counts.
        
        Args:
            z: (B, N_src, latent_dim) latent representation
            target: Name of the target model
            sigma: (B,) noise levels (required for diffusion targets)
            source: Name of the source model (needed for token interpolation)
        
        Returns:
            x_hat: (B, N_tgt, D_tgt) reconstructed activations
        """
        # Interpolate tokens if source and target have different token counts
        if source is not None and source in self.model_tokens and target in self.model_tokens:
            src_tokens = self.model_tokens[source]
            tgt_tokens = self.model_tokens[target]
            if src_tokens != tgt_tokens:
                z = interpolate_tokens(z, src_tokens, tgt_tokens, mode=self.interpolation_mode)
        
        x_hat = self.saes[target].decode(z)
        
        # diffusion-only: adapt after decoder
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
