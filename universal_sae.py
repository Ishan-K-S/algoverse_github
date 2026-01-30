# universal_sae.py
import math
from typing import Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    Diffusion-only:
      x --pre_affine(sigma)--> SAE encoder --> z
      z --> SAE decoder --> x_hat --post_affine(sigma)--> x_hat_final
    """
    def __init__(
        self,
        model_dims: Dict[str, int],
        latent_dim: int,
        diffusion_models: Set[str],
        timestep_dim: int = 256,
        top_k: Optional[int] = None,
        topk_temperature: float = 0.1,
        use_soft_topk: bool = True,
    ):
        super().__init__()
        self.model_names = list(model_dims.keys())
        self.diffusion_models = set(diffusion_models)

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

    def encode(self, x: torch.Tensor, source: str, sigma: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # diffusion-only: adapt before encoder
        if source in self.diffusion_models:
            assert sigma is not None, "sigma required for diffusion source"
            x = self.pre[source](x, sigma)

        z_pre = self.saes[source].encode_pre(x)
        z = F.relu(z_pre)
        z = self.apply_topk(z)
        return z_pre, z

    def decode(self, z: torch.Tensor, target: str, sigma: Optional[torch.Tensor]) -> torch.Tensor:
        x_hat = self.saes[target].decode(z)
        # diffusion-only: adapt after decoder
        if target in self.diffusion_models:
            assert sigma is not None, "sigma required for diffusion target"
            x_hat = self.post[target](x_hat, sigma)
        return x_hat
