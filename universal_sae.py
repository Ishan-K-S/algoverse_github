import math
from typing import Dict, Literal, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Type alias for the token reshape strategy
TokenReshapeMode = Literal["interpolation", "attention"]


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
        # sigma may arrive as (B,), (B,1), (B,1,1) etc. - flatten to (B,)
        B = x.shape[0]
        sigma = sigma.flatten(start_dim=1)[:, 0].float() if sigma.dim() > 1 else sigma.float()
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

    # Validate actual tensor shape vs declared src_tokens
    actual_tokens = x.shape[1]
    if actual_tokens != src_tokens:
        raise ValueError(
            f"interpolate_tokens: declared src_tokens={src_tokens} but tensor has "
            f"{actual_tokens} tokens (shape {tuple(x.shape)}). "
            f"Check that model_tokens in config matches the cached activation shape, "
            f"or that shared_latent_tokens is set correctly."
        )

    src_h = int(math.sqrt(src_tokens))
    tgt_h = int(math.sqrt(tgt_tokens))

    if src_h * src_h != src_tokens:
        raise ValueError(
            f"interpolate_tokens: src_tokens={src_tokens} is not a perfect square. "
            f"Cannot reshape to a 2-D grid for interpolation."
        )
    if tgt_h * tgt_h != tgt_tokens:
        raise ValueError(
            f"interpolate_tokens: tgt_tokens={tgt_tokens} is not a perfect square. "
            f"Cannot reshape to a 2-D grid for interpolation."
        )

    # (B, N, D) -> (B, D, H, W)
    x = rearrange(x, "b (h w) d -> b d h w", h=src_h, w=src_h)

    # Interpolate spatially
    x = F.interpolate(x, size=(tgt_h, tgt_h), mode=mode, align_corners=False if mode != "nearest" else None)

    # (B, D, H, W) -> (B, N, D)
    x = rearrange(x, "b d h w -> b (h w) d")

    return x


class GlobalAttentionReshape(nn.Module):
    """
    Learned token reshaping via stacked cross-attention.

    To pool (N_src -> N_tgt where N_tgt < N_src):
      - N_tgt learned query tokens attend over N_src input tokens
      - Each output position learns to aggregate relevant input tokens

    To unpool (N_src -> N_tgt where N_tgt > N_src):
      - N_tgt learned query tokens attend over N_src input tokens
      - Each output position learns to interpolate from compressed representation

    Both directions use the same cross-attention mechanism; the distinction is
    just whether N_tgt < or > N_src. The queries are fixed learned embeddings
    (not input-dependent), making this a form of learned spatial resampling.

    Args:
        dim: Feature dimension (D)
        src_tokens: Number of input tokens
        tgt_tokens: Number of output tokens
        num_heads: Number of attention heads (default: 8)
        dropout: Attention dropout (default: 0.0)
    """

    def __init__(
        self,
        dim: int,
        src_tokens: int,
        tgt_tokens: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        self.dim = dim

        # Learned query embeddings - one per output token
        # These are fixed positional queries, not derived from the input
        self.queries = nn.Parameter(torch.randn(1, tgt_tokens, dim) * 0.02)

        self.attn1 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn2 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_src, D)
        Returns:
            (B, N_tgt, D)
        """
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)  # (B, N_tgt, D)

        # First cross-attention pass builds the initial resampled representation.
        out1, _ = self.attn1(query=q, key=x, value=x)  # (B, N_tgt, D)
        if self.src_tokens == self.tgt_tokens:
            out1 = self.norm1(out1 + x)
        else:
            out1 = self.norm1(out1 + q)

        # A second cross-attention layer lets the learned output tokens refine
        # themselves against the same source tokens.
        out2, _ = self.attn2(query=out1, key=x, value=x)  # (B, N_tgt, D)
        out = self.norm2(out2 + out1)

        return out


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
      Input (variable tokens) -> Pool to canonical size -> Encode -> Sparse latent (fixed size)
      -> Decode -> Unpool to target size -> Output (variable tokens)

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
        token_reshape_mode: TokenReshapeMode = "attention",
        attention_heads: int = 8,
        attention_dropout: float = 0.0,
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
            interpolation_mode: Mode for spatial interpolation ("bilinear", "nearest", "bicubic").
                                 Only used when token_reshape_mode="interpolation".
            token_reshape_mode: Strategy for reshaping token counts between model space and
                                 shared latent space.
                                 - "interpolation": bilinear/nearest/bicubic spatial interpolation.
                                   Fast, parameter-free, assumes tokens are on a spatial grid.
                                 - "attention": learned cross-attention pooling/unpooling.
                                   Adds per-model learned query parameters. More expressive
                                   but slower and uses more memory, especially for FLUX (4096 tokens).
            attention_heads: Number of heads for GlobalAttentionReshape (only used when
                             token_reshape_mode="attention").
            attention_dropout: Dropout for GlobalAttentionReshape attention weights.
        """
        super().__init__()
        self.model_names = list(model_dims.keys())
        self.diffusion_models = set(diffusion_models)
        self.model_dims = model_dims
        self.model_tokens = model_tokens or {}
        self.shared_latent_tokens = shared_latent_tokens
        self.interpolation_mode = interpolation_mode
        self.token_reshape_mode = token_reshape_mode

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

        # Token reshape modules (only built when attention mode is selected,
        # since interpolation is parameter-free and applied inline)
        if token_reshape_mode == "attention":
            # Each model needs a pooler (src_tokens -> shared_latent_tokens)
            # and an unpooler (shared_latent_tokens -> src_tokens)
            # We build these only when src_tokens != shared_latent_tokens
            self.token_poolers = nn.ModuleDict()
            self.token_unpoolers = nn.ModuleDict()
            for name, dim in model_dims.items():
                src_tokens = self.model_tokens[name]
                if src_tokens != shared_latent_tokens:
                    self.token_poolers[name] = GlobalAttentionReshape(
                        dim=dim,
                        src_tokens=src_tokens,
                        tgt_tokens=shared_latent_tokens,
                        num_heads=attention_heads,
                        dropout=attention_dropout,
                    )
                    self.token_unpoolers[name] = GlobalAttentionReshape(
                        dim=dim,
                        src_tokens=shared_latent_tokens,
                        tgt_tokens=src_tokens,
                        num_heads=attention_heads,
                        dropout=attention_dropout,
                    )

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

    def _reshape_tokens(
        self,
        x: torch.Tensor,
        model_name: str,
        src_tokens: int,
        tgt_tokens: int,
        direction: Literal["pool", "unpool"],
    ) -> torch.Tensor:
        """
        Reshape token count from src_tokens to tgt_tokens using the configured strategy.

        Args:
            x: (B, src_tokens, D)
            model_name: Model name (used to look up attention modules)
            src_tokens: Number of input tokens
            tgt_tokens: Number of output tokens
            direction: "pool" (downsampling toward shared latent) or
                       "unpool" (upsampling toward model space)
        Returns:
            (B, tgt_tokens, D)
        """
        if src_tokens == tgt_tokens:
            return x

        if self.token_reshape_mode == "interpolation":
            return interpolate_tokens(
                x,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens,
                mode=self.interpolation_mode,
            )

        if self.token_reshape_mode == "attention":
            if direction == "pool":
                return self.token_poolers[model_name](x)
            return self.token_unpoolers[model_name](x)

        raise ValueError(
            f"Unknown token_reshape_mode: {self.token_reshape_mode!r}. "
            "Expected 'interpolation' or 'attention'."
        )

    def apply_topk(self, z: torch.Tensor) -> torch.Tensor:
        if self.top_k is None:
            return z

        if self.use_soft_topk:
            # soft mask around kth value threshold
            # kthvalue requires k in [1, latent_dim]; clamp so top_k <= latent_dim is safe
            k_index = max(0, z.shape[-1] - self.top_k)
            k = max(1, k_index + 1)
            thr = torch.kthvalue(z, k, dim=-1, keepdim=True).values
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

        # Auto-correct model_tokens if the actual tensor disagrees with config.
        # This handles the common case where num_tokens in config.yaml is wrong
        # (e.g. cached with CLS token included but use_class_tokens=false strips it,
        # or the cache was generated with a different patch size).
        actual_tokens = x.shape[1]
        declared_tokens = self.model_tokens[source]
        if actual_tokens != declared_tokens:
            import warnings

            warnings.warn(
                f"[UniversalSAE] model_tokens['{source}'] is {declared_tokens} in config "
                f"but actual tensor has {actual_tokens} tokens. "
                f"Auto-correcting to {actual_tokens}. "
                f"Please update num_tokens in config.yaml to suppress this warning.",
                stacklevel=2,
            )
            self.model_tokens[source] = actual_tokens

        # Pool to canonical token count
        src_tokens = self.model_tokens[source]
        x = self._reshape_tokens(
            x,
            model_name=source,
            src_tokens=src_tokens,
            tgt_tokens=self.shared_latent_tokens,
            direction="pool",
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
        # Note: model_tokens[target] may have been auto-corrected by a prior encode() call.
        tgt_tokens = self.model_tokens[target]
        x_hat = self._reshape_tokens(
            x_hat,
            model_name=target,
            src_tokens=self.shared_latent_tokens,
            tgt_tokens=tgt_tokens,
            direction="unpool",
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
