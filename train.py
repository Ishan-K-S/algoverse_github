# train.py (patched with token interpolation + optional global attention)
import math
import random
from typing import Optional, Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

# Global Attention Layers

class GlobalAttentionLayer(nn.Module):
    """
    A single global self-attention layer over the token (N) dimension of the
    shared latent z (B, N, D).

    Uses multi-head attention with pre-LayerNorm and a residual connection,
    plus an optional feed-forward sub-layer (also pre-norm + residual).

    Args:
        dim:         Latent feature dimension D.
        num_heads:   Number of attention heads. Must divide dim evenly.
        ff_mult:     Feed-forward hidden size multiplier (set to 0 to disable FF).
        dropout:     Dropout probability applied inside attention and FF.
        bias:        Whether to use bias in projection layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,   # expects (B, N, D)
        )
        self.attn_drop = nn.Dropout(dropout)

        # Feed-forward sub-layer (optional)
        if ff_mult > 0:
            ff_hidden = int(dim * ff_mult)
            self.norm_ff = nn.LayerNorm(dim)
            self.ff = nn.Sequential(
                nn.Linear(dim, ff_hidden, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_hidden, dim, bias=bias),
                nn.Dropout(dropout),
            )
        else:
            self.norm_ff = None
            self.ff = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        # --- Self-attention with pre-norm + residual ---
        residual = x
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.attn_drop(attn_out)

        # --- Feed-forward with pre-norm + residual (optional) ---
        if self.ff is not None:
            x = x + self.ff(self.norm_ff(x))

        return x


class GlobalAttentionStack(nn.Module):
    """
    A configurable stack of GlobalAttentionLayers designed to be inserted
    between the encoder and decoder of a Universal SAE.

    The stack operates on the shared latent z (B, N, D).

    Args:
        dim:       Latent feature dimension D.
        depth:     Number of stacked attention layers.
        num_heads: Number of attention heads per layer.
        ff_mult:   Feed-forward multiplier (0 disables the FF sub-layer).
        dropout:   Dropout probability.
        bias:      Whether to use bias in projection layers.

    Example usage::

        attn_stack = GlobalAttentionStack(dim=512, depth=2, num_heads=8)
        attn_stack = attn_stack.to(device)

        # Pass to train_universal_sae:
        train_universal_sae(
            model, dataloader, optimizer,
            diffusion_models={"flux"},
            attention_stack=attn_stack,
        )
    """

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            GlobalAttentionLayer(
                dim=dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
                bias=bias,
            )
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, N, D) shared latent from the SAE encoder.
        Returns:
            (B, N, D) refined latent.
        """
        for layer in self.layers:
            z = layer(z)
        return self.final_norm(z)


# Loss helpers

def mse_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss between two tensors, flattening the token dimension.

    Args:
        a, b: (B, N, D) tensors (must have same shape)

    Returns:
        Scalar MSE loss
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    return F.mse_loss(
        rearrange(a, "b n d -> (b n) d"),
        rearrange(b, "b n d -> (b n) d"),
    )


def _ensure_bt(sigmas: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Ensure sigmas is shaped (B, T). Accepts (T,) or (1, T) or (B, T)."""
    if sigmas.dim() == 1:
        sigmas = sigmas.unsqueeze(0)  # (1, T)
    if sigmas.shape[0] == 1 and batch_size > 1:
        sigmas = sigmas.expand(batch_size, -1)
    return sigmas


def _get_sigmas_bt(
    meta: dict,
    model_name: str,
    batch_size: int,
    device: str,
) -> Optional[torch.Tensor]:
    """
    Extract sigmas for a given model from metadata.

    Your data.py returns metadata like:
      meta["sigmas"] : (1, T) or (B, T)  (picked from first diffusion source)
      meta["sigmas_by_model"][name] : (1, T) or (B, T)

    This helper pulls the best available sigmas for a given model and returns (B, T) on device.
    """
    sig = None

    # Preferred: per-model sigmas
    if isinstance(meta, dict) and "sigmas_by_model" in meta:
        if model_name in meta["sigmas_by_model"]:
            sig = meta["sigmas_by_model"][model_name]

    # Fallback: global sigmas
    if sig is None and isinstance(meta, dict) and "sigmas" in meta:
        sig = meta["sigmas"]

    if sig is None:
        return None

    sig = sig.to(device)
    sig = _ensure_bt(sig, batch_size)
    return sig


def _map_timestep_idx(t_src: int, t_src_len: int, t_tgt_len: int) -> int:
    """
    Map timestep index from source schedule to target schedule if lengths differ.
    Uses proportional mapping [0..Tsrc-1] -> [0..Ttgt-1].
    """
    if t_tgt_len <= 1:
        return 0
    if t_src_len <= 1:
        return min(t_src, t_tgt_len - 1)
    return int(round(t_src * (t_tgt_len - 1) / (t_src_len - 1)))


# Training loop

def train_universal_sae(
    model,
    dataloader,
    optimizer,
    diffusion_models: Set[str],
    model_tokens: Optional[Dict[str, int]] = None,
    attention_stack: Optional[GlobalAttentionStack] = None,
    device: str = "cuda",
):
    """
    Train the Universal SAE with cross-model reconstruction.

    Training policy:
      - Pick a random source each batch.
      - Encode source -> shared latent z.
      - Optionally refine z with a GlobalAttentionStack (over the N dimension).
      - Decode z into EVERY target model (with token interpolation as needed).
      - Vision targets compare against (B, N, D).
      - Diffusion targets compare against ONE selected timestep (B, N, D)
        using that target's sigma.

    Args:
        model:            UniversalSAE instance.
        dataloader:       DataLoader yielding ((acts_dict, metadata), labels).
        optimizer:        Optimizer for model parameters (should include
                          attention_stack.parameters() if attention_stack is used).
        diffusion_models: Set of model names that are diffusion/flow models.
        model_tokens:     Dict mapping model name -> number of tokens.
                          Used for interpolation when cross-reconstructing.
        attention_stack:  Optional GlobalAttentionStack applied to z after
                          encoding and before decoding.  When provided:
                            - It must already be on `device`.
                            - Its parameters should be added to the optimizer.
                          Pass None (default) to disable.
        device:           Device to train on.

    Example — enabling attention::

        attn_stack = GlobalAttentionStack(dim=512, depth=2, num_heads=8).to(device)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(attn_stack.parameters()),
            lr=1e-4,
        )
        train_universal_sae(
            model, dataloader, optimizer,
            diffusion_models={"flux"},
            attention_stack=attn_stack,
        )
    """
    model.train()
    if attention_stack is not None:
        attention_stack.train()

    diffusion_models = set(diffusion_models)
    model_tokens = model_tokens or {}

    for (acts, meta), _y in tqdm(dataloader, desc="train", dynamic_ncols=True):
        source = random.choice(list(acts.keys()))

        optimizer.zero_grad()
        loss = 0.0

        if source in diffusion_models:
            # Diffusion source: acts[source] is (B, T, N, D)
            x_src = acts[source].to(device)
            B, Tsrc, N, D = x_src.shape

            src_sigmas_bt = _get_sigmas_bt(meta, source, B, device)
            if src_sigmas_bt is None:
                raise KeyError(
                    f"Source '{source}' is diffusion but sigmas not found in metadata. "
                    f"Expected meta['sigmas_by_model'][{source!r}] or meta['sigmas']."
                )

            for t_src in range(Tsrc):
                sigma_src = src_sigmas_bt[:, t_src]   # (B,)
                x_t = x_src[:, t_src]                 # (B, N, D)

                _z_pre, z = model.encode(x_t, source=source, sigma=sigma_src)

                # --- Optional global attention over token dimension ---
                if attention_stack is not None:
                    z = attention_stack(z)

                for target, x_target in acts.items():
                    x_target = x_target.to(device)

                    if target in diffusion_models:
                        if x_target.dim() != 4:
                            raise ValueError(
                                f"Expected diffusion target '{target}' to be (B, T, N, D), "
                                f"got {tuple(x_target.shape)}"
                            )
                        Ttgt = x_target.shape[1]
                        tgt_sigmas_bt = _get_sigmas_bt(meta, target, B, device)
                        if tgt_sigmas_bt is None:
                            raise KeyError(
                                f"Target '{target}' is diffusion but sigmas not found in metadata. "
                                f"Expected meta['sigmas_by_model'][{target!r}] or meta['sigmas']."
                            )

                        t_tgt = _map_timestep_idx(t_src, Tsrc, Ttgt)
                        sigma_tgt = tgt_sigmas_bt[:, t_tgt]  # (B,)

                        x_hat = model.decode(z, target=target, sigma=sigma_tgt, source=source)
                        x_target_t = x_target[:, t_tgt]      # (B, N_tgt, D_tgt)
                        loss = loss + mse_flat(x_hat, x_target_t)

                    else:
                        if x_target.dim() != 3:
                            raise ValueError(
                                f"Expected vision target '{target}' to be (B, N, D), "
                                f"got {tuple(x_target.shape)}"
                            )
                        x_hat = model.decode(z, target=target, sigma=None, source=source)
                        loss = loss + mse_flat(x_hat, x_target)

        else:
            # Vision source: acts[source] is (B, N, D)
            x_src = acts[source].to(device)
            if x_src.dim() != 3:
                raise ValueError(
                    f"Expected vision source '{source}' to be (B, N, D), "
                    f"got {tuple(x_src.shape)}"
                )
            B, N, D = x_src.shape

            _z_pre, z = model.encode(x_src, source=source, sigma=None)

            # --- Optional global attention over token dimension ---
            if attention_stack is not None:
                z = attention_stack(z)

            for target, x_target in acts.items():
                x_target = x_target.to(device)

                if target in diffusion_models:
                    if x_target.dim() != 4:
                        raise ValueError(
                            f"Expected diffusion target '{target}' to be (B, T, N, D), "
                            f"got {tuple(x_target.shape)}"
                        )
                    Ttgt = x_target.shape[1]
                    t_tgt = random.randrange(Ttgt)

                    tgt_sigmas_bt = _get_sigmas_bt(meta, target, B, device)
                    if tgt_sigmas_bt is None:
                        raise KeyError(
                            f"Target '{target}' is diffusion but sigmas not found in metadata. "
                            f"Expected meta['sigmas_by_model'][{target!r}] or meta['sigmas']."
                        )

                    sigma_tgt = tgt_sigmas_bt[:, t_tgt]  # (B,)
                    x_hat = model.decode(z, target=target, sigma=sigma_tgt, source=source)
                    x_target_t = x_target[:, t_tgt]       # (B, N_tgt, D_tgt)
                    loss = loss + mse_flat(x_hat, x_target_t)

                else:
                    if x_target.dim() != 3:
                        raise ValueError(
                            f"Expected vision target '{target}' to be (B, N, D), "
                            f"got {tuple(x_target.shape)}"
                        )
                    x_hat = model.decode(z, target=target, sigma=None, source=source)
                    loss = loss + mse_flat(x_hat, x_target)

        loss.backward()
        optimizer.step()
