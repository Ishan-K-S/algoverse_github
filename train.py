# train.py (patched with token interpolation + wandb logging)
import random
from typing import Optional, Dict, Set

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
    """
    Ensure sigmas is shaped (B, T).
    Accepts (T,) or (1, T) or (B, T).
    """
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
    # proportional
    return int(round(t_src * (t_tgt_len - 1) / (t_src_len - 1)))


def train_universal_sae(
    model,
    dataloader,
    optimizer,
    diffusion_models: Set[str],
    model_tokens: Optional[Dict[str, int]] = None,
    device: str = "cuda",
    epoch: int = 0,
    use_wandb: bool = False,
    log_every: int = 50,
):
    """
    Train the Universal SAE with cross-model reconstruction.
    
    Training policy:
      - Pick a random source each batch.
      - Encode source -> shared latent z (always at canonical token count).
      - Decode z into EVERY target model (unpooled to target token count).
      - Vision targets compare against (B, N, D).
      - Diffusion targets compare against ONE selected timestep (B, N, D) using that target's sigma.

    Args:
        model: UniversalSAE instance
        dataloader: DataLoader yielding ((acts_dict, metadata), labels)
        optimizer: Optimizer for model parameters
        diffusion_models: Set of model names that are diffusion/flow models
        model_tokens: Dict mapping model name -> number of tokens (for reference, not used in training)
        device: Device to train on
        epoch: Current epoch index (used for wandb step calculation)
        use_wandb: Whether to log metrics to wandb
        log_every: Log to wandb every N steps
    """
    model.train()
    diffusion_models = set(diffusion_models)
    model_tokens = model_tokens or {}

    global_step = epoch * len(dataloader)

    for batch_idx, ((acts, meta), _y) in enumerate(tqdm(dataloader, desc="train", dynamic_ncols=True)):
        source = random.choice(list(acts.keys()))

        optimizer.zero_grad()
        loss = 0.0
        per_target_losses: Dict[str, float] = {}

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

            # Iterate all timesteps for the source diffusion model
            for t_src in range(Tsrc):
                sigma_src = src_sigmas_bt[:, t_src]  # (B,)
                x_t = x_src[:, t_src]                # (B, N, D)

                _z_pre, z = model.encode(x_t, source=source, sigma=sigma_src)
                # z is now (B, shared_latent_tokens, latent_dim) - canonical size

                # Decode to every target
                for target, x_target in acts.items():
                    x_target = x_target.to(device)

                    if target in diffusion_models:
                        # Diffusion target: x_target is (B, Ttgt, N, D)
                        if x_target.dim() != 4:
                            raise ValueError(
                                f"Expected diffusion target '{target}' to be (B, T, N, D), got {tuple(x_target.shape)}"
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
                        
                        x_hat = model.decode(z, target=target, sigma=sigma_tgt)
                        x_target_t = x_target[:, t_tgt]  # (B, N_tgt, D_tgt)
                        
                        target_loss = mse_flat(x_hat, x_target_t)
                        loss = loss + target_loss
                        per_target_losses[f"{source}->{target}"] = (
                            per_target_losses.get(f"{source}->{target}", 0.0) + target_loss.item()
                        )

                    else:
                        # Vision target: x_target is (B, N, D)
                        if x_target.dim() != 3:
                            raise ValueError(
                                f"Expected vision target '{target}' to be (B, N, D), got {tuple(x_target.shape)}"
                            )
                        
                        x_hat = model.decode(z, target=target, sigma=None)
                        target_loss = mse_flat(x_hat, x_target)
                        loss = loss + target_loss
                        per_target_losses[f"{source}->{target}"] = (
                            per_target_losses.get(f"{source}->{target}", 0.0) + target_loss.item()
                        )

        else:
            # Vision source: acts[source] is (B, N, D)
            x_src = acts[source].to(device)
            if x_src.dim() != 3:
                raise ValueError(
                    f"Expected vision source '{source}' to be (B, N, D), got {tuple(x_src.shape)}"
                )
            B, N, D = x_src.shape

            _z_pre, z = model.encode(x_src, source=source, sigma=None)
            # z is now (B, shared_latent_tokens, latent_dim) - canonical size

            # Decode to every target, INCLUDING diffusion
            for target, x_target in acts.items():
                x_target = x_target.to(device)

                if target in diffusion_models:
                    # Diffusion target: pick a random timestep
                    if x_target.dim() != 4:
                        raise ValueError(
                            f"Expected diffusion target '{target}' to be (B, T, N, D), got {tuple(x_target.shape)}"
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
                    
                    x_hat = model.decode(z, target=target, sigma=sigma_tgt)
                    x_target_t = x_target[:, t_tgt]  # (B, N_tgt, D_tgt)
                    
                    target_loss = mse_flat(x_hat, x_target_t)
                    loss = loss + target_loss
                    per_target_losses[f"{source}->{target}"] = target_loss.item()

                else:
                    # Vision target: (B, N, D)
                    if x_target.dim() != 3:
                        raise ValueError(
                            f"Expected vision target '{target}' to be (B, N, D), got {tuple(x_target.shape)}"
                        )
                    
                    x_hat = model.decode(z, target=target, sigma=None)
                    target_loss = mse_flat(x_hat, x_target)
                    loss = loss + target_loss
                    per_target_losses[f"{source}->{target}"] = target_loss.item()

        loss.backward()
        optimizer.step()

        # --- wandb logging ---
        if use_wandb and WANDB_AVAILABLE and (batch_idx % log_every == 0):
            log_dict = {
                "train/total_loss": loss.item() if hasattr(loss, "item") else float(loss),
                "train/source_model": source,
                "train/epoch": epoch,
                "train/global_step": global_step + batch_idx,
            }

            # Per source->target pair losses
            for pair, val in per_target_losses.items():
                safe_key = pair.replace("->", "_to_")
                log_dict[f"train/loss_{safe_key}"] = val

            # Latent sparsity: fraction of zero activations in z
            with torch.no_grad():
                sparsity = (z == 0).float().mean().item()
            log_dict["train/latent_sparsity"] = sparsity

            wandb.log(log_dict, step=global_step + batch_idx)

    return loss  # return last batch loss for convenience
