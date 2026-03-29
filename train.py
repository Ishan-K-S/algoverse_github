import random
from typing import Optional, Dict, Set

import torch
import torch.nn as nn
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
    """Ensure sigmas is shaped (B, T)."""
    # Squeeze out any size-1 dims in the middle
    sigmas = sigmas.squeeze()  # removes ALL size-1 dims -> (T,) or (B, T)
    if sigmas.dim() == 1:
        sigmas = sigmas.unsqueeze(0).expand(batch_size, -1)  # (B, T)
    assert sigmas.shape == (batch_size, sigmas.shape[1]), \
        f"_ensure_bt: unexpected shape {tuple(sigmas.shape)} for batch_size={batch_size}"
    return sigmas


def _get_sigmas_bt(meta, model_name, batch_size, device):
    """
    Extract sigmas for a given model from metadata.

    Looks for per-model sigmas first, then falls back to global sigmas.
    Returns (B, T) on device, or None if not found.
    """
    sig = None

    if isinstance(meta, dict) and "sigmas_by_model" in meta:
        if model_name in meta["sigmas_by_model"]:
            sig = meta["sigmas_by_model"][model_name]

    if sig is None and isinstance(meta, dict) and "sigmas" in meta:
        sig = meta["sigmas"]

    if sig is None:
        return None

    sig = sig.to(device)
    sig = _ensure_bt(sig, batch_size)
    return sig


def _map_timestep_idx(t_src, t_src_len, t_tgt_len):
    """
    Map timestep index from source schedule to target schedule if lengths differ.
    Uses proportional mapping [0..Tsrc-1] -> [0..Ttgt-1].
    """
    if t_tgt_len <= 1:
        return 0
    if t_src_len <= 1:
        return min(t_src, t_tgt_len - 1)
    return int(round(t_src * (t_tgt_len - 1) / (t_src_len - 1)))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_universal_sae(
    model: nn.Module,
    dataloader,
    optimizer,
    diffusion_models: Set[str],
    model_tokens: Optional[Dict[str, int]] = None,
    device: str = "cuda",
    epoch: int = 0,
    use_wandb: bool = False,
    log_every: int = 50,
    curriculum_epochs: int = 5,
    curriculum_self_only: bool = True,
    balanced_sources: bool = True,
):
    """
    Train the Universal SAE with optional self/cross reconstruction.

    Each batch can be balanced across sources, and early epochs can
    run curriculum warm start with self-reconstruction only.

    - Encode source activations to shared latent z.
    - Decode z to every target model for loss computation.
    - For vision targets: compare full (B, N, D) output.
    - For diffusion targets: compare one timestep (B, N, D) with sigma.
    - Per source->target pair: aggregate loss, normalize by term count,
      and scale by running EMA.

    Args:
        model: UniversalSAE instance
        dataloader: DataLoader yielding ((acts_dict, metadata), labels)
        optimizer: Optimizer for model parameters
        diffusion_models: Set of model names that are diffusion/flow models
        model_tokens: Dict mapping model name -> number of tokens (informational)
        device: Device to train on
        epoch: Current epoch index (used for wandb step calculation)
        use_wandb: Whether to log metrics to wandb
        log_every: Log to wandb every N steps
    """
    model.train()

    diffusion_models = set(diffusion_models)
    model_tokens = model_tokens or {}

    if pair_ema is None:
        pair_ema = {}

    cross_weight = 2.0
    self_weight = 1.0

    warmup_steps = 1000
    base_lr = optimizer.param_groups[0]["lr"]

    global_step = epoch * len(dataloader)
    last_loss = torch.tensor(0.0)

    for batch_idx, ((acts, meta), _y) in enumerate(tqdm(dataloader, desc="train", dynamic_ncols=True)):
        optimizer.zero_grad()

        source_list = sorted(list(acts.keys()))
        if balanced_sources:
            active_sources = source_list
        else:
            active_sources = [source_list[batch_idx % len(source_list)]]

        allow_cross = (epoch >= curriculum_epochs) if curriculum_self_only else True

        pair_loss_tensors: Dict[str, torch.Tensor] = {}
        pair_term_count: Dict[str, int] = {}
        per_target_losses: Dict[str, float] = {}

        for source in active_sources:
            if source not in acts:
                continue

            if source in diffusion_models:
                # ----------------------------------------------------------------
                # Diffusion source: acts[source] is (B, T, N, D)
                # ----------------------------------------------------------------
                x_src = acts[source].to(device)

                if x_src.dim() != 4:
                    raise ValueError(
                        f"Expected diffusion source '{source}' to be (B, T, N, D), "
                        f"got {tuple(x_src.shape)}"
                    )

                B, Tsrc, N, D = x_src.shape

                src_sigmas_bt = _get_sigmas_bt(meta, source, B, device)
                if src_sigmas_bt is None:
                    raise KeyError(
                        f"Source '{source}' is diffusion but sigmas not found in metadata. "
                        f"Expected meta['sigmas_by_model'][{source!r}] or meta['sigmas']."
                    )

                for t_src in range(Tsrc):
                    sigma_src = src_sigmas_bt[:, t_src]
                    x_t = x_src[:, t_src]
                    _z_pre, z = model.encode(x_t, source=source, sigma=sigma_src)

                    for target, x_target in acts.items():
                        if source != target and not allow_cross:
                            continue

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
                                    f"Target '{target}' is diffusion but sigmas not found in metadata."
                                )

                            t_tgt = _map_timestep_idx(t_src, Tsrc, Ttgt)
                            sigma_tgt = tgt_sigmas_bt[:, t_tgt]

                            x_hat = model.decode(z, target=target, sigma=sigma_tgt)
                            x_target_t = x_target[:, t_tgt]

                            target_loss = mse_flat(x_hat, x_target_t)
                        else:
                            if x_target.dim() != 3:
                                raise ValueError(
                                    f"Expected vision target '{target}' to be (B, N, D), "
                                    f"got {tuple(x_target.shape)}"
                                )

                            x_hat = model.decode(z, target=target, sigma=None)
                            target_loss = mse_flat(x_hat, x_target)

                        w = cross_weight if source != target else self_weight
                        weighted_target_loss = w * target_loss

                        key = f"{source}->{target}"
                        pair_loss_tensors[key] = pair_loss_tensors.get(key, torch.tensor(0.0, device=device)) + weighted_target_loss
                        pair_term_count[key] = pair_term_count.get(key, 0) + 1

            else:
                # ----------------------------------------------------------------
                # Vision source: acts[source] is (B, N, D)
                # ----------------------------------------------------------------
                x_src = acts[source].to(device)

                if x_src.dim() != 3:
                    raise ValueError(
                        f"Expected vision source '{source}' to be (B, N, D), "
                        f"got {tuple(x_src.shape)}"
                    )

                B = x_src.shape[0]

                _z_pre, z = model.encode(x_src, source=source, sigma=None)

                for target, x_target in acts.items():
                    if source != target and not allow_cross:
                        continue

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
                                f"Target '{target}' is diffusion but sigmas not found in metadata."
                            )

                        sigma_tgt = tgt_sigmas_bt[:, t_tgt]
                        x_hat = model.decode(z, target=target, sigma=sigma_tgt)
                        x_target_t = x_target[:, t_tgt]

                        target_loss = mse_flat(x_hat, x_target_t)
                    else:
                        if x_target.dim() != 3:
                            raise ValueError(
                                f"Expected vision target '{target}' to be (B, N, D), "
                                f"got {tuple(x_target.shape)}"
                            )

                        x_hat = model.decode(z, target=target, sigma=None)
                        target_loss = mse_flat(x_hat, x_target)

                    w = cross_weight if source != target else self_weight
                    weighted_target_loss = w * target_loss

                    key = f"{source}->{target}"
                    pair_loss_tensors[key] = pair_loss_tensors.get(key, torch.tensor(0.0, device=device)) + weighted_target_loss
                    pair_term_count[key] = pair_term_count.get(key, 0) + 1

        loss = torch.tensor(0.0, device=device)
        per_target_losses = {}
        for key, pt_loss in pair_loss_tensors.items():
            count = pair_term_count.get(key, 1)
            normalized_pt_loss = pt_loss / float(count)

            loss = loss + normalized_pt_loss

            per_target_losses[key] = normalized_pt_loss.item()

        last_loss = loss

        loss.backward()
        optimizer.step()
        last_loss = loss

        global_step_actual = global_step + batch_idx

        if global_step_actual < warmup_steps:
            warmup_lr = base_lr * (global_step_actual / warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # ----------------------------------------------------------------
        # wandb logging
        # ----------------------------------------------------------------
        if use_wandb and WANDB_AVAILABLE and (batch_idx % log_every == 0):
            log_dict = {
                "train/total_loss": loss.item(),
                "train/source_model": source,
                "train/epoch": epoch,
                "train/global_step": global_step_actual,
            }
            for pair, val in per_target_losses.items():
                safe_key = pair.replace("->", "_to_")
                log_dict[f"train/loss_{safe_key}"] = val

            try:
                with torch.no_grad():
                    sparsity = (z == 0).float().mean().item()
                log_dict["train/latent_sparsity"] = sparsity
            except NameError:
                pass

            wandb.log(log_dict, step=global_step_actual)

    return last_loss, pair_ema
