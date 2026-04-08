import random
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
import wandb


def mse_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss between two activation tensors of shape (B, N, D).
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return F.mse_loss(
        rearrange(a, "b n d -> (b n) d"),
        rearrange(b, "b n d -> (b n) d"),
    )


def cosine_reconstruction_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Cosine-distance reconstruction loss on flattened token activations.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")

    a_flat = rearrange(a, "b n d -> (b n) d")
    b_flat = rearrange(b, "b n d -> (b n) d")
    return (1.0 - F.cosine_similarity(a_flat, b_flat, dim=-1)).mean()


def decoder_orthogonality_loss(decoder: nn.Linear) -> torch.Tensor:
    """
    Encourage decoder atoms to be less entangled.

    For a decoder mapping latent_dim -> activation_dim, each latent feature is a
    decoder column. We penalize off-diagonal cosine similarities between atoms.
    """
    weight = decoder.weight
    atoms = F.normalize(weight.t(), dim=-1)
    gram = atoms @ atoms.t()
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    off_diag = gram - eye
    return off_diag.pow(2).mean()


def _ensure_bt(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Ensure per-batch timesteps/noise levels are shaped (B, T).
    """
    values = values.squeeze()
    if values.dim() == 1:
        values = values.unsqueeze(0).expand(batch_size, -1)
    if values.dim() != 2 or values.shape[0] != batch_size:
        raise ValueError(
            f"_ensure_bt expected shape (B, T); got {tuple(values.shape)} for B={batch_size}"
        )
    return values


def _get_sigmas_bt(meta, model_name: str, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
    """
    Extract timestep conditioning for a given model from metadata.

    Preferred keys:
      - meta["timesteps_by_model"][model_name]
      - meta["timesteps"]

    Backward-compatible fallbacks:
      - meta["sigmas_by_model"][model_name]
      - meta["sigmas"]
    """
    values = None

    if isinstance(meta, dict):
        if "timesteps_by_model" in meta and model_name in meta["timesteps_by_model"]:
            values = meta["timesteps_by_model"][model_name]
        elif "timesteps" in meta:
            values = meta["timesteps"]
        elif "sigmas_by_model" in meta and model_name in meta["sigmas_by_model"]:
            values = meta["sigmas_by_model"][model_name]
        elif "sigmas" in meta:
            values = meta["sigmas"]

    if values is None:
        return None

    return _ensure_bt(values.to(device), batch_size)


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


def _nearest_timestep_idx(
    src_timestep_value: torch.Tensor,
    tgt_timestep_values: torch.Tensor,
) -> int:
    """
    Align timesteps by actual conditioning value rather than proportional index.

    Uses the batch-mean timestep/noise level as a robust summary when values are
    provided per example.
    """
    src_value = src_timestep_value.float().mean()
    tgt_values = tgt_timestep_values.float().mean(dim=0)
    return int(torch.argmin((tgt_values - src_value).abs()).item())


def _pick_diffusion_slice(
    x: torch.Tensor,
    t_bt: torch.Tensor,
    timestep_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Select one diffusion timestep slice
    """
    if x.dim() != 4:
        raise ValueError(f"Expected diffusion activations shaped (B, T, N, D), got {tuple(x.shape)}")

    _, total_steps, _, _ = x.shape
    if timestep_idx is None:
            #Old code:
        #timestep_idx = random.randrange(total_steps)
            #For now, we're always using the last timestep, as done below:
        timestep_idx = total_steps - 1
    return x[:, timestep_idx], t_bt[:, timestep_idx], timestep_idx


def _pick_temporal_neighbor_idx(timestep_idx: int, total_steps: int) -> Optional[int]:
    """
    Pick a neighboring timestep index for temporal consistency.

    Prefers an adjacent step because temporal consistency objective is
    meant to smooth nearby timesteps along the denoising trajectory.
    """
    if total_steps <= 1:
        return None
    if timestep_idx == 0:
        return 1
    if timestep_idx == total_steps - 1:
        return total_steps - 2
    return timestep_idx - 1 if random.random() < 0.5 else timestep_idx + 1


def temporal_consistency_loss(
    z_curr: torch.Tensor,
    z_neigh: torch.Tensor,
    t_curr: torch.Tensor,
    t_neigh: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage neighboring timesteps to have smoothly varying sparse features.

    This is a local continuity penalty between adjacent or near-adjacent
    diffusion steps. We avoid inverse timestep weighting because that can make
    gradients unstable when the step spacing is very small.
    """
    if z_curr.shape != z_neigh.shape:
        raise ValueError(
            f"Temporal consistency shape mismatch: {tuple(z_curr.shape)} vs {tuple(z_neigh.shape)}"
        )

    del t_curr, t_neigh
    return F.mse_loss(z_curr, z_neigh)


def sparsity_loss(z: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """
    Huber-style sparsity penalty on the latent activations.

    This keeps the explicit sparse objective even when top-k masking is already
    present in the encoder.
    """
    return F.huber_loss(z, torch.zeros_like(z), delta=delta, reduction="mean")


def _sample_layer_index(num_layers: int) -> int:
    if num_layers <= 1:
        return 0
    return random.randrange(num_layers)


def _extract_source_slice(
    x: torch.Tensor,
    is_diffusion: bool,
    timestep_values_bt: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Normalize source activation layouts into a single slice the SAE can encode.

    Returns:
      x_slice: (B, N, D)
      timestep_slice: (B,) or None
      layer_idx: selected layer index or None
      timestep_idx: selected timestep index or None
      total_layers: total number of layers or None
      total_steps: total number of diffusion steps or None

    Supported shapes:
      vision: (B, N, D) or (B, L, N, D)
      diffusion: (B, T, N, D) or (B, L, T, N, D)
    """
    if is_diffusion:
        if x.dim() == 5:
            _, total_layers, _, _, _ = x.shape
            layer_idx = _sample_layer_index(total_layers)
            x = x[:, layer_idx]
        elif x.dim() == 4:
            layer_idx = None
            total_layers = None
        else:
            raise ValueError(
                f"Expected diffusion activations shaped (B, T, N, D) or (B, L, T, N, D), got {tuple(x.shape)}"
            )

        if timestep_values_bt is None:
            raise ValueError("Diffusion source requires timestep values.")
        x_slice, timestep_slice, timestep_idx = _pick_diffusion_slice(x, timestep_values_bt)
        return x_slice, timestep_slice, layer_idx, timestep_idx, total_layers, x.shape[1]

    if x.dim() == 4:
        _, total_layers, _, _ = x.shape
        layer_idx = _sample_layer_index(total_layers)
        x = x[:, layer_idx]
    elif x.dim() == 3:
        layer_idx = None
        total_layers = None
    else:
        raise ValueError(
            f"Expected vision activations shaped (B, N, D) or (B, L, N, D), got {tuple(x.shape)}"
        )

    return x, None, layer_idx, None, total_layers, None


def _extract_target_slice(
    x: torch.Tensor,
    is_diffusion: bool,
    timestep_values_bt: Optional[torch.Tensor],
    source_timestep_values: Optional[torch.Tensor],
    source_layer_idx: Optional[int],
    source_total_layers: Optional[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[int]]:
    """
    Normalize target activation layouts into one comparison slice.

    Layer alignment uses the same layer index when possible, with proportional
    fallback if source and target expose different numbers of layers.
    Diffusion timestep alignment uses nearest actual timestep/noise level.
    """
    layer_idx = None
    if is_diffusion:
        if x.dim() == 5:
            total_layers = x.shape[1]
            if source_layer_idx is None:
                layer_idx = _sample_layer_index(total_layers)
            else:
                src_layers = source_total_layers if source_total_layers is not None else total_layers
                layer_idx = _map_timestep_idx(source_layer_idx, src_layers, total_layers)
            x = x[:, layer_idx]
        elif x.dim() != 4:
            raise ValueError(
                f"Expected diffusion activations shaped (B, T, N, D) or (B, L, T, N, D), got {tuple(x.shape)}"
            )

        if timestep_values_bt is None:
            raise ValueError("Diffusion target requires timestep values.")

        if source_timestep_values is None:
            timestep_idx = random.randrange(x.shape[1])
        else:
            timestep_idx = _nearest_timestep_idx(source_timestep_values, timestep_values_bt)

        return x[:, timestep_idx], timestep_values_bt[:, timestep_idx], layer_idx, timestep_idx

    if x.dim() == 4:
        total_layers = x.shape[1]
        if source_layer_idx is None:
            layer_idx = _sample_layer_index(total_layers)
        else:
            src_layers = source_total_layers if source_total_layers is not None else total_layers
            layer_idx = _map_timestep_idx(source_layer_idx, src_layers, total_layers)
        x = x[:, layer_idx]
    elif x.dim() != 3:
        raise ValueError(
            f"Expected vision activations shaped (B, N, D) or (B, L, N, D), got {tuple(x.shape)}"
        )

    return x, None, layer_idx, None


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
    temporal_consistency_weight: float = 0.1,
    cosine_weight: float = 0.0,
    sparsity_weight: float = 1e-2,
    orthogonality_weight: float = 0,
):
    """
    Train a Universal SAE with Temporal Awareness
    """
    model.train()

    diffusion_models = set(diffusion_models)
    cross_weight = 2.0
    self_weight = 1.0

    warmup_steps = 1000
    base_lr = optimizer.param_groups[0]["lr"]

    global_step = epoch * len(dataloader)
    last_loss = torch.tensor(0.0, device=device)

    for batch_idx, ((acts, meta), _y) in enumerate(tqdm(dataloader, desc="train", dynamic_ncols=True)):
        source = random.choice(list(acts.keys()))
        optimizer.zero_grad()

        loss = torch.tensor(0.0, device=device)
        per_target_losses: Dict[str, float] = {}
        batch_sparsity_loss = None

        if source in diffusion_models:
            x_src_full = acts[source].to(device)
            batch_size = x_src_full.shape[0]
            src_timesteps_bt = _get_sigmas_bt(meta, source, batch_size, x_src_full.device)
            if src_timesteps_bt is None:
                raise KeyError(
                    f"Missing timestep metadata for diffusion source '{source}'. "
                    f"Expected timesteps_by_model/timesteps or sigmas_by_model/sigmas."
                )

            x_src, t_src_values, source_layer_idx, t_src_idx, source_total_layers, source_total_steps = _extract_source_slice(
                x_src_full,
                is_diffusion=True,
                timestep_values_bt=src_timesteps_bt,
            )
            _z_pre, z = model.encode(x_src, source=source, sigma=t_src_values)

            neighbor_idx = _pick_temporal_neighbor_idx(t_src_idx, source_total_steps)
            temporal_loss_value = None
            if neighbor_idx is not None and temporal_consistency_weight > 0:
                if x_src_full.dim() == 5 and source_layer_idx is not None:
                    x_neighbor = x_src_full[:, source_layer_idx, neighbor_idx]
                else:
                    x_neighbor = x_src_full[:, neighbor_idx]
                t_neighbor_values = src_timesteps_bt[:, neighbor_idx]
                _z_pre_neighbor, z_neighbor = model.encode(x_neighbor, source=source, sigma=t_neighbor_values)
                temporal_loss_value = temporal_consistency_loss(
                    z,
                    z_neighbor,
                    t_src_values,
                    t_neighbor_values,
                )
                loss = loss + temporal_consistency_weight * temporal_loss_value
                per_target_losses[f"{source}_temporal"] = temporal_loss_value.item()
        else:
            x_src = acts[source].to(device)
            batch_size = x_src.shape[0]
            x_src, _unused_t, source_layer_idx, t_src_idx, source_total_layers, source_total_steps = _extract_source_slice(
                x_src,
                is_diffusion=False,
            )
            _z_pre, z = model.encode(x_src, source=source, sigma=None)

        if sparsity_weight > 0:
            batch_sparsity_loss = sparsity_loss(z)
            loss = loss + sparsity_weight * batch_sparsity_loss
            per_target_losses[f"{source}_sparsity"] = batch_sparsity_loss.item()

        if orthogonality_weight > 0:
            source_ortho = decoder_orthogonality_loss(model.saes[source].dec)
            loss = loss + orthogonality_weight * source_ortho
            per_target_losses[f"{source}_orthogonality"] = source_ortho.item()

        for target, x_target in acts.items():
            x_target = x_target.to(device)

            if target in diffusion_models:
                tgt_timesteps_bt = _get_sigmas_bt(meta, target, batch_size, x_target.device)
                if tgt_timesteps_bt is None:
                    raise KeyError(
                        f"Missing timestep metadata for diffusion target '{target}'. "
                        f"Expected timesteps_by_model/timesteps or sigmas_by_model/sigmas."
                    )

                x_target_t, t_tgt_values, _target_layer_idx, t_tgt_idx = _extract_target_slice(
                    x_target,
                    is_diffusion=True,
                    timestep_values_bt=tgt_timesteps_bt,
                    source_timestep_values=t_src_values if source in diffusion_models else None,
                    source_layer_idx=source_layer_idx,
                    source_total_layers=source_total_layers,
                )
                x_hat = model.decode(z, target=target, sigma=t_tgt_values)
            else:
                x_target_t, _unused_tgt_t, _target_layer_idx, _unused_tgt_idx = _extract_target_slice(
                    x_target,
                    is_diffusion=False,
                    timestep_values_bt=None,
                    source_timestep_values=None,
                    source_layer_idx=source_layer_idx,
                    source_total_layers=source_total_layers,
                )
                x_hat = model.decode(z, target=target, sigma=None)

            target_mse_loss = mse_flat(x_hat, x_target_t)
            target_cosine_loss = cosine_reconstruction_loss(x_hat, x_target_t) if cosine_weight > 0 else None
            weight = cross_weight if source != target else self_weight
            target_total_loss = target_mse_loss
            if target_cosine_loss is not None:
                target_total_loss = target_total_loss + cosine_weight * target_cosine_loss
                per_target_losses[f"{source}->{target}_cosine"] = target_cosine_loss.item()

            loss = loss + weight * target_total_loss
            per_target_losses[f"{source}->{target}"] = target_mse_loss.item()

        loss.backward()
        optimizer.step()
        if hasattr(model, "normalize_decoder_dictionaries_"):
            model.normalize_decoder_dictionaries_()
        last_loss = loss.detach()

        global_step_actual = global_step + batch_idx
        if global_step_actual < warmup_steps:
            warmup_lr = base_lr * (global_step_actual / warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        if use_wandb and WANDB_AVAILABLE and (batch_idx % log_every == 0):
            log_dict = {
                "train/total_loss": loss.item(),
                "train/source_model": source,
                "train/epoch": epoch,
                "train/global_step": global_step_actual,
            }
            for pair, value in per_target_losses.items():
                safe_key = pair.replace("->", "_to_")
                log_dict[f"train/loss_{safe_key}"] = value

            with torch.no_grad():
                log_dict["train/latent_sparsity"] = (z == 0).float().mean().item()

            wandb.log(log_dict, step=global_step_actual)

    return last_loss
