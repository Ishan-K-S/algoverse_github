import random
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
import os

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

SAVE_MODEL_PATH = "./models/universal_sae_final.pt"


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
    Map layer index from source layout to target layout if layer counts differ.
    Uses proportional mapping [0..Tsrc-1] -> [0..Ttgt-1].
    """
    if t_tgt_len <= 1:
        return 0
    if t_src_len <= 1:
        return min(t_src, t_tgt_len - 1)
    return int(round(t_src * (t_tgt_len - 1) / (t_src_len - 1)))


def _pick_diffusion_slice(
    x: torch.Tensor,
    t_bt: torch.Tensor,
    timestep_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Select one diffusion timestep slice for training.

    By default, train only on the final diffusion timestep.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected diffusion activations shaped (B, T, N, D), got {tuple(x.shape)}")

    _, total_steps, _, _ = x.shape
    if timestep_idx is None:
        timestep_idx = total_steps - 1
    return x[:, timestep_idx], t_bt[:, timestep_idx], timestep_idx


def _attention_to_wandb_image(attn: torch.Tensor, caption: str, max_size: int = 512):
    """
    Convert a cached attention matrix to a compact W&B heatmap image.
    """
    if not WANDB_AVAILABLE:
        return None

    image = attn.detach().float().cpu()
    while image.dim() > 2:
        image = image.mean(dim=0)
    if image.dim() != 2:
        return None

    height, width = image.shape
    scale = min(1.0, max_size / max(height, width))
    if scale < 1.0:
        image = F.interpolate(
            image[None, None],
            size=(max(1, round(height * scale)), max(1, round(width * scale))),
            mode="area",
        )[0, 0]

    image = image - image.min()
    image = image / image.max().clamp_min(1e-8)
    image = (image * 255).to(torch.uint8)
    image = image.unsqueeze(-1).expand(-1, -1, 3).numpy()
    return wandb.Image(image, caption=caption)


def _iter_attention_modules(model: nn.Module):
    candidate_groups = ("token_poolers", "token_unpoolers", "attention_modules", "attn_modules")
    for group_name in candidate_groups:
        module_group = getattr(model, group_name, None)
        if module_group is None:
            continue

        for module_name, module in module_group.items():
            yield group_name, module_name, module


def attention_component_loss(model: nn.Module) -> Optional[torch.Tensor]:
    """
    Diagnostic loss for the attention reshape modules.

    This is logged separately from the SAE reconstruction loss; it is not added
    to the training objective.
    """
    losses = []
    for _group_name, _module_name, module in _iter_attention_modules(model):
        params = [param.float().pow(2).mean() for param in module.parameters() if param.numel() > 0]
        if params:
            losses.append(torch.stack(params).mean())

    if not losses:
        return None
    return torch.stack(losses).mean()


def _attention_module_wandb_logs(model: nn.Module) -> Dict[str, object]:
    """
    Build W&B plots for attention reshape modules.

    If attention modules cache maps as last_attn1/last_attn2, those maps are
    logged as heatmap images. Their learned parameters are also logged as
    histograms so attention modules show up in W&B even when maps are not cached.
    """
    if not WANDB_AVAILABLE:
        return {}

    logs: Dict[str, object] = {}
    for group_name, module_name, module in _iter_attention_modules(model):
        for attn_name in ("last_attn1", "last_attn2", "last_attention", "attention_weights"):
            attn = getattr(module, attn_name, None)
            if attn is None:
                continue
            caption = f"{group_name}/{module_name}/{attn_name}"
            image = _attention_to_wandb_image(attn, caption=caption)
            if image is not None:
                logs[f"train/attention/{caption}"] = image

        for param_name, param in module.named_parameters():
            if param.numel() == 0:
                continue
            key = f"train/attention_params/{group_name}/{module_name}/{param_name}"
            logs[key] = wandb.Histogram(param.detach().float().cpu().flatten().numpy())

    return logs


def _sample_layer_index(num_layers: int) -> int:
    if num_layers <= 1:
        return 0
    return random.randrange(num_layers)


def _set_optimizer_warmup_lr(optimizer, warmup_scale: float) -> None:
    for pg in optimizer.param_groups:
        initial_lr = pg.setdefault("initial_lr", pg["lr"])
        pg["lr"] = initial_lr * warmup_scale


def _pick_source(acts: Dict[str, torch.Tensor], step: int, balanced_sources: bool) -> str:
    source_names = sorted(acts.keys())
    if balanced_sources:
        return source_names[step % len(source_names)]
    return random.choice(source_names)


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
    source_layer_idx: Optional[int],
    source_total_layers: Optional[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[int]]:
    """
    Normalize target activation layouts into one comparison slice.

    Diffusion targets always use the final timestep. Layer alignment uses the
    same layer index when possible, with proportional fallback if source and
    target expose different numbers of layers.
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

        timestep_idx = x.shape[1] - 1
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
    cosine_weight: float = 0.0,
    curriculum_epochs: int = 5,
    curriculum_self_only: bool = True,
    balanced_sources: bool = False,
    warmup_steps: int = 1000,
    ema_decay: float = 0.98,
    save_model_path: str = SAVE_MODEL_PATH,
):
    """
    Train a Universal SAE using only the final diffusion timestep.

    Compared to the pasted version:
      - Diffusion sources and targets use the final timestep only.
      - Temporal consistency loss is removed.
      - Sparsity loss is removed.
      - Decoder orthogonality loss is removed.
      - SAE reconstruction loss and attention component loss are logged
        separately to W&B.
      - W&B logs include attention module heatmaps when cached and parameter
        histograms for attention reshape modules.
    """
    del model_tokens
    model.train()

    if epoch == 0:
        print("\n" + "=" * 60)
        print("ACTIVATION STATISTICS CHECK")
        print("=" * 60)

        for (acts, meta), _y in dataloader:
            del meta
            for source_name, source_acts in acts.items():
                source_acts = source_acts.to(device)
                flat = source_acts.reshape(source_acts.shape[0], -1)

                print(
                    f"{source_name:12s} - "
                    f"min: {flat.min():.4f}, "
                    f"max: {flat.max():.4f}, "
                    f"mean: {flat.mean():.4f}, "
                    f"std: {flat.std():.4f}"
                )

            print("=" * 60 + "\n")
            break

    diffusion_models = set(diffusion_models)
    cross_weight = 2.0
    self_weight = 1.0

    global_step = epoch * len(dataloader)
    last_loss = torch.tensor(0.0, device=device)
    ema_loss = getattr(model, "_train_loss_ema", None)
    in_curriculum = epoch < curriculum_epochs

    for batch_idx, ((acts, meta), _y) in enumerate(tqdm(dataloader, desc="train", dynamic_ncols=True)):
        global_step_actual = global_step + batch_idx
        if 0 < global_step_actual + 1 <= warmup_steps:
            warmup_scale = (global_step_actual + 1) / warmup_steps
            _set_optimizer_warmup_lr(optimizer, warmup_scale)

        source = _pick_source(acts, global_step_actual, balanced_sources)
        optimizer.zero_grad()

        loss = torch.tensor(0.0, device=device)
        reconstruction_loss = torch.tensor(0.0, device=device)
        reconstruction_weight_total = 0.0
        per_target_losses: Dict[str, float] = {}

        if source in diffusion_models:
            x_src_full = acts[source].to(device)
            batch_size = x_src_full.shape[0]
            src_timesteps_bt = _get_sigmas_bt(meta, source, batch_size, x_src_full.device)
            if src_timesteps_bt is None:
                raise KeyError(
                    f"Missing timestep metadata for diffusion source '{source}'. "
                    f"Expected timesteps_by_model/timesteps or sigmas_by_model/sigmas."
                )

            x_src, t_src_values, source_layer_idx, _t_src_idx, source_total_layers, _source_total_steps = (
                _extract_source_slice(
                    x_src_full,
                    is_diffusion=True,
                    timestep_values_bt=src_timesteps_bt,
                )
            )
            _z_pre, z = model.encode(x_src, source=source, sigma=t_src_values)
        else:
            x_src = acts[source].to(device)
            batch_size = x_src.shape[0]
            x_src, _unused_t, source_layer_idx, _t_src_idx, source_total_layers, _source_total_steps = (
                _extract_source_slice(
                    x_src,
                    is_diffusion=False,
                )
            )
            _z_pre, z = model.encode(x_src, source=source, sigma=None)

        for target, x_target in acts.items():
            if in_curriculum and curriculum_self_only and target != source:
                continue

            x_target = x_target.to(device)

            if target in diffusion_models:
                tgt_timesteps_bt = _get_sigmas_bt(meta, target, batch_size, x_target.device)
                if tgt_timesteps_bt is None:
                    raise KeyError(
                        f"Missing timestep metadata for diffusion target '{target}'. "
                        f"Expected timesteps_by_model/timesteps or sigmas_by_model/sigmas."
                    )

                x_target_t, t_tgt_values, _target_layer_idx, _t_tgt_idx = _extract_target_slice(
                    x_target,
                    is_diffusion=True,
                    timestep_values_bt=tgt_timesteps_bt,
                    source_layer_idx=source_layer_idx,
                    source_total_layers=source_total_layers,
                )
                x_hat = model.decode(z, target=target, sigma=t_tgt_values)
            else:
                x_target_t, _unused_tgt_t, _target_layer_idx, _unused_tgt_idx = _extract_target_slice(
                    x_target,
                    is_diffusion=False,
                    timestep_values_bt=None,
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

            reconstruction_loss = reconstruction_loss + weight * target_total_loss
            reconstruction_weight_total += weight
            per_target_losses[f"{source}->{target}"] = target_mse_loss.item()

        sae_loss = None
        if reconstruction_weight_total > 0:
            sae_loss = reconstruction_loss / reconstruction_weight_total
            loss = loss + sae_loss

        loss.backward()
        optimizer.step()
        if hasattr(model, "normalize_decoder_dictionaries_"):
            model.normalize_decoder_dictionaries_()
        last_loss = loss.detach()

        loss_value = loss.item()
        ema_loss = loss_value if ema_loss is None else (ema_decay * ema_loss + (1.0 - ema_decay) * loss_value)
        model._train_loss_ema = ema_loss

        if use_wandb and WANDB_AVAILABLE and (batch_idx % log_every == 0):
            log_dict = {
                "train/total_loss": loss_value,
                "train/total_loss_ema": ema_loss,
                "train/sae_loss": sae_loss.item() if sae_loss is not None else 0.0,
                "train/source_model": source,
                "train/epoch": epoch,
                "train/global_step": global_step_actual,
                "train/in_curriculum": float(in_curriculum),
            }
            for pair, value in per_target_losses.items():
                safe_key = pair.replace("->", "_to_")
                log_dict[f"train/loss_{safe_key}"] = value

            with torch.no_grad():
                log_dict["train/latent_sparsity"] = (z == 0).float().mean().item()
                attn_loss = attention_component_loss(model)
                if attn_loss is not None:
                    log_dict["train/attention_component_loss"] = attn_loss.item()
                log_dict.update(_attention_module_wandb_logs(model))

            wandb.log(log_dict, step=global_step_actual)

    os.makedirs(
        os.path.dirname(save_model_path) or ".",
        exist_ok=True,
    )

    torch.save(model, save_model_path)

    print("\n[save] Full model saved:")
    print(f"       {save_model_path}\n")

    return last_loss
