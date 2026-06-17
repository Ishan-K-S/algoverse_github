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
    Map layer/timestep index from source layout to target layout if counts differ.
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

    If timestep_idx is None, samples a random timestep per batch so the SAE
    sees the full denoising trajectory. Returns the activation slice and
    the corresponding sigma value at that timestep, which is forwarded to
    the encoder as conditioning when use_tide=true.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected diffusion activations shaped (B, T, N, D), got {tuple(x.shape)}")

    _, total_steps, _, _ = x.shape
    if timestep_idx is None:
        timestep_idx = random.randrange(total_steps)
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
    source_timestep_idx: Optional[int] = None,
    source_total_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[int]]:
    """
    Normalize target activation layouts into one comparison slice.

    Diffusion targets reuse the source's timestep so that cross-reconstruction
    compares activations at matching noise levels. If the source was also
    diffusion and chose timestep t, this picks t (mapped proportionally
    if T differs between source and target). If the source was vision (no
    timestep_idx available), the target samples a random timestep.

    Layer alignment uses the same layer index when possible, with proportional
    fallback if source and target expose different numbers of layers.
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

        total_steps = x.shape[1]
        if source_timestep_idx is None:
            timestep_idx = random.randrange(total_steps)
        else:
            src_steps = source_total_steps if source_total_steps is not None else total_steps
            timestep_idx = _map_timestep_idx(source_timestep_idx, src_steps, total_steps)
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

def _pool_target_for_loss(model, x_target_t):
    """Apply the same pooling the encoder applies, so target shape matches x_hat."""
    mode = getattr(model, "cls_pool_mode", "none")
    if mode == "none":
        return x_target_t
    if mode == "mean":
        return x_target_t.mean(dim=1, keepdim=True)
    if mode == "first":
        return x_target_t[:, :1]
    raise ValueError(f"Unknown cls_pool_mode={mode!r}")


def _reset_adam_state_slice(optimizer, param, rows=None, cols=None) -> None:
    """
    Zero the Adam moment estimates (exp_avg, exp_avg_sq) for specific rows/cols
    of a parameter tensor.

    This is run after re-initializing a dead feature so the optimizer doesn't
    immediately re-freeze it with stale, near-zero momentum from when it was
    dead. `rows` indexes dim 0, `cols` indexes dim 1 — pass whichever axis the
    feature lives on (W_enc.weight: feature = row; W_dec.weight: feature =
    column). The per-tensor `step` count is left alone (it can't be reset per
    row); with exp_avg/exp_avg_sq zeroed the first revived update is a normal
    ~lr-scaled step, not an explosion.
    """
    state = optimizer.state.get(param, None)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        buf = state.get(key, None)
        if buf is None:
            continue
        if rows is not None:
            buf[rows] = 0.0
        if cols is not None:
            buf[:, cols] = 0.0


@torch.no_grad()
def resample_dead_features(
    model: nn.Module,
    optimizer,
    acts: Dict[str, torch.Tensor],
    meta,
    diffusion_models: Set[str],
    spatial_aligner=None,
    dead_threshold: float = 1e-3,
    enc_scale_factor: float = 0.2,
    max_per_event: int = 0,
    eps: float = 1e-8,
) -> int:
    """
    Dead-feature resampling for the shared TopK dictionary, following
    Bricken et al. 2023 ("Towards Monosemanticity", Neuron Resampling).

    A feature index is "dead" when its per-model usage EMA (tracked in the
    training loop as model._usage_ema_<name>) is below `dead_threshold` for
    EVERY model — i.e. it sits in the "used by none" partition. Because the
    dictionary index space is shared across models, each dead feature is revived
    JOINTLY: for every model we re-point that same index at one of the
    activations that model currently reconstructs worst.

    Per model, for each revived feature index k:
      - decoder column W_dec[:, k]  <- normalized (x - b_pre) of a sampled
                                       high-loss token (a direction the current
                                       dictionary is failing to cover)
      - encoder row    W_enc[k, :]  <- the same direction, scaled to
                                       enc_scale_factor * mean encoder-row norm
                                       of the alive features (so it fires on
                                       that token without dominating)
      - Adam moment state for those rows/cols is zeroed.

    High-loss tokens are sampled with probability proportional to squared
    self-reconstruction error. Decoder columns are unit-normalized by the
    training loop's normalize_decoder_dictionaries_() on the next step, which
    is consistent with the normalized init here.

    Returns the number of features resampled (0 if none were dead, or if the
    usage EMAs haven't been populated for every model yet).
    """
    # --- 1. Identify features dead across ALL models from the usage EMAs ---
    ema_attrs = {
        name.removeprefix("_usage_ema_"): getattr(model, name)
        for name in dir(model) if name.startswith("_usage_ema_")
    }
    # Need a warmed EMA for every model we have activations for, else too early.
    if any(name not in ema_attrs for name in acts.keys()):
        return 0

    model_names = sorted(acts.keys())
    used_stack = torch.stack([(ema_attrs[n] > dead_threshold) for n in model_names])  # (M, K)
    dead_mask = ~used_stack.any(dim=0)   # (K,) True where dead in every model
    alive_mask = ~dead_mask
    dead_idx = dead_mask.nonzero(as_tuple=False).flatten()
    if dead_idx.numel() == 0:
        return 0

    # Optionally cap revivals per event: pick a random subset so different dead
    # features get chances across successive events (lottery-ticket spirit).
    if max_per_event and dead_idx.numel() > max_per_event:
        perm = torch.randperm(dead_idx.numel(), device=dead_idx.device)[:max_per_event]
        dead_idx = dead_idx[perm]
    n_dead = int(dead_idx.numel())

    # --- 2. Revive those indices in every model, each toward its own residual ---
    for model_name in model_names:
        sae = model.saes[model_name]
        device = sae.W_enc.weight.device

        # Recover an aligned (B, N, D) activation slice for this model.
        x = acts[model_name].to(device)
        is_diff = model_name in diffusion_models
        if is_diff:
            ts_bt = _get_sigmas_bt(meta, model_name, x.shape[0], device)
            x_slice, t_slice, _, _, _, _ = _extract_source_slice(
                x, is_diffusion=True, timestep_values_bt=ts_bt,
            )
        else:
            x_slice, t_slice, _, _, _, _ = _extract_source_slice(x, is_diffusion=False)
        if spatial_aligner is not None:
            x_slice = spatial_aligner.align(x_slice, source=model_name)

        # Self-reconstruct to score per-token reconstruction error.
        _z_pre, z = model.encode(x_slice, source=model_name, sigma=t_slice)
        x_hat = model.decode(z, target=model_name, sigma=t_slice)
        x_cmp = _pool_target_for_loss(model, x_slice)  # match x_hat shape if pooling

        x_cmp_flat = x_cmp.reshape(-1, x_cmp.shape[-1])   # (T, D)
        x_hat_flat = x_hat.reshape(-1, x_hat.shape[-1])   # (T, D)
        recon_err = ((x_hat_flat - x_cmp_flat) ** 2).sum(dim=-1)  # (T,)

        # Sample high-loss tokens with prob proportional to err^2.
        probs = recon_err.double() ** 2
        total = probs.sum()
        if not torch.isfinite(total) or total <= 0:
            probs = torch.ones_like(probs)
        chosen = torch.multinomial(probs, num_samples=n_dead, replacement=True)
        chosen_acts = x_cmp_flat[chosen]  # (n_dead, D)

        # Decoder dictionary directions: point at what we currently miss.
        directions = chosen_acts - sae.b_pre  # (n_dead, D)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(eps)

        # Encoder scale: enc_scale_factor * mean encoder-row norm of alive feats.
        enc_norms = sae.W_enc.weight.data.norm(dim=-1)  # (K,)
        ref_norm = enc_norms[alive_mask].mean() if alive_mask.any() else enc_norms.mean()
        enc_scale = enc_scale_factor * ref_norm

        # Write the revived weights (decoder columns, encoder rows).
        sae.W_dec.weight.data[:, dead_idx] = directions.t().to(sae.W_dec.weight.dtype)
        sae.W_enc.weight.data[dead_idx, :] = (directions * enc_scale).to(sae.W_enc.weight.dtype)

        # Wipe optimizer momentum for just these rows/cols so they actually train.
        _reset_adam_state_slice(optimizer, sae.W_enc.weight, rows=dead_idx)
        _reset_adam_state_slice(optimizer, sae.W_dec.weight, cols=dead_idx)

    return n_dead


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
    latent_align_weight: float = 3.0,  # cosine loss between z_src and z_tgt in latent space
    latent_align_mode: str = "per_token",  # "bag" | "per_token" | "both"
    pre_topk_align_weight: float = 1.0,  # weight on dense pre-activation cosine alignment.
                                          # Adds gradient signal to non-selected feature
                                          # positions, letting the encoder reorganize
                                          # which indices fire (the post-TopK loss can't).
                                          # 0.0 disables.
    curriculum_epochs: int = 5,
    curriculum_self_only: bool = True,
    balanced_sources: bool = False,
    warmup_steps: int = 1000,
    ema_decay: float = 0.98,
    save_model_path: str = SAVE_MODEL_PATH,
    spatial_aligner=None,  # optional SpatialAligner; pools tokens to a common grid
    resample_dead: bool = False,        # enable dead-feature resampling
    resample_interval: int = 500,       # global steps between resampling events
    resample_dead_threshold: float = 1e-3,  # usage-EMA threshold for "dead"
    resample_enc_scale: float = 0.2,    # revived enc-row norm = this * mean alive norm
    resample_max_per_event: int = 0,    # cap revivals per event (0 = all dead)
    resample_start_step: int = 0,       # don't resample before this global step
    resample_end_step: int = 0,         # stop resampling at/after this step (0 = no limit)
):
    """
    One epoch of Universal SAE training.

    For diffusion sources, a random timestep is sampled per batch; diffusion
    targets reuse the source's sampled timestep (mapped proportionally if T
    differs) so cross-reconstruction compares matching noise regimes. Sigma
    values are passed to model.encode and model.decode; whether the encoder
    actually conditions on them is controlled by use_tide in the model config.

    Reconstruction loss is the sum of self-recon (source==target) and
    cross-recon (source!=target) MSE, weighted by self_weight and
    cross_weight respectively. An optional latent-alignment cosine loss
    pulls per-image latent codes toward agreement across models.
    """
    del model_tokens
    model.train()

    # Critical: can_cross_reconstruct() checks model.model_tokens to decide if
    # cross-model reconstruction is structurally possible. With spatial alignment
    # active, every model's tokens are pooled down to target_n_tokens before
    # encoding, so the effective token count is the same for all models. Patch
    # model_tokens here so can_cross_reconstruct() sees the post-alignment counts
    # and doesn't silently skip the cross-model loss.
    if spatial_aligner is not None:
        target_n = spatial_aligner.target_n_tokens
        for name in list(model.model_tokens.keys()):
            if name in spatial_aligner.native_grid_sizes:
                model.model_tokens[name] = target_n
        if epoch == 0:
            print(f"[train] spatial_aligner active: patched model_tokens -> "
                  f"{{k: {target_n} for all aligned models}}")
            print(f"[train] can_cross_reconstruct will now return True for all aligned pairs.")

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
    resampled_since_log = 0

    for batch_idx, ((acts, meta), _y) in enumerate(tqdm(dataloader, desc="train", dynamic_ncols=True)):
        global_step_actual = global_step + batch_idx
        if 0 < global_step_actual + 1 <= warmup_steps:
            warmup_scale = (global_step_actual + 1) / warmup_steps
            _set_optimizer_warmup_lr(optimizer, warmup_scale)

        # Dead-feature resampling (Bricken et al. 2023). Runs at intervals using
        # the current batch's activations to score reconstruction error. Revived
        # features participate in this batch's forward/backward immediately.
        if (
            resample_dead
            and resample_interval > 0
            and global_step_actual > 0
            and global_step_actual % resample_interval == 0
            and global_step_actual >= resample_start_step
            and (resample_end_step <= 0 or global_step_actual < resample_end_step)
        ):
            n_resampled = resample_dead_features(
                model,
                optimizer,
                acts,
                meta,
                diffusion_models,
                spatial_aligner=spatial_aligner,
                dead_threshold=resample_dead_threshold,
                enc_scale_factor=resample_enc_scale,
                max_per_event=resample_max_per_event,
            )
            if n_resampled > 0:
                resampled_since_log += n_resampled
                print(f"[resample] step {global_step_actual}: revived {n_resampled} dead features")

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

            x_src, t_src_values, source_layer_idx, source_timestep_idx, source_total_layers, source_total_steps = (
                _extract_source_slice(
                    x_src_full,
                    is_diffusion=True,
                    timestep_values_bt=src_timesteps_bt,
                )
            )
            if spatial_aligner is not None:
                x_src = spatial_aligner.align(x_src, source=source)
            z_pre, z = model.encode(x_src, source=source, sigma=t_src_values)
        else:
            x_src = acts[source].to(device)
            batch_size = x_src.shape[0]
            x_src, _unused_t, source_layer_idx, source_timestep_idx, source_total_layers, source_total_steps = (
                _extract_source_slice(
                    x_src,
                    is_diffusion=False,
                )
            )
            if spatial_aligner is not None:
                x_src = spatial_aligner.align(x_src, source=source)
            z_pre, z = model.encode(x_src, source=source, sigma=None)

        for target, x_target in acts.items():
            if in_curriculum and curriculum_self_only and target != source:
                continue
 
            # Skip cross-model targets we structurally can't reconstruct
            if not model.can_cross_reconstruct(source, target):
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
                    source_timestep_idx=source_timestep_idx,
                    source_total_steps=source_total_steps,
                )
                if spatial_aligner is not None:
                    x_target_t = spatial_aligner.align(x_target_t, source=target)
                x_hat = model.decode(z, target=target, sigma=t_tgt_values)
            else:
                x_target_t, _unused_tgt_t, _target_layer_idx, _unused_tgt_idx = _extract_target_slice(
                    x_target,
                    is_diffusion=False,
                    timestep_values_bt=None,
                    source_layer_idx=source_layer_idx,
                    source_total_layers=source_total_layers,
                )
                if spatial_aligner is not None:
                    x_target_t = spatial_aligner.align(x_target_t, source=target)
                x_hat = model.decode(z, target=target, sigma=None)
 
            # Pool the target if encoder pools the source
            x_target_t = _pool_target_for_loss(model, x_target_t)
 
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

        # Latent alignment loss: for each cross-model pair, encode the target too
        # and penalise the cosine distance between the two mean-pooled latent codes.
        # This is a direct gradient signal forcing both encoders to fire the SAME
        # features for the same image — cross-reconstruction alone is insufficient
        # because a model can reconstruct the other's space using disjoint features.
        latent_align_loss = torch.tensor(0.0, device=device)
        n_align_pairs = 0
        if latent_align_weight > 0 and not in_curriculum:
            for tgt_name, x_tgt_acts in acts.items():
                if tgt_name == source:
                    continue
                if not model.can_cross_reconstruct(source, tgt_name):
                    continue
                x_tgt_raw = x_tgt_acts.to(device)
                if tgt_name in diffusion_models:
                    tgt_ts_bt = _get_sigmas_bt(meta, tgt_name, batch_size, x_tgt_raw.device)
                    x_tgt_sl, t_tgt_sl, _, _ = _extract_target_slice(
                        x_tgt_raw, is_diffusion=True,
                        timestep_values_bt=tgt_ts_bt,
                        source_layer_idx=source_layer_idx,
                        source_total_layers=source_total_layers,
                        source_timestep_idx=source_timestep_idx,
                        source_total_steps=source_total_steps,
                    )
                    if spatial_aligner is not None:
                        x_tgt_sl = spatial_aligner.align(x_tgt_sl, source=tgt_name)
                    z_pre_tgt, z_tgt = model.encode(x_tgt_sl, source=tgt_name, sigma=t_tgt_sl)
                else:
                    x_tgt_sl, _, _, _ = _extract_target_slice(
                        x_tgt_raw, is_diffusion=False,
                        timestep_values_bt=None,
                        source_layer_idx=source_layer_idx,
                        source_total_layers=source_total_layers,
                    )
                    if spatial_aligner is not None:
                        x_tgt_sl = spatial_aligner.align(x_tgt_sl, source=tgt_name)
                    z_pre_tgt, z_tgt = model.encode(x_tgt_sl, source=tgt_name, sigma=None)

                # Alignment loss: pull z_src and z_tgt toward agreement.
                #
                # mode='bag' (legacy): mean-pool tokens, cosine across image-level vectors.
                #   Only constrains the per-image FEATURE-USAGE HISTOGRAM; the encoder can
                #   satisfy this by picking different features per token while keeping the
                #   mean similar. Diagnostic symptom: bag-of-features cosine ~1.0,
                #   per-feature co-fire ~0.
                #
                # mode='per_token': cosine at each spatial position. Forces the two encoders
                #   to agree on which features fire AT THIS PATCH. Requires spatial
                #   alignment to be enabled so token positions correspond between models.
                #
                # mode='both': average of bag and per_token.
                #
                # Pre-TopK alignment: an additional cosine term on the DENSE pre-activations
                # (z_pre, before TopK). The post-TopK gradient is killed at the ~99.5% of
                # positions that aren't selected, so it can only adjust magnitudes on
                # already-overlapping indices and cannot reorganize WHICH indices fire.
                # The pre-TopK term sees every dimension and can pull both encoders'
                # readout directions into agreement, eventually shifting which indices
                # cross the TopK threshold. Controlled by pre_topk_align_weight.
                if latent_align_mode == "per_token":
                    pair_align_post = (1.0 - F.cosine_similarity(z, z_tgt, dim=-1)).mean()
                elif latent_align_mode == "both":
                    z_src_mean = z.mean(dim=1)
                    z_tgt_mean = z_tgt.mean(dim=1)
                    bag = (1.0 - F.cosine_similarity(z_src_mean, z_tgt_mean, dim=-1)).mean()
                    per_tok = (1.0 - F.cosine_similarity(z, z_tgt, dim=-1)).mean()
                    pair_align_post = 0.5 * (bag + per_tok)
                else:  # "bag" (legacy)
                    z_src_mean = z.mean(dim=1)
                    z_tgt_mean = z_tgt.mean(dim=1)
                    pair_align_post = (1.0 - F.cosine_similarity(z_src_mean, z_tgt_mean, dim=-1)).mean()

                if pre_topk_align_weight > 0:
                    pair_align_pre = (1.0 - F.cosine_similarity(z_pre, z_pre_tgt, dim=-1)).mean()
                    pair_align = pair_align_post + pre_topk_align_weight * pair_align_pre
                    per_target_losses[f"latent_align_pre_{source}_vs_{tgt_name}"] = pair_align_pre.item()
                else:
                    pair_align = pair_align_post

                latent_align_loss = latent_align_loss + pair_align
                n_align_pairs += 1
                per_target_losses[f"latent_align_{source}_vs_{tgt_name}"] = pair_align_post.item()

        sae_loss = None
        if reconstruction_weight_total > 0:
            sae_loss = reconstruction_loss / reconstruction_weight_total
            loss = loss + sae_loss

        if n_align_pairs > 0:
            loss = loss + latent_align_weight * (latent_align_loss / n_align_pairs)

        loss.backward()
        optimizer.step()
        if hasattr(model, "normalize_decoder_dictionaries_"):
            model.normalize_decoder_dictionaries_()
        last_loss = loss.detach()

        loss_value = loss.item()
        ema_loss = loss_value if ema_loss is None else (ema_decay * ema_loss + (1.0 - ema_decay) * loss_value)
        model._train_loss_ema = ema_loss

        # ---- Per-feature usage EMA tracking (partition diagnostic) ----
        # For each model, track an EMA of per-feature firing rate over training
        # batches. At log time we derive: how many features fire for *every*
        # model vs only one. A healthy shared dictionary has most features
        # firing for both models; a partitioned dictionary has features
        # firing exclusively for one model.
        with torch.no_grad():
            usage_now = (z != 0).float().mean(dim=tuple(range(z.dim() - 1)))  # (K,)
            attr = f"_usage_ema_{source}"
            prev = getattr(model, attr, None)
            if prev is None or prev.shape != usage_now.shape:
                setattr(model, attr, usage_now.clone())
            else:
                # EMA: usage_decay ~ 0.95 -> effective window ~20 batches per source
                getattr(model, attr).mul_(0.95).add_(0.05 * usage_now)

        if use_wandb and WANDB_AVAILABLE and (batch_idx % log_every == 0):
            log_dict = {
                "train/total_loss": loss_value,
                "train/total_loss_ema": ema_loss,
                "train/sae_loss": sae_loss.item() if sae_loss is not None else 0.0,
                "train/latent_align_loss": (latent_align_loss / max(n_align_pairs, 1)).item(),
                "train/source_model": source,
                "train/epoch": epoch,
                "train/global_step": global_step_actual,
                "train/in_curriculum": float(in_curriculum),
                "train/resampled_features": resampled_since_log,
            }
            resampled_since_log = 0
            if source_timestep_idx is not None:
                log_dict["train/source_timestep_idx"] = int(source_timestep_idx)
            for pair, value in per_target_losses.items():
                safe_key = pair.replace("->", "_to_")
                log_dict[f"train/loss_{safe_key}"] = value

            with torch.no_grad():
                log_dict["train/latent_sparsity"] = (z == 0).float().mean().item()
                attn_loss = attention_component_loss(model)
                if attn_loss is not None:
                    log_dict["train/attention_component_loss"] = attn_loss.item()
                log_dict.update(_attention_module_wandb_logs(model))

                # ---- Partition diagnostic logging ----
                # Need EMAs from BOTH models to compute. If only one has been
                # seen so far (e.g. very early in training), skip.
                ema_attrs = {
                    name.removeprefix("_usage_ema_"): getattr(model, name)
                    for name in dir(model) if name.startswith("_usage_ema_")
                }
                if len(ema_attrs) >= 2:
                    # Threshold: feature is "used" if EMA firing rate > 1e-3
                    # (i.e. fires on at least 0.1% of tokens averaged over recent batches).
                    used_per_model = {n: (e > 1e-3) for n, e in ema_attrs.items()}
                    model_names = sorted(used_per_model.keys())
                    used_stack = torch.stack([used_per_model[n] for n in model_names])  # (M, K)

                    used_by_all = used_stack.all(dim=0).sum().item()
                    used_by_none = (~used_stack.any(dim=0)).sum().item()
                    K_total = used_stack.shape[1]

                    log_dict["partition/used_by_all_models"] = used_by_all
                    log_dict["partition/used_by_none"] = used_by_none
                    log_dict["partition/frac_shared"] = used_by_all / K_total
                    for n in model_names:
                        only_this = (used_per_model[n] & ~torch.stack(
                            [used_per_model[m] for m in model_names if m != n]
                        ).any(dim=0)).sum().item()
                        log_dict[f"partition/used_by_{n}_only"] = only_this

                    # Partition score: max exclusive count / shared count.
                    # >1.0 means more features are exclusive to some model than shared.
                    max_excl = max(
                        log_dict[f"partition/used_by_{n}_only"] for n in model_names
                    )
                    log_dict["partition/score"] = max_excl / max(used_by_all, 1)

                    # Per-feature firing-rate cosine across models — continuous
                    # version of frac_shared. 1 = identical usage profile,
                    # 0 = orthogonal (disjoint feature sets).
                    if len(model_names) == 2:
                        e0 = ema_attrs[model_names[0]]
                        e1 = ema_attrs[model_names[1]]
                        cos = torch.nn.functional.cosine_similarity(
                            e0.unsqueeze(0), e1.unsqueeze(0)
                        ).item()
                        log_dict[f"partition/usage_cosine_{model_names[0]}_vs_{model_names[1]}"] = cos

            wandb.log(log_dict, step=global_step_actual)

    os.makedirs(
        os.path.dirname(save_model_path) or ".",
        exist_ok=True,
    )

    torch.save(model, save_model_path)

    print("\n[save] Full model saved:")
    print(f"       {save_model_path}\n")

    return last_loss
