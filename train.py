# train.py (patched)
import random
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


def mse_flat(a, b):
    # a,b: (B,N,D)
    return F.mse_loss(
        rearrange(a, "b n d -> (b n) d"),
        rearrange(b, "b n d -> (b n) d"),
    )


def _ensure_bt(sigmas: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Ensure sigmas is shaped (B,T).
    Accepts (T,) or (1,T) or (B,T).
    """
    if sigmas.dim() == 1:
        sigmas = sigmas.unsqueeze(0)  # (1,T)
    if sigmas.shape[0] == 1 and batch_size > 1:
        sigmas = sigmas.expand(batch_size, -1)
    return sigmas


def _get_sigmas_bt(meta: dict, model_name: str, batch_size: int, device: str) -> Optional[torch.Tensor]:
    """
    Your data.py returns metadata like:
      meta["sigmas"] : (1,T) or (B,T)  (picked from first diffusion source)
      meta["sigmas_by_model"][name] : (1,T) or (B,T)
    NOT meta[name]["sigmas"].

    This helper pulls the best available sigmas for a given model and returns (B,T) on device.
    """
    sig = None

    # Preferred: per-model
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


def train_universal_sae(model, dataloader, optimizer, diffusion_models, device="cuda"):
    """
    Training policy:
      - Pick a random source each batch.
      - Encode source -> shared latent z.
      - Decode z into EVERY target model.
      - Vision targets compare against (B,N,D).
      - Diffusion targets compare against ONE selected timestep (B,N,D) using that target's sigma.

    This fixes the bug where diffusion targets were skipped when source was vision.
    """
    model.train()
    diffusion_models = set(diffusion_models)

    for (acts, meta), _y in tqdm(dataloader, desc="train", dynamic_ncols=True):
        source = random.choice(list(acts.keys()))

        optimizer.zero_grad()
        loss = 0.0

        if source in diffusion_models:
            # acts[source]: (B,T,N,D)
            x_src = acts[source].to(device)
            B, Tsrc, N, D = x_src.shape

            src_sigmas_bt = _get_sigmas_bt(meta, source, B, device)
            if src_sigmas_bt is None:
                raise KeyError(
                    f"Source '{source}' is diffusion but sigmas not found in metadata. "
                    f"Expected meta['sigmas_by_model'][{source!r}] or meta['sigmas']."
                )

            # iterate all timesteps for the source diffusion model
            for t_src in range(Tsrc):
                sigma_src = src_sigmas_bt[:, t_src]  # (B,)
                x_t = x_src[:, t_src]                # (B,N,D)

                _z_pre, z = model.encode(x_t, source=source, sigma=sigma_src)

                # decode to every target
                for target, x_target in acts.items():
                    x_target = x_target.to(device)

                    if target in diffusion_models:
                        # x_target: (B,Ttgt,N,D)
                        if x_target.dim() != 4:
                            raise ValueError(
                                f"Expected diffusion target '{target}' to be (B,T,N,D), got {tuple(x_target.shape)}"
                            )
                        Ttgt = x_target.shape[1]

                        tgt_sigmas_bt = _get_sigmas_bt(meta, target, B, device)
                        if tgt_sigmas_bt is None:
                            # if you truly don't have sigmas for a diffusion target, you can't apply temporal affine
                            raise KeyError(
                                f"Target '{target}' is diffusion but sigmas not found in metadata. "
                                f"Expected meta['sigmas_by_model'][{target!r}] or meta['sigmas']."
                            )

                        t_tgt = _map_timestep_idx(t_src, Tsrc, Ttgt)
                        sigma_tgt = tgt_sigmas_bt[:, t_tgt]   # (B,)
                        x_hat = model.decode(z, target=target, sigma=sigma_tgt)
                        loss = loss + mse_flat(x_hat, x_target[:, t_tgt])

                    else:
                        # vision target: (B,N,D)
                        if x_target.dim() != 3:
                            raise ValueError(
                                f"Expected vision target '{target}' to be (B,N,D), got {tuple(x_target.shape)}"
                            )
                        x_hat = model.decode(z, target=target, sigma=None)
                        loss = loss + mse_flat(x_hat, x_target)

        else:
            # vision source: (B,N,D)
            x_src = acts[source].to(device)
            B, N, D = x_src.shape

            _z_pre, z = model.encode(x_src, source=source, sigma=None)

            # decode to every target, INCLUDING diffusion
            for target, x_target in acts.items():
                x_target = x_target.to(device)

                if target in diffusion_models:
                    # pick a timestep for THIS diffusion target
                    if x_target.dim() != 4:
                        raise ValueError(
                            f"Expected diffusion target '{target}' to be (B,T,N,D), got {tuple(x_target.shape)}"
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
                    loss = loss + mse_flat(x_hat, x_target[:, t_tgt])

                else:
                    # vision target: (B,N,D)
                    if x_target.dim() != 3:
                        raise ValueError(
                            f"Expected vision target '{target}' to be (B,N,D), got {tuple(x_target.shape)}"
                        )
                    x_hat = model.decode(z, target=target, sigma=None)
                    loss = loss + mse_flat(x_hat, x_target)

        loss.backward()
        optimizer.step()
