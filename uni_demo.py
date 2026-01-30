"""
Unified USAE entry point:
- One UniversalSAE shared latent space
- Separate encoder/decoder per model (vision + diffusion)
- Diffusion models use temporal-aware pre/post affine adapters (sigma-conditioned)

Expects:
- combined_npz=True
- per-image combined files: .../<split>/<class>/<img>_combined.npz
- each combined file contains:
    - activations: keys == model name, arrays shaped:
        vision:    (N, D)
        diffusion: (T, N, D)
    - diffusion metadata keys (recommended):
        <MODEL>_sigmas or <MODEL>__sigmas  (T,)
        <MODEL>_timesteps or <MODEL>__timesteps (T,) optional
      also supports global: sigmas, timesteps
"""

from __future__ import annotations

import os
import time
import math
import yaml
import shutil
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageNet
from tqdm import tqdm

from universal_sae import UniversalSAE
from train import train_universal_sae


# ----------------------------
# Helpers
# ----------------------------

MODELS_WITH_CLS_TOKEN = {"DinoV2", "ViT", "CLIP"}


def _expand_run_name(template: str, model_names: list[str], cfg_global: Dict[str, Any], sae_params: Dict[str, Any]) -> str:
    # Fill ${model_1}... with model names in order
    mapping = dict(cfg_global)
    mapping.update(sae_params)
    for i in range(1, 9):
        mapping[f"model_{i}"] = model_names[i - 1] if (i - 1) < len(model_names) else ""
    out = template
    for k, v in mapping.items():
        out = out.replace("${" + str(k) + "}", str(v))
    return out


def _flatten_tokens_for_stats(x: torch.Tensor) -> torch.Tensor:
    """
    (N,D) -> (N,D)
    (T,N,D) -> (T*N, D)
    """
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        t, n, d = x.shape
        return x.reshape(t * n, d)
    raise ValueError(f"Unexpected activation rank {x.dim()} shape={tuple(x.shape)}")


def _maybe_strip_cls(x: torch.Tensor, source: str, use_class_tokens: bool) -> torch.Tensor:
    # Only meaningful for vision (N,D). For diffusion (T,N,D), do nothing.
    if (x.dim() == 2) and (source in MODELS_WITH_CLS_TOKEN) and (not use_class_tokens):
        return x[1:, :]
    return x


def _get_npz_key(npz: np.lib.npyio.NpzFile, key: str) -> Optional[np.ndarray]:
    return npz[key] if key in npz.files else None


def _find_meta(npz: np.lib.npyio.NpzFile, model: str, field: str) -> Optional[np.ndarray]:
    """
    Supports:
      <MODEL>_sigmas, <MODEL>__sigmas, sigmas
      <MODEL>_timesteps, <MODEL>__timesteps, timesteps
    """
    candidates = [f"{model}_{field}", f"{model}__{field}", field]
    for k in candidates:
        if k in npz.files:
            return npz[k]
    return None


# ----------------------------
# Dataset that returns (acts, meta) for unified training
# ----------------------------

class ImageNetCombinedActsWithMeta(Dataset):
    def __init__(
        self,
        imagenet_root: str,
        activation_root: str,
        split: str,
        sources: list[str],
        diffusion_models: set[str],
        combined_npz: bool = True,
        target_class: Optional[str | int] = None,
        standardize: bool = False,
        divide_norm: bool = False,
        use_class_tokens: bool = True,
        stats_sample_size: int = 1000,
    ):
        assert combined_npz, "This unified pipeline expects combined_npz=True"

        self.split = split
        self.sources = sources
        self.diffusion_models = set(diffusion_models)
        self.activation_root = activation_root
        self.standardize = standardize
        self.divide_norm = divide_norm
        self.use_class_tokens = use_class_tokens

        # Load ImageNet index
        ds = ImageNet(imagenet_root, split=split)
        self.class_to_idx = ds.class_to_idx
        samples = ds.samples  # list[(img_path, target)]

        # Optional class filter
        if target_class is not None and target_class != "ALL":
            if isinstance(target_class, str):
                if target_class not in self.class_to_idx:
                    raise ValueError(f"Invalid WordNet ID/class name: {target_class}")
                target_class = self.class_to_idx[target_class]
            samples = [(p, y) for (p, y) in samples if y == int(target_class)]
            print(f"[dataset] Filtered to {len(samples)} samples for class {target_class}")
        else:
            print(f"[dataset] Using ALL classes ({len(samples)} samples)")

        # Precompute combined npz paths aligned with samples
        self.samples_used: list[tuple[str, int]] = []
        split_root = os.path.join(imagenet_root, split)
        for img_path, y in tqdm(samples, desc="[dataset] Indexing combined npz", dynamic_ncols=True):
            rel_path = os.path.relpath(img_path, split_root)
            class_dir = os.path.dirname(rel_path)
            act_path = os.path.join(
                activation_root,
                split,
                class_dir,
                os.path.basename(img_path).replace(".JPEG", "_combined.npz"),
            )
            self.samples_used.append((act_path, int(y)))

        # Standardization stats
        self.standardization_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        if self.standardize:
            self._compute_standardization_stats(stats_sample_size)

    def _compute_standardization_stats(self, sample_size: int):
        sample_size = min(sample_size, len(self.samples_used))
        idxs = np.random.choice(len(self.samples_used), sample_size, replace=False)

        for source in self.sources:
            print(f"[stats] Computing mean/std for {source} over {sample_size} samples")
            rows = []

            for i in tqdm(idxs, desc=f"[stats] {source}", dynamic_ncols=True):
                act_path, _ = self.samples_used[int(i)]
                npz = np.load(act_path, mmap_mode="r")
                raw = _get_npz_key(npz, source)
                if raw is None:
                    raise KeyError(f"Missing activation key '{source}' in {act_path}")
                x = torch.from_numpy(raw)
                x = _maybe_strip_cls(x, source, self.use_class_tokens)
                rows.append(_flatten_tokens_for_stats(x))

            rows = torch.cat(rows, dim=0)  # (total_tokens, D)
            mean = rows.mean(dim=0)
            std = rows.std(dim=0)

            self.standardization_stats[source] = {"mean": mean, "std": std}
            print(f"[stats] {source}: mean/std shape = {tuple(mean.shape)}")

    def __len__(self):
        return len(self.samples_used)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]], int]:
        act_path, y = self.samples_used[idx]
        npz = np.load(act_path, mmap_mode="r")

        acts: Dict[str, torch.Tensor] = {}
        meta: Dict[str, Dict[str, torch.Tensor]] = {}

        for source in self.sources:
            raw = _get_npz_key(npz, source)
            if raw is None:
                raise KeyError(f"Missing activation key '{source}' in combined npz: {act_path}")

            x = torch.from_numpy(raw)  # (N,D) or (T,N,D)
            x = _maybe_strip_cls(x, source, self.use_class_tokens)

            # standardize/normalize
            if self.standardize:
                m = self.standardization_stats[source]["mean"]
                s = self.standardization_stats[source]["std"]
                x = (x - m) / (s + 1e-5)
            elif self.divide_norm:
                x = x / (x.norm(dim=-1, keepdim=True) + 1e-9)

            acts[source] = x

            # diffusion metadata
            if source in self.diffusion_models:
                sig = _find_meta(npz, source, "sigmas")
                if sig is None:
                    raise KeyError(
                        f"Diffusion source '{source}' present but no sigmas found in {act_path}. "
                        f"Expected '{source}_sigmas' or '{source}__sigmas' or 'sigmas'."
                    )
                sig_t = torch.from_numpy(sig).float()  # (T,)
                meta[source] = {"sigmas": sig_t}

                ts = _find_meta(npz, source, "timesteps")
                if ts is not None:
                    meta[source]["timesteps"] = torch.from_numpy(ts).long()

        return (acts, meta), y


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    # ----- Load config -----
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    CONFIG: Dict[str, Any] = cfg["global"]
    SAE_PARAMS: Dict[str, Any] = cfg.get("sae_params", {})
    MODEL_ZOO: Dict[str, Any] = cfg["model_zoo"]
    VIZ_CFG: Dict[str, Any] = cfg.get("viz", {})

    # ----- Sources + diffusion models -----
    sources = list(MODEL_ZOO.keys())
    diffusion_models = set(CONFIG.get("diffusion_models", []))  # e.g. ["SD3","FLUX"]
    if len(diffusion_models) == 0:
        print("[warn] CONFIG.global.diffusion_models is empty. "
              "If you intend to train with diffusion sources, add them there.")

    # ----- Run name -----
    run_name_template = CONFIG.get("run_name", "usae_run")
    run_name = _expand_run_name(run_name_template, sources, CONFIG, SAE_PARAMS)
    CONFIG["run_name"] = run_name

    # ----- Build dataset (ALWAYS unified vision+diffusion) -----
    assert CONFIG.get("combined_npz", True) is True, "Unified training expects combined_npz=True"

    dataset = ImageNetCombinedActsWithMeta(
        imagenet_root=CONFIG["imagenet_root"],
        activation_root=CONFIG["path_to_cache"],
        split="train",
        sources=sources,
        diffusion_models=diffusion_models,
        combined_npz=True,
        target_class=CONFIG.get("target_class", "ALL"),
        standardize=bool(CONFIG.get("standardize", False)),
        divide_norm=bool(CONFIG.get("divide_norm", False)),
        use_class_tokens=bool(CONFIG.get("use_class_tokens", True)),
        stats_sample_size=int(CONFIG.get("stats_sample_size", 1000)),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=True,
        num_workers=int(CONFIG.get("num_workers", 8)),
        pin_memory=True,
    )

    # ----- Build UniversalSAE -----
    model_dims = {m: int(MODEL_ZOO[m]["input_shape"]) for m in MODEL_ZOO.keys()}

    # shared latent space size (pick one key; support old config too)
    latent_dim = int(CONFIG.get("latent_dim", CONFIG.get("nb_components", int(CONFIG["exp_factor"]) * int(CONFIG["input_shape"]))))

    model = UniversalSAE(
        model_dims=model_dims,
        latent_dim=latent_dim,
        diffusion_models=diffusion_models,
        timestep_dim=int(CONFIG.get("timestep_dim", 256)),
        top_k=int(SAE_PARAMS.get("top_k", CONFIG.get("top_k", 32))),
        topk_temperature=float(CONFIG.get("topk_temperature", 0.1)),
        use_soft_topk=bool(CONFIG.get("use_soft_topk", True)),
    )

    device = str(CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    # ----- Optimizer -----
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(CONFIG.get("lr", 3e-4)),
        weight_decay=float(CONFIG.get("weight_decay", 1e-5)),
    )

    # ----- Train epochs -----
    nb_epochs = int(CONFIG.get("nb_epochs", 1))
    ckpt_dir = os.path.join("weights", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(nb_epochs):
        t0 = time.time()

        # Your train_universal_sae loops once over the dataloader and steps optimizer each batch :contentReference[oaicite:3]{index=3}
        train_universal_sae(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            diffusion_models=diffusion_models,
            device=device,
        )

        dt = time.time() - t0
        print(f"[epoch] {epoch+1}/{nb_epochs} done in {dt:.2f}s")

        # checkpoint
        if (epoch % int(CONFIG.get("save_every", 5)) == 0) or (epoch + 1 == nb_epochs):
            ckpt_path = os.path.join(ckpt_dir, f"usae_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "run_name": run_name,
                    "diffusion_models": sorted(list(diffusion_models)),
                    "model_dims": model_dims,
                    "latent_dim": latent_dim,
                },
                ckpt_path,
            )
            print(f"[ckpt] saved: {ckpt_path}")

            # write checkpoint path into each model entry (single shared USAE checkpoint)
            for m in MODEL_ZOO:
                MODEL_ZOO[m]["checkpoint_path"] = ckpt_path

    # ----- Save results config -----
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # copy original yaml
    results_yaml = os.path.join(results_dir, os.path.basename(config_path))
    shutil.copy(config_path, results_yaml)

    # update copied yaml with mean/std + checkpoint paths
    with open(results_yaml, "r") as f:
        out_cfg = yaml.safe_load(f)

    out_cfg["global"] = CONFIG
    out_cfg["viz"] = VIZ_CFG
    out_cfg["sae_params"] = SAE_PARAMS
    out_cfg["model_zoo"] = MODEL_ZOO

    if bool(CONFIG.get("standardize", False)):
        # store vectors as lists for yaml
        for m in sources:
            mean = dataset.standardization_stats[m]["mean"]
            std = dataset.standardization_stats[m]["std"]
            out_cfg["model_zoo"][m]["model_mean"] = mean.detach().cpu().tolist()
            out_cfg["model_zoo"][m]["model_std"] = std.detach().cpu().tolist()

    with open(results_yaml, "w") as f:
        yaml.dump(out_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"[done] Results stored in: {results_dir}")
