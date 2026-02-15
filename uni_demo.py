"""Unified USAE entry point.

This version removes the custom dataset and uses data.py::ImageNetActivationDataset
so the metadata format is unified with train.py.

Expected combined npz format per image:
  .../<split>/<class>/<img>_combined.npz
and each combined file contains activation arrays keyed by model name.

For diffusion/flow models, include sigma metadata (preferred):
  <MODEL>__sigmas  (T,)
Optionally:
  <MODEL>__timesteps (T,)

See data.py::_extract_metadata_from_combined for the full supported key set.
"""

from __future__ import annotations

import os
import time
import yaml
import shutil
from typing import Dict, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import ImageNetActivationDataset
from universal_sae import UniversalSAE
from train import train_universal_sae


def _expand_run_name(template: str, model_names: list[str], cfg_global: Dict[str, Any], sae_params: Dict[str, Any]) -> str:
    """Expand ${model_1}.. placeholders in a run-name template."""
    mapping = dict(cfg_global)
    mapping.update(sae_params)
    for i in range(1, 9):
        mapping[f"model_{i}"] = model_names[i - 1] if (i - 1) < len(model_names) else ""
    out = template
    for k, v in mapping.items():
        out = out.replace("${" + str(k) + "}", str(v))
    return out


def _require_nonempty(cfg: Dict[str, Any], key: str) -> str:
    v = cfg.get(key, None)
    if v is None or (isinstance(v, str) and len(v.strip()) == 0):
        raise ValueError(f"CONFIG.global.{key} is empty. Please set it in config.yaml.")
    return str(v)


def _parse_int_field(x: Any, name: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(
            f"Expected {name} to be an int-like value, got {x!r}. "
            "Please replace placeholders like '<D_SD3>' with real integers in config.yaml."
        ) from e


if __name__ == "__main__":
    # ----- Load config -----
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    CONFIG: Dict[str, Any] = cfg.get("global", {})
    SAE_PARAMS: Dict[str, Any] = cfg.get("sae_params", {})
    MODEL_ZOO: Dict[str, Any] = cfg.get("model_zoo", {})
    VIZ_CFG: Dict[str, Any] = cfg.get("viz", {})

    if not MODEL_ZOO:
        raise ValueError("config.yaml missing model_zoo entries.")

    # ----- Required paths -----
    imagenet_root = _require_nonempty(CONFIG, "imagenet_root")
    activation_root = _require_nonempty(CONFIG, "path_to_cache")

    # ----- Sources + diffusion models -----
    sources = list(MODEL_ZOO.keys())
    diffusion_models = set(CONFIG.get("diffusion_models", []))

    # ----- Extract model dimensions AND token counts from model_zoo -----
    model_dims = {}
    model_tokens = {}
    for m in MODEL_ZOO.keys():
        model_dims[m] = _parse_int_field(MODEL_ZOO[m]["input_shape"], f"model_zoo.{m}.input_shape")
        if "num_tokens" in MODEL_ZOO[m]:
            model_tokens[m] = _parse_int_field(MODEL_ZOO[m]["num_tokens"], f"model_zoo.{m}.num_tokens")

    print(f"[config] Model dimensions: {model_dims}")
    print(f"[config] Model tokens: {model_tokens}")
    print(f"[config] Diffusion models: {diffusion_models}")

    # ----- Run name -----
    run_name_template = CONFIG.get("run_name", "usae_run")
    run_name = _expand_run_name(run_name_template, sources, CONFIG, SAE_PARAMS)
    CONFIG["run_name"] = run_name

    # ----- Dataset (unified w/ train.py metadata) -----
    if CONFIG.get("combined_npz", True) is not True:
        raise ValueError("Unified training expects CONFIG.global.combined_npz: true")

    dataset = ImageNetActivationDataset(
        root=imagenet_root,
        activation_root=activation_root,
        sources=sources,
        combined_npz=True,
        split="train",
        target_class=CONFIG.get("target_class", "ALL"),
        standardize=bool(CONFIG.get("standardize", False)),
        divide_norm=bool(CONFIG.get("divide_norm", False)),
        use_class_tokens=bool(CONFIG.get("use_class_tokens", True)),
        return_metadata=True,
        diffusion_models=list(diffusion_models),
    )

    # Optionally recompute standardization stats with a configured sample size
    if bool(CONFIG.get("standardize", False)) and CONFIG.get("stats_sample_size") is not None:
        stats_n = _parse_int_field(CONFIG.get("stats_sample_size"), "CONFIG.global.stats_sample_size")
        dataset._compute_standardization_stats(sample_size=stats_n)

    dataloader = DataLoader(
        dataset,
        batch_size=_parse_int_field(CONFIG.get("batch_size", 32), "CONFIG.global.batch_size"),
        shuffle=True,
        num_workers=_parse_int_field(CONFIG.get("num_workers", 8), "CONFIG.global.num_workers"),
        pin_memory=True,
    )

    # ----- Build UniversalSAE -----
    exp_factor = _parse_int_field(CONFIG.get("exp_factor", 8), "CONFIG.global.exp_factor")
    input_shape = _parse_int_field(CONFIG.get("input_shape", 768), "CONFIG.global.input_shape")
    default_latent = exp_factor * input_shape
    latent_dim = _parse_int_field(CONFIG.get("latent_dim", CONFIG.get("nb_components", default_latent)), "CONFIG.global.latent_dim")
    
    # NEW: Shared latent tokens (canonical token count)
    shared_latent_tokens = _parse_int_field(
        CONFIG.get("shared_latent_tokens", 256), 
        "CONFIG.global.shared_latent_tokens"
    )

    model = UniversalSAE(
        model_dims=model_dims,
        latent_dim=latent_dim,
        diffusion_models=diffusion_models,
        model_tokens=model_tokens,
        shared_latent_tokens=shared_latent_tokens,  # NEW: pass canonical token count
        timestep_dim=_parse_int_field(CONFIG.get("timestep_dim", 256), "CONFIG.global.timestep_dim"),
        top_k=_parse_int_field(SAE_PARAMS.get("top_k", CONFIG.get("top_k", 32)), "sae_params.top_k"),
        topk_temperature=float(CONFIG.get("topk_temperature", 0.1)),
        use_soft_topk=bool(CONFIG.get("use_soft_topk", True)),
        interpolation_mode=str(CONFIG.get("interpolation_mode", "bilinear")),
    )

    device = str(CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    print(f"[model] UniversalSAE created with latent_dim={latent_dim}")
    print(f"[model] Shared latent tokens: {shared_latent_tokens}")
    print(f"[model] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ----- Optimizer -----
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(CONFIG.get("lr", 3e-4)),
        weight_decay=float(CONFIG.get("weight_decay", 1e-5)),
    )

    # ----- Train epochs -----
    nb_epochs = _parse_int_field(CONFIG.get("nb_epochs", 1), "CONFIG.global.nb_epochs")
    ckpt_dir = os.path.join("weights", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(nb_epochs):
        t0 = time.time()

        train_universal_sae(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            diffusion_models=diffusion_models,
            model_tokens=model_tokens,  # NEW: pass token counts for interpolation
            device=device,
        )

        dt = time.time() - t0
        print(f"[epoch] {epoch+1}/{nb_epochs} done in {dt:.2f}s")

        save_every = _parse_int_field(CONFIG.get("save_every", 5), "CONFIG.global.save_every")
        if (epoch % save_every == 0) or (epoch + 1 == nb_epochs):
            ckpt_path = os.path.join(ckpt_dir, f"usae_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "run_name": run_name,
                    "diffusion_models": sorted(list(diffusion_models)),
                    "model_dims": model_dims,
                    "model_tokens": model_tokens,
                    "shared_latent_tokens": shared_latent_tokens,  # NEW: save canonical token count
                    "latent_dim": latent_dim,
                },
                ckpt_path,
            )
            print(f"[ckpt] saved: {ckpt_path}")

            for m in MODEL_ZOO:
                MODEL_ZOO[m]["checkpoint_path"] = ckpt_path

    # ----- Save results config -----
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    results_yaml = os.path.join(results_dir, os.path.basename(config_path))
    shutil.copy(config_path, results_yaml)

    with open(results_yaml, "r") as f:
        out_cfg = yaml.safe_load(f)

    out_cfg["global"] = CONFIG
    out_cfg["viz"] = VIZ_CFG
    out_cfg["sae_params"] = SAE_PARAMS
    out_cfg["model_zoo"] = MODEL_ZOO

    if bool(CONFIG.get("standardize", False)):
        for m in sources:
            mean = dataset.standardization_stats[m]["mean"]
            std = dataset.standardization_stats[m]["std"]
            out_cfg["model_zoo"][m]["model_mean"] = mean.detach().cpu().tolist()
            out_cfg["model_zoo"][m]["model_std"] = std.detach().cpu().tolist()

    with open(results_yaml, "w") as f:
        yaml.dump(out_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"[done] Results stored in: {results_dir}")
