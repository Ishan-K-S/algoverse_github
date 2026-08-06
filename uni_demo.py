"""Unified USAE entry point for COCO-cached activations.

Expected cache layout (flat directory, no class subdirectories):
  <cache_root>/<img_stem>_<MODEL>.npz          (non-combined mode)
  <cache_root>/<img_stem>_combined.npz         (combined_npz=true mode)

Vision npz keys:    activation (N, D)
Diffusion npz keys: activation (T, N, D), sigmas (T,), timesteps (T,)
Combined npz keys:  <MODEL> (N,D or T,N,D),  <MODEL>__sigmas (T,),  <MODEL>__timesteps (T,)
"""

from __future__ import annotations

import os
import subprocess
import time
import yaml
import shutil
from typing import Dict, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import CocoActivationDataset
from universal_sae import UniversalSAE
from train import train_universal_sae, evaluate_universal_sae
from spatial_align import build_spatial_aligner_from_config
from coco_dataset_setup import split_train_val

# ---------------------------------------------------------------------------
# wandb setup
# ---------------------------------------------------------------------------

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[wandb] wandb not installed. Run `pip install wandb` to enable logging.")


def _init_wandb(cfg: Dict[str, Any], run_name: str) -> bool:
    """
    Initialize a wandb run. Returns True if successful.

    Authentication is delegated to WANDB_API_KEY in the environment, a Colab
    secret, or wandb's own stored credentials. Never commit API keys here.
    """
    if not WANDB_AVAILABLE:
        return False

    global_cfg = cfg.get("global", {})
    if not bool(global_cfg.get("use_wandb", True)):
        print("[wandb] Disabled via config (use_wandb: false).")
        return False

    wandb_project = global_cfg.get("wandb_project", "universal-sae")
    wandb_entity = global_cfg.get("wandb_entity", None)

    try:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                **cfg.get("global", {}),
                **cfg.get("sae_params", {}),
                "model_zoo_keys": list(cfg.get("model_zoo", {}).keys()),
            },
            resume="allow",
        )
        print(f"[wandb] Run started: {wandb.run.url}")
        return True
    except Exception as e:
        print(f"[wandb] Failed to initialize: {e}")
        return False


def _expand_run_name(
    template: str,
    model_names: list[str],
    cfg_global: Dict[str, Any],
    sae_params: Dict[str, Any],
) -> str:
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


def _code_version() -> str:
    """Short git hash of the code that produced a run, '-dirty' if uncommitted.

    We had no way to tell which commit a checkpoint came from, so a run that
    looked wrong could not be traced back to the code that made it.
    """
    repo = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo, capture_output=True, text=True, timeout=5,
        )
        if head.returncode != 0:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo, capture_output=True, text=True, timeout=5,
        )
        suffix = "-dirty" if dirty.stdout.strip() else ""
        return head.stdout.strip() + suffix
    except Exception:
        return "unknown"


def _parse_int_field(x: Any, name: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(
            f"Expected {name} to be an int-like value, got {x!r}. "
            "Please replace placeholders with real integers in config.yaml."
        ) from e


if __name__ == "__main__":
    # ----- Startup checks -----
    print(f"[uni_demo] ---- Starting training ----")
    print(f"[uni_demo] GPU available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[uni_demo] GPU           : {torch.cuda.get_device_name(0)}")
        print(f"[uni_demo] VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ----- Load config -----
    config_path = "/content/algoverse_github/config.yaml"
    print(f"[uni_demo] config_path   : {config_path}  exists={os.path.isfile(config_path)}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"[uni_demo] config.yaml not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    CONFIG: Dict[str, Any] = cfg.get("global", {})
    SAE_PARAMS: Dict[str, Any] = cfg.get("sae_params", {})
    MODEL_ZOO: Dict[str, Any] = cfg.get("model_zoo", {})
    VIZ_CFG: Dict[str, Any] = cfg.get("viz", {})

    if not MODEL_ZOO:
        raise ValueError("config.yaml missing model_zoo entries.")

    # ----- Required paths -----
    cache_root = _require_nonempty(CONFIG, "path_to_cache")
    print(f"[uni_demo] cache_root    : {cache_root}  exists={os.path.isdir(cache_root)}")
    if not os.path.isdir(cache_root):
        raise FileNotFoundError(
            f"[uni_demo] Combined cache not found: {cache_root}\n"
            "  Run combine_cached_acts.py first, or update path_to_cache in config.yaml."
        )

    # ----- Sources + diffusion models -----
    sources = list(MODEL_ZOO.keys())
    diffusion_models = set(CONFIG.get("diffusion_models", []))

    # ----- Extract model dimensions and token counts from model_zoo -----
    model_dims: Dict[str, int] = {}
    model_tokens: Dict[str, int] = {}
    for m in MODEL_ZOO.keys():
        model_dims[m] = _parse_int_field(MODEL_ZOO[m]["input_shape"], f"model_zoo.{m}.input_shape")
        model_tokens[m] = _parse_int_field(MODEL_ZOO[m]["num_tokens"], f"model_zoo.{m}.num_tokens")

    print(f"[config] Model dimensions : {model_dims}")
    print(f"[config] Model tokens     : {model_tokens}")
    print(f"[config] Diffusion models : {diffusion_models}")

    # ----- Run name -----
    # run_name comes straight from the config values, so if two of us run the same config
    # we get the same name. The runs stay separate in wandb but the checkpoint artifacts
    # are named f"{run_name}_epoch_{n}", so they all stack up as versions of one collection
    # and you can't tell whose weights are whose. run_tag keeps them apart.
    run_name_template = CONFIG.get("run_name", "usae_run")
    run_name = _expand_run_name(run_name_template, sources, CONFIG, SAE_PARAMS)
    run_tag = str(CONFIG.get("run_tag", "") or "").strip()
    if run_tag:
        run_name = f"{run_name}_{run_tag}"
    CONFIG["run_name"] = run_name
    print(f"[config] Run name         : {run_name}")

    # ----- wandb -----
    use_wandb = _init_wandb(cfg, run_name)
    log_every = _parse_int_field(CONFIG.get("wandb_log_every", 50), "CONFIG.global.wandb_log_every")

    # ----- Spatial aligner (built BEFORE the dataset, not after) -----
    # Build spatial aligner from config (None if disabled).
    # When enabled, model_tokens is rewritten to post-alignment counts so
    # the model and any downstream consumers see the effective token grid.
    # Moved ahead of dataset construction so the dataset
    # can compute standardization stats on the POOLED distribution -- a std
    # fit to individual native tokens under-normalizes whatever a source
    # looks like after alignment, since averaging correlated neighbouring
    # tokens lowers variance below what any one token had on its own.
    spatial_aligner = build_spatial_aligner_from_config(CONFIG, model_tokens)
    if spatial_aligner is not None:
        effective_tokens = spatial_aligner.effective_token_counts()
        print(f"[spatial-align] enabled: pooling all models to N={spatial_aligner.target_n_tokens}")
        for m_name in model_tokens:
            print(f"  {m_name:10s}: native={model_tokens[m_name]:5d}  effective={effective_tokens[m_name]:5d}")
        model_tokens_effective = dict(effective_tokens)
    else:
        print(f"[spatial-align] disabled (no spatial_align_to in config)")
        model_tokens_effective = dict(model_tokens)

    # ----- Dataset -----
    combined_npz = bool(CONFIG.get("combined_npz", True))

    # ----- Train/val split -----
    # Every number this project has produced -- every heatmap, every
    # top-activating-image, and the fixed_timestep_idx choice itself -- was
    # measured on the same 2000 images used to train. Discover the cached
    # stems the same way CocoActivationDataset does (anchor on the combined
    # suffix or the first source's suffix) so the split operates on exactly
    # the image set that's actually available, then persist an 80/20 split
    # next to selected_images.txt so it's stable across runs and resumes.
    anchor_suffix = "_combined.npz" if combined_npz else f"_{sources[0]}.npz"
    all_stems = sorted(
        f[: -len(anchor_suffix)] for f in os.listdir(cache_root) if f.endswith(anchor_suffix)
    )
    train_stems, val_stems = split_train_val(
        all_stems,
        save_dir=cache_root,
        val_fraction=float(CONFIG.get("val_fraction", 0.2)),
        seed=_parse_int_field(CONFIG.get("val_split_seed", 42), "CONFIG.global.val_split_seed"),
    )
    print(f"[uni_demo] Train/val split: {len(train_stems)} train / {len(val_stems)} val "
          f"(val_fraction={CONFIG.get('val_fraction', 0.2)})")

    dataset = CocoActivationDataset(
        cache_root=cache_root,
        sources=sources,
        combined_npz=combined_npz,
        standardize=bool(CONFIG.get("standardize", True)),
        divide_norm=bool(CONFIG.get("divide_norm", False)),
        use_class_tokens=bool(CONFIG.get("use_class_tokens", True)),
        return_metadata=combined_npz,
        diffusion_models=list(diffusion_models),
        stats_seed=CONFIG.get("stats_seed"),
        spatial_aligner=spatial_aligner,
        allowed_stems=train_stems,
    )

    # Optionally recompute standardisation stats with a configured sample size
    if bool(CONFIG.get("standardize", False)) and CONFIG.get("stats_sample_size") is not None:
        stats_n = _parse_int_field(CONFIG["stats_sample_size"], "CONFIG.global.stats_sample_size")
        dataset._compute_standardization_stats(sample_size=stats_n)

    dataloader = DataLoader(
        dataset,
        batch_size=_parse_int_field(CONFIG.get("batch_size", 32), "CONFIG.global.batch_size"),
        shuffle=True,
        num_workers=_parse_int_field(CONFIG.get("num_workers", 8), "CONFIG.global.num_workers"),
        pin_memory=True,
    )
    print(f"[uni_demo] Dataset size  : {len(dataset)} images (train split)")
    print(f"[uni_demo] Dataloader    : {len(dataloader)} batches  (batch_size={CONFIG.get('batch_size', 32)})")

    # Held-out val dataset: same cache, same standardization stats and spatial
    # aligner as train (stats/alignment are properties of the pipeline, not
    # something the val set should compute independently), disjoint images.
    val_dataset = CocoActivationDataset(
        cache_root=cache_root,
        sources=sources,
        combined_npz=combined_npz,
        standardize=bool(CONFIG.get("standardize", True)),
        divide_norm=bool(CONFIG.get("divide_norm", False)),
        use_class_tokens=bool(CONFIG.get("use_class_tokens", True)),
        return_metadata=combined_npz,
        diffusion_models=list(diffusion_models),
        standardization_stats=getattr(dataset, "standardization_stats", None),
        spatial_aligner=spatial_aligner,
        allowed_stems=val_stems,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_parse_int_field(CONFIG.get("batch_size", 32), "CONFIG.global.batch_size"),
        shuffle=False,
        # Same worker count as train. Against the current 15-timestep cache every
        # read is a full zlib inflate of a ~35MB array, so a
        # single-threaded val pass every epoch adds real wall-clock to the exact
        # bottleneck Stage 2 exists to remove.
        num_workers=_parse_int_field(CONFIG.get("num_workers", 8), "CONFIG.global.num_workers"),
        pin_memory=True,
    )
    print(f"[uni_demo] Val dataset   : {len(val_dataset)} images, {len(val_dataloader)} batches")

    # ----- Build UniversalSAE -----
    exp_factor = _parse_int_field(CONFIG.get("exp_factor", 8), "CONFIG.global.exp_factor")
    input_shape = _parse_int_field(CONFIG.get("input_shape", 768), "CONFIG.global.input_shape")
    default_latent = exp_factor * input_shape
    latent_dim = _parse_int_field(
        CONFIG.get("latent_dim", CONFIG.get("nb_components", default_latent)),
        "CONFIG.global.latent_dim",
    )

    model = UniversalSAE(
        model_dims=model_dims,
        latent_dim=latent_dim,
        diffusion_models=diffusion_models,
        model_tokens=model_tokens_effective,
        timestep_dim=_parse_int_field(CONFIG.get("timestep_dim", 256), "CONFIG.global.timestep_dim"),
        top_k=_parse_int_field(SAE_PARAMS.get("top_k", CONFIG.get("top_k", 32)), "sae_params.top_k"),
        cls_pool_mode=str(CONFIG.get("cls_pool_mode", "none")),
        use_tide=bool(CONFIG.get("use_tide", False)),
    )

    device = str(CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] UniversalSAE created with latent_dim={latent_dim}")
    print(f"[model] cls_pool_mode          : {CONFIG.get('cls_pool_mode', 'none')}")
    print(f"[model] use_tide               : {CONFIG.get('use_tide', False)}")
    print(f"[model] Total parameters       : {total_params:,}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.config.update({
            "total_params": total_params,
            "latent_dim": latent_dim,
            "model_dims": model_dims,
            "model_tokens_native": model_tokens,
            "model_tokens_effective": model_tokens_effective,
            "spatial_align_to": CONFIG.get("spatial_align_to", None),
            "cls_pool_mode": CONFIG.get("cls_pool_mode", "none"),
            "diffusion_models": sorted(list(diffusion_models)),
            "device": device,
        }, allow_val_change=True)

    # ----- Optimizer -----
    base_lr = float(CONFIG.get("lr", 3e-4))
    weight_decay = float(CONFIG.get("weight_decay", 1e-5))

    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )
    # train.py reads initial_lr off each group for warmup_scale logic, so set it.
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]
    print(f"[optim] lr={base_lr:.2e}  params={total_params:,}")

    # ----- LR scheduler (cosine decay to final_lr) -----
    nb_epochs = _parse_int_field(CONFIG.get("nb_epochs", 1), "CONFIG.global.nb_epochs")
    final_lr = float(CONFIG.get("final_lr", 1e-6))
    warmup_steps = _parse_int_field(CONFIG.get("warmup_steps", 1000), "CONFIG.global.warmup_steps")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=nb_epochs,
        eta_min=final_lr,
    )

    # ----- Train epochs -----
    ckpt_dir = os.path.join("weights", run_name)
    save_every = _parse_int_field(CONFIG.get("save_every", 5), "CONFIG.global.save_every")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(nb_epochs):
        t0 = time.time()

        last_loss = train_universal_sae(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            diffusion_models=diffusion_models,
            model_tokens=model_tokens_effective,
            device=device,
            epoch=epoch,
            use_wandb=use_wandb,
            log_every=log_every,
            curriculum_epochs=int(CONFIG.get("curriculum_epochs", 2)),
            curriculum_self_only=bool(CONFIG.get("curriculum_self_only", True)),
            balanced_sources=bool(CONFIG.get("balanced_sources", True)),
            warmup_steps=warmup_steps,
            ema_decay=float(CONFIG.get("ema_decay", 0.98)),
            spatial_aligner=spatial_aligner,
            latent_align_weight=float(
                SAE_PARAMS.get("latent_align_weight", CONFIG.get("latent_align_weight", 1.0))
            ),
            latent_align_mode=str(
                SAE_PARAMS.get("latent_align_mode", CONFIG.get("latent_align_mode", "per_token"))
            ),
            self_weight=float(SAE_PARAMS.get("self_weight", CONFIG.get("self_weight", 1.0))),
            cross_weight=float(SAE_PARAMS.get("cross_weight", CONFIG.get("cross_weight", 1.0))),
            fixed_timestep_idx=(
                None if CONFIG.get("fixed_timestep_idx", None) is None
                else int(CONFIG["fixed_timestep_idx"])
            ),
            pre_topk_align_weight=float(SAE_PARAMS.get("pre_topk_align_weight", CONFIG.get("pre_topk_align_weight", 0.0))),
            resample_dead=bool(SAE_PARAMS.get("resample_dead", CONFIG.get("resample_dead", False))),
            resample_interval=int(SAE_PARAMS.get("resample_interval", 500)),
            resample_dead_threshold=float(SAE_PARAMS.get("resample_dead_threshold", 1e-3)),
            resample_enc_scale=float(SAE_PARAMS.get("resample_enc_scale", 0.2)),
            resample_max_per_event=int(SAE_PARAMS.get("resample_max_per_event", 0)),
            resample_start_step=int(SAE_PARAMS.get("resample_start_step", 0)),
            resample_end_step=int(SAE_PARAMS.get("resample_end_step", 0)),
            grad_clip_norm=float(SAE_PARAMS.get("grad_clip_norm", CONFIG.get("grad_clip_norm", 1.0))),
        )

        steps_seen = (epoch + 1) * len(dataloader)
        if steps_seen >= warmup_steps:
            scheduler.step()

        dt = time.time() - t0
        current_lr = optimizer.param_groups[-1]["lr"]
        print(f"[epoch] {epoch + 1}/{nb_epochs} done in {dt:.2f}s  "
              f"lr={current_lr:.2e}")

        # Held-out validation pass: the only
        # number in this project measured on images the model never trained
        # on. No backward pass, no resampling, no curriculum/alignment terms.
        val_metrics = evaluate_universal_sae(
            model=model,
            dataloader=val_dataloader,
            diffusion_models=diffusion_models,
            device=device,
            fixed_timestep_idx=(
                None if CONFIG.get("fixed_timestep_idx", None) is None
                else int(CONFIG["fixed_timestep_idx"])
            ),
            spatial_aligner=spatial_aligner,
        )
        val_loss_str = ", ".join(
            f"{k.replace('val/loss_', '')}={v:.4f}"
            for k, v in val_metrics.items() if k.startswith("val/loss_")
        )
        print(f"[epoch] {epoch + 1}/{nb_epochs} val: {val_loss_str}")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({"epoch/time_seconds": dt, "epoch/index": epoch,
                       "epoch/lr": current_lr, **val_metrics})

        if (epoch % save_every == 0) or (epoch + 1 == nb_epochs):
            ckpt_path = os.path.join(ckpt_dir, f"usae_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "run_name": run_name,
                    "diffusion_models": sorted(list(diffusion_models)),
                    "model_dims": model_dims,
                    "model_tokens": model_tokens_effective,   # post-alignment
                    "model_tokens_native": model_tokens,       # pre-alignment
                    "spatial_align_to": CONFIG.get("spatial_align_to", None),
                    "latent_dim": latent_dim,
                    # Enough to identify what produced this checkpoint. The full
                    # config is above; this is the part config.yaml can't tell you.
                    "manifest": {
                        "code_version": _code_version(),
                        "cache_root": CONFIG.get("path_to_cache"),
                        "n_images": len(dataset),
                        "fixed_timestep_idx": CONFIG.get("fixed_timestep_idx"),
                    },
                    "standardization_stats": {
                        source: {
                            "mean": values["mean"].detach().cpu(),
                            "std": values["std"].detach().cpu(),
                        }
                        for source, values in getattr(dataset, "standardization_stats", {}).items()
                    },
                },
                ckpt_path,
            )
            print(f"[ckpt] saved: {ckpt_path}")

            if use_wandb and WANDB_AVAILABLE:
                artifact = wandb.Artifact(
                    name=f"{run_name}_epoch_{epoch}",
                    type="model",
                    description=f"UniversalSAE checkpoint at epoch {epoch}",
                )
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)

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

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("[wandb] Run finished.")
