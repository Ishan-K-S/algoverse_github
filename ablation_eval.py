"""Reviewer Q1: what survives when the per-token alignment term is removed?

Evaluates one or two checkpoints under an IDENTICAL protocol and prints a
comparison table. Both columns must come from the same script on the same
split -- reading the main run's numbers off an old W&B history while computing
the ablation's fresh invites exactly the "were these measured the same way?"
objection this table exists to close.

Produces, on the held-out val split:
  - per-token co-fire Jaccard, its chance baseline, and the lift
  - self/cross reconstruction MSE for all four (source, target) pairs
  - R^2 per pair = 1 - MSE / Var(target), with Var reported so the number is
    checkable rather than assumed (activations are standardized, so Var ~ 1
    and R^2 ~ 1 - MSE, but this measures it instead of asserting it)

Dictionary occupancy / partition score / bag cosine / top-k Jaccard come from
dictionary_diagnostic.py -- run that separately per checkpoint rather than
having this script produce a second, subtly-different occupancy number.

    python ablation_eval.py --ckpt_main <main.pth> --ckpt_noalign <noalign.pth>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coco_dataset_setup import split_train_val
from data import CocoActivationDataset
from top_activating_images import build_spatial_aligner, load_universal_sae
from train import evaluate_universal_sae

CACHE_ROOT = "/content/combined_cache"
CONFIG_PATH = "/content/algoverse_github/config.yaml"


def build_val_loader(cache_root, sources, model, cfg, aligner, eval_g, batch_size):
    """The same val split uni_demo.py trains against: seeded, persisted, disjoint."""
    anchor = "_combined.npz"
    all_stems = sorted(
        f[: -len(anchor)] for f in os.listdir(cache_root) if f.endswith(anchor)
    )
    _train_stems, val_stems = split_train_val(
        all_stems,
        save_dir=cache_root,
        val_fraction=float(eval_g.get("val_fraction", 0.2)),
        seed=int(eval_g.get("val_split_seed", 42)),
    )
    ds = CocoActivationDataset(
        cache_root=cache_root,
        sources=list(sources),
        combined_npz=True,
        standardize=bool(eval_g.get("standardize", True)),
        divide_norm=bool(eval_g.get("divide_norm", False)),
        use_class_tokens=bool(eval_g.get("use_class_tokens", False)),
        return_metadata=True,
        diffusion_models=list(model.diffusion_models),
        # Training stats, never refit to val -- refitting would leak.
        standardization_stats=getattr(model, "_standardization_stats", None),
        spatial_aligner=aligner,
        allowed_stems=val_stems,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False), len(val_stems)


@torch.no_grad()
def target_variance(loader, model, aligner, eval_g, device) -> Dict[str, float]:
    """Var of each model's activations as the loss actually sees them: after
    standardization, timestep selection and spatial alignment. This is the
    denominator of R^2.
    """
    fixed_t = eval_g.get("fixed_timestep_idx", None)
    n = {}
    s = {}
    ss = {}
    for (acts, _meta), _ in loader:
        for name, x in acts.items():
            x = x.to(device).float()
            if name in model.diffusion_models and x.dim() == 4:  # (B,T,N,D)
                t = 0 if fixed_t is None else max(0, min(int(fixed_t), x.shape[1] - 1))
                x = x[:, t]
            if aligner is not None:
                x = aligner.align(x, source=name)
            s[name] = s.get(name, 0.0) + x.sum().item()
            ss[name] = ss.get(name, 0.0) + (x * x).sum().item()
            n[name] = n.get(name, 0) + x.numel()
    return {k: ss[k] / n[k] - (s[k] / n[k]) ** 2 for k in n}


def eval_checkpoint(ckpt_path, args) -> dict:
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model, cfg = load_universal_sae(ckpt_path, args.config, device)
    eval_g = getattr(model, "_training_global", cfg.get("global", {}))
    aligner = build_spatial_aligner({"global": eval_g, "model_zoo": cfg.get("model_zoo", {})})

    loader, n_val = build_val_loader(
        args.cache_root, args.sources, model, cfg, aligner, eval_g, args.batch_size
    )
    print(f"[eval] {os.path.basename(ckpt_path)}: {n_val} held-out images")

    metrics = evaluate_universal_sae(
        model=model,
        dataloader=loader,
        diffusion_models=set(model.diffusion_models),
        device=device,
        fixed_timestep_idx=eval_g.get("fixed_timestep_idx", None),
        spatial_aligner=aligner,
    )
    var = target_variance(loader, model, aligner, eval_g, device)

    # R^2 = 1 - MSE / Var(target). Key is "val/loss_<src>_to_<tgt>".
    for key in [k for k in metrics if k.startswith("val/loss_")]:
        pair = key[len("val/loss_"):]
        if "_to_" not in pair:
            continue
        tgt = pair.split("_to_")[1]
        if tgt in var and var[tgt] > 0:
            metrics[f"val/r2_{pair}"] = 1.0 - metrics[key] / var[tgt]
    for k, v in var.items():
        metrics[f"val/target_variance_{k}"] = v

    metrics["_checkpoint"] = ckpt_path
    # From the CHECKPOINT's saved config, not config.yaml -- this is the line
    # that proves the two columns really differ in the intended way.
    trained_sae = getattr(model, "_training_sae_params", {}) or {}
    metrics["_latent_align_weight"] = trained_sae.get("latent_align_weight")
    metrics["_top_k"] = trained_sae.get("top_k")
    metrics["_cross_weight"] = trained_sae.get("cross_weight")
    metrics["_pre_topk_align_weight"] = trained_sae.get("pre_topk_align_weight")
    metrics["_n_val_images"] = n_val
    return metrics


def fmt(v, nd=4):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def print_table(main: dict, abl: Optional[dict]):
    keys = [k for k in sorted(main) if k.startswith("val/") and not k.startswith("val/target_variance")]
    label = {"val/": ""}
    print("\n" + "=" * 92)
    if abl is None:
        print(f"{'METRIC':<46} {'VALUE':>18}")
        print("=" * 92)
        for k in keys:
            print(f"{k.replace('val/',''):<46} {fmt(main[k]):>18}")
    else:
        print(f"{'METRIC':<46} {'MAIN (align=0.5)':>18} {'NO-ALIGN (0.0)':>18}")
        print("=" * 92)
        for k in keys:
            a, b = main.get(k), abl.get(k)
            print(f"{k.replace('val/',''):<46} {fmt(a):>18} {fmt(b):>18}")
    print("=" * 92)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_main", required=True)
    p.add_argument("--ckpt_noalign", default=None,
                   help="Omit to evaluate a single checkpoint.")
    p.add_argument("--cache_root", default=CACHE_ROOT)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--sources", nargs="+", default=["DinoV2", "PixArt"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", default="/content/results/ablation_eval.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    main_m = eval_checkpoint(args.ckpt_main, args)
    abl_m = eval_checkpoint(args.ckpt_noalign, args) if args.ckpt_noalign else None
    print_table(main_m, abl_m)

    with open(args.out, "w") as f:
        json.dump({"main": main_m, "noalign": abl_m}, f, indent=2)
    print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
