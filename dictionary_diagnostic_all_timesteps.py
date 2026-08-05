"""
dictionary_diagnostic_all_timesteps.py

Same as dictionary_diagnostic.py, but processes EVERY PixArt timestep per
image instead of just the last. This is the correct measurement when the
model was trained with random PixArt timestep sampling — the encoder may
use different features at different denoising stages, so evaluating only
the last timestep undercounts PixArt's effective dictionary.

For each image:
  - DinoV2 encoded once (vision, no timestep)
  - PixArt encoded T times (once per timestep)
  - Feature usage accumulated over all PixArt encodings

Top-K Jaccard is computed against the per-image AVERAGED PixArt code
(mean over timesteps), since that's the most natural "this image's PixArt
representation" summary.

Usage: same as dictionary_diagnostic.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
import yaml


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--repo_root", default=None)
    p.add_argument("--n_images", type=int, default=200)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="dict_diag_all_t.npz")
    p.add_argument("--plot", default="dict_diag_all_t.png")
    return p.parse_args()


def main():
    args = _parse_args()
    repo_root = args.repo_root or os.path.dirname(os.path.abspath(args.config))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from data import CocoActivationDataset
    from universal_sae import UniversalSAE
    from spatial_align import build_spatial_aligner_from_config

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    print(f"[diag] loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    with open(args.config) as f:
        cfg_file = yaml.safe_load(f)
    g_file = cfg_file.get("global", {})
    sae_p = cfg_file.get("sae_params", {})
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    g_ckpt = ckpt_cfg.get("global", {})
    sae_p_ckpt = ckpt_cfg.get("sae_params", {})

    def pick(k, default=None):
        if k in g_ckpt: return g_ckpt[k]
        if k in g_file: return g_file[k]
        return default

    model_tokens_eff = ckpt["model_tokens"]
    model_tokens_native = ckpt.get("model_tokens_native", model_tokens_eff)

    model = UniversalSAE(
        model_dims=ckpt["model_dims"],
        latent_dim=ckpt["latent_dim"],
        diffusion_models=set(ckpt.get("diffusion_models", g_file.get("diffusion_models", []))),
        model_tokens=model_tokens_eff,
        timestep_dim=int(pick("timestep_dim", 256)),
        # Checkpoint's own sae_params wins -- this sets the model's actual TopK
        # width, which must match what it was trained with.
        top_k=int(sae_p_ckpt.get("top_k", sae_p.get("top_k", pick("top_k", 64)))),
        cls_pool_mode=str(pick("cls_pool_mode", "none")),
        use_tide=bool(pick("use_tide", False)),
    )
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:    print(f"[diag] missing keys: {len(missing)}")
    if unexpected: print(f"[diag] unexpected keys: {len(unexpected)}")
    model = model.to(device).eval()

    align_to = ckpt.get("spatial_align_to", pick("spatial_align_to", None))
    aligner = build_spatial_aligner_from_config({"spatial_align_to": align_to}, model_tokens_native)
    if aligner is not None:
        print(f"[diag] spatial alignment ON: target grid {aligner.target_grid_size}x{aligner.target_grid_size}")

    sources = ["DinoV2", "PixArt"]
    ds = CocoActivationDataset(
        cache_root=args.cache,
        sources=sources,
        combined_npz=True,
        standardize=bool(g_file.get("standardize", True)),
        divide_norm=bool(g_file.get("divide_norm", False)),
        use_class_tokens=bool(g_file.get("use_class_tokens", False)),
        return_metadata=True,
        diffusion_models=list(model.diffusion_models),
        # Reuse the exact stats the model was trained with, rather than recomputing
        # from a fresh (previously unseeded) random cache sample every invocation.
        standardization_stats=ckpt.get("standardization_stats"),
        stats_seed=pick("stats_seed", 0),
    )
    n_use = min(args.n_images, len(ds))
    print(f"[diag] analyzing {n_use}/{len(ds)} images, ALL PixArt timesteps each")

    K = model.latent_dim
    print(f"[diag] latent_dim K = {K}, top_k = {model.top_k}")

    # Per-feature firing tracking
    fires_dino = torch.zeros(K, dtype=torch.long, device=device)
    fires_pixart = torch.zeros(K, dtype=torch.long, device=device)
    cofire = torch.zeros(K, dtype=torch.long, device=device)

    # Per-timestep PixArt firing breakdown (so we can see if individual timesteps
    # use narrow subsets even when the union is wide)
    fires_per_timestep = None  # will allocate after we see T

    jaccard_scores = []
    image_cosines = []

    # Also track per-timestep feature usage to see if individual timesteps collapse
    with torch.no_grad():
        for i in range(n_use):
            (acts, meta), _ = ds[i]
            x_dino = acts["DinoV2"].to(device).unsqueeze(0).float()
            x_pix_full = acts["PixArt"].to(device).float()  # (T, N, D)
            T = x_pix_full.shape[0]

            if fires_per_timestep is None:
                fires_per_timestep = torch.zeros(T, K, dtype=torch.long, device=device)

            sig_map = meta.get("sigmas_by_model", {})
            sigmas_pix = sig_map.get("PixArt", meta.get("sigmas"))

            # Align DinoV2 once
            if aligner is not None:
                x_dino = aligner.align(x_dino, source="DinoV2")
            _, z_d = model.encode(x_dino, source="DinoV2", sigma=None)
            fired_d = (z_d != 0).any(dim=(0, 1))  # (K,)
            fires_dino += fired_d.long()

            # Mean magnitude scoring for DinoV2 (image-level)
            scores_d = z_d.abs().mean(dim=(0, 1))       # for bag of features cosine
            scores_d_max = z_d.abs().amax(dim=(0, 1))   # for top-k rankig

            # Process every PixArt timestep
            z_p_sum = torch.zeros(1, model_tokens_eff["PixArt"], K, device=device)
            fired_p_any = torch.zeros(K, dtype=torch.bool, device=device)

            for t_idx in range(T):
                x_pix_t = x_pix_full[t_idx].unsqueeze(0)
                sigma_t = sigmas_pix.view(-1)[t_idx].to(device).float().view(1) if sigmas_pix is not None else None

                if aligner is not None:
                    x_pix_t = aligner.align(x_pix_t, source="PixArt")
                _, z_pt = model.encode(x_pix_t, source="PixArt", sigma=sigma_t)

                fired_pt = (z_pt != 0).any(dim=(0, 1))  # (K,)
                fires_per_timestep[t_idx] += fired_pt.long()
                fired_p_any |= fired_pt
                z_p_sum += z_pt.abs()

            fires_pixart += fired_p_any.long()
            cofire += (fired_d & fired_p_any).long()

            # Image-level top-K Jaccard using averaged |z|
            z_p_avg = z_p_sum / T
            scores_p_mean = z_p_avg.mean(dim=(0, 1))
            scores_d_max = z_d.abs().amax(dim=(0, 1))
            scores_p_max = z_p_avg.amax(dim=(0, 1))
            _, top_d_idx = torch.topk(scores_d_max, k=min(args.top_k, K))
            _, top_p_idx = torch.topk(scores_p_max, k=min(args.top_k, K))
            sa, sb = set(top_d_idx.cpu().tolist()), set(top_p_idx.cpu().tolist())
            u = sa | sb
            jaccard_scores.append(len(sa & sb) / len(u) if u else 0.0)

            # Bag-of-features cosine (averaged across timesteps for PixArt)
            vd = scores_d
            vp = scores_p_mean
            cos = torch.nn.functional.cosine_similarity(vd.unsqueeze(0), vp.unsqueeze(0)).item()
            image_cosines.append(cos)

            if (i + 1) % 50 == 0:
                print(f"  processed {i+1}/{n_use}")

    fires_dino_np = fires_dino.cpu().numpy()
    fires_pix_np = fires_pixart.cpu().numpy()
    cofire_np = cofire.cpu().numpy()
    jaccard_arr = np.array(jaccard_scores)
    cosine_arr = np.array(image_cosines)
    fires_per_t_np = fires_per_timestep.cpu().numpy()  # (T, K)

    used_d = fires_dino_np > 0
    used_p = fires_pix_np > 0
    n_dino_only = int((used_d & ~used_p).sum())
    n_pix_only = int((~used_d & used_p).sum())
    n_both = int((used_d & used_p).sum())
    n_neither = int((~used_d & ~used_p).sum())
    partition_score = max(n_dino_only, n_pix_only) / max(n_both, 1)

    either_count = fires_dino_np + fires_pix_np - cofire_np
    cofire_jaccard = np.where(either_count > 0, cofire_np / either_count, 0.0)
    cofire_jaccard_active = cofire_jaccard[either_count > 0]

    # Per-timestep PixArt usage breakdown
    per_t_n_used = (fires_per_t_np > 0).sum(axis=1)  # (T,) — features used at each timestep

    print()
    print("=" * 70)
    print("DICTIONARY USAGE BREAKDOWN  (PixArt aggregated over ALL timesteps)")
    print("=" * 70)
    print(f"  Total latent features (K)     : {K}")
    print(f"  Used by BOTH models           : {n_both:>6d}  ({100*n_both/K:5.1f}%)")
    print(f"  Used by DinoV2 ONLY           : {n_dino_only:>6d}  ({100*n_dino_only/K:5.1f}%)")
    print(f"  Used by PixArt ONLY           : {n_pix_only:>6d}  ({100*n_pix_only/K:5.1f}%)")
    print(f"  Dead (used by NEITHER)        : {n_neither:>6d}  ({100*n_neither/K:5.1f}%)")
    print()
    print(f"  Partition score              : {partition_score:.3f}")
    print()
    print("PIXART FEATURE USAGE PER TIMESTEP")
    print(f"  total features used (union across all timesteps): {int(used_p.sum())}")
    for t in range(fires_per_t_np.shape[0]):
        print(f"    timestep {t:2d}: {per_t_n_used[t]:>5d} features  ({100*per_t_n_used[t]/K:5.1f}%)")
    print()
    print("CO-FIRE-JACCARD PER FEATURE")
    print(f"  mean   : {cofire_jaccard_active.mean():.4f}")
    print(f"  median : {np.median(cofire_jaccard_active):.4f}")
    print()
    print("BAG-OF-FEATURES COSINE (Dino vs PixArt-mean-over-t)")
    print(f"  mean : {cosine_arr.mean():.4f}")
    print()
    print("TOP-K JACCARD")
    print(f"  mean : {jaccard_arr.mean():.4f}")
    print()
    print("=" * 70)

    np.savez(
        args.out,
        fires_dino=fires_dino_np, fires_pixart=fires_pix_np, cofire=cofire_np,
        fires_per_timestep=fires_per_t_np,
        cofire_jaccard_per_feature=cofire_jaccard,
        topk_jaccard_per_image=jaccard_arr,
        bag_of_features_cosine_per_image=cosine_arr,
        partition_score=partition_score,
        n_both=n_both, n_dino_only=n_dino_only, n_pix_only=n_pix_only, n_neither=n_neither,
        latent_dim=K,
    )
    print(f"[diag] saved arrays to: {args.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        ax = axes[0, 0]
        ax.bar(["both", "dino only", "pixart only", "neither"],
               [n_both, n_dino_only, n_pix_only, n_neither],
               color=["#2ca02c", "#1f77b4", "#ff7f0e", "#888888"], edgecolor="black")
        ax.set_ylabel("# features")
        ax.set_title(f"Usage (partition = {partition_score:.2f})")

        ax = axes[0, 1]
        ax.plot(range(len(per_t_n_used)), per_t_n_used, marker="o", linewidth=2)
        ax.set_xlabel("PixArt timestep index")
        ax.set_ylabel("# features used")
        ax.set_title("PixArt feature usage per timestep")
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.hist(cofire_jaccard_active, bins=40, color="#2ca02c", edgecolor="black", alpha=0.85)
        ax.axvline(cofire_jaccard_active.mean(), color="red", ls="--",
                   label=f"mean = {cofire_jaccard_active.mean():.3f}")
        ax.set_xlabel("Per-feature co-fire Jaccard")
        ax.set_title("Per-feature co-fire")
        ax.legend()

        ax = axes[1, 1]
        ax.hist(jaccard_arr, bins=40, color="#1f77b4", edgecolor="black", alpha=0.85)
        ax.axvline(jaccard_arr.mean(), color="red", ls="--",
                   label=f"mean = {jaccard_arr.mean():.3f}")
        ax.set_xlabel(f"Per-image top-{args.top_k} Jaccard")
        ax.set_title("Image-level top-K Jaccard")
        ax.legend()

        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"[diag] saved plot to: {args.plot}")
    except ImportError:
        print("[diag] matplotlib not available")


if __name__ == "__main__":
    main()
