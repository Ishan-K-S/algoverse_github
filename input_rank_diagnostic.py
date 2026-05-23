"""
input_rank_diagnostic.py

Measure the effective rank of each model's standardized input activations
(post-spatial-alignment if alignment is enabled). If PixArt's input rank is
much lower than DinoV2's, PixArt's feature collapse is upstream of the SAE
and the fix has to be in the data pipeline, not the encoder.

Effective rank metrics computed:
  - participation_ratio: (sum sigma)^2 / sum(sigma^2) — soft rank.
  - rank_90: smallest r such that top-r singular values explain 90% of variance.
  - rank_99: same at 99%.
  - condition_number: sigma_max / sigma_min_nonzero.

Usage:
    python input_rank_diagnostic.py \
        --ckpt /path/to/usae_epoch_N.pth \
        --cache /content/combined_cache \
        --config /content/algoverse_github/config.yaml \
        --n_images 500
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--repo_root", default=None)
    p.add_argument("--n_images", type=int, default=500)
    p.add_argument("--pixart_timestep", type=int, default=-1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="input_rank.npz")
    return p.parse_args()


def effective_rank(s: np.ndarray, eps: float = 1e-10):
    """Given singular values s (sorted desc), compute several rank metrics."""
    s = s[s > eps]
    if len(s) == 0:
        return dict(participation_ratio=0.0, rank_90=0, rank_99=0, condition=float("inf"))
    pr = (s.sum() ** 2) / (s ** 2).sum()
    var = (s ** 2)
    cum = np.cumsum(var) / var.sum()
    rank_90 = int(np.searchsorted(cum, 0.90) + 1)
    rank_99 = int(np.searchsorted(cum, 0.99) + 1)
    cond = float(s[0] / s[-1])
    return dict(
        participation_ratio=float(pr),
        rank_90=rank_90,
        rank_99=rank_99,
        condition=cond,
    )


def main():
    args = _parse_args()
    repo_root = args.repo_root or os.path.dirname(os.path.abspath(args.config))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from data import CocoActivationDataset
    from spatial_align import build_spatial_aligner_from_config

    print(f"[rank] loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    with open(args.config) as f:
        cfg_file = yaml.safe_load(f)
    g_file = cfg_file.get("global", {})
    model_tokens_native = ckpt.get("model_tokens_native", ckpt["model_tokens"])
    align_to = ckpt.get("spatial_align_to", g_file.get("spatial_align_to", None))
    aligner = build_spatial_aligner_from_config({"spatial_align_to": align_to}, model_tokens_native)
    if aligner is not None:
        print(f"[rank] spatial alignment ON: target grid {aligner.target_grid_size}x{aligner.target_grid_size}")
    else:
        print(f"[rank] spatial alignment OFF")

    sources = ["DinoV2", "PixArt"]
    ds = CocoActivationDataset(
        cache_root=args.cache,
        sources=sources,
        combined_npz=True,
        standardize=bool(g_file.get("standardize", True)),
        divide_norm=bool(g_file.get("divide_norm", False)),
        use_class_tokens=bool(g_file.get("use_class_tokens", False)),
        return_metadata=True,
        diffusion_models=list(set(ckpt.get("diffusion_models", g_file.get("diffusion_models", [])))),
    )
    n_use = min(args.n_images, len(ds))
    print(f"[rank] collecting activations from {n_use} images")

    # Collect post-alignment activations per model. Flatten (B*N, D) for SVD.
    bufs = {"DinoV2": [], "PixArt": []}
    for i in range(n_use):
        (acts, meta), _ = ds[i]
        x_dino = acts["DinoV2"].float()
        x_pix_full = acts["PixArt"].float()
        T = x_pix_full.shape[0]
        t_idx = args.pixart_timestep if args.pixart_timestep >= 0 else T + args.pixart_timestep
        x_pix = x_pix_full[t_idx]

        if aligner is not None:
            x_dino = aligner.align(x_dino.unsqueeze(0), source="DinoV2").squeeze(0)
            x_pix = aligner.align(x_pix.unsqueeze(0), source="PixArt").squeeze(0)

        bufs["DinoV2"].append(x_dino)   # (N, D)
        bufs["PixArt"].append(x_pix)
        if (i + 1) % 100 == 0:
            print(f"  collected {i+1}/{n_use}")

    print()
    print("=" * 70)
    print("INPUT ACTIVATION RANK PER MODEL  (post-alignment, post-standardization)")
    print("=" * 70)
    results = {}
    for name in sources:
        X = torch.cat(bufs[name], dim=0).numpy()  # (n_images * N_tokens, D)
        D = X.shape[1]
        # Center
        Xc = X - X.mean(axis=0, keepdims=True)
        # SVD on (samples, D) — singular values match those of X^T X / n
        # Use econ SVD for speed (gives min(samples, D) singular values).
        # If too big, downsample to keep computation tractable.
        max_samples = 50000
        if Xc.shape[0] > max_samples:
            idx = np.random.RandomState(0).choice(Xc.shape[0], size=max_samples, replace=False)
            Xc = Xc[idx]
        s = np.linalg.svd(Xc, compute_uv=False)
        stats = effective_rank(s)
        results[name] = dict(D=D, n_samples=Xc.shape[0], **stats)
        print()
        print(f"  {name}: D={D}, samples={Xc.shape[0]}")
        print(f"    per-feature mean abs        : {np.abs(X).mean():.4f}")
        print(f"    per-feature std             : {X.std(axis=0).mean():.4f} (expected ~1 if standardized)")
        print(f"    participation ratio         : {stats['participation_ratio']:.1f} / {D}")
        print(f"    rank_90 (90% variance)      : {stats['rank_90']} / {D}  ({100*stats['rank_90']/D:.1f}%)")
        print(f"    rank_99 (99% variance)      : {stats['rank_99']} / {D}  ({100*stats['rank_99']/D:.1f}%)")
        print(f"    top-1 singular val fraction : {(s[0]**2 / (s**2).sum()):.4f}")
        print(f"    top-10 singular val fraction: {((s[:10]**2).sum() / (s**2).sum()):.4f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    pr_d = results["DinoV2"]["participation_ratio"]
    pr_p = results["PixArt"]["participation_ratio"]
    if pr_p < 0.3 * pr_d:
        print(f"  PixArt input is ~{pr_d/pr_p:.1f}x lower-rank than DinoV2's.")
        print("  The encoder collapse is UPSTREAM of the SAE — PixArt activations")
        print("  themselves are low-rank, probably due to single-timestep training")
        print("  or standardization stats computed across mismatched timesteps.")
        print()
        print("  Fixes (in priority order):")
        print("    1. Train on RANDOM timesteps, not just the last. Set t_idx=random in")
        print("       _pick_diffusion_slice instead of total_steps-1.")
        print("    2. Recompute standardization stats USING ONLY the timestep you train on.")
        print("    3. Set use_tide=true so sigma conditioning at least varies the encoder")
        print("       (only helps if you also vary timestep).")
    elif results["PixArt"]["X_per_feature_std" if False else "rank_90"] > 0.5 * results["PixArt"]["D"]:
        print("  PixArt input has comparable rank to DinoV2 — the collapse is in the")
        print("  encoder/optimization, not the data. Fix at the encoder side:")
        print("    1. Tied/shared decoder architecture.")
        print("    2. Explicit cross-encoder alignment loss.")
        print("    3. Warm-start PixArt's encoder from DinoV2's after self-recon converges.")
    else:
        print("  Input ranks are intermediate. The collapse is likely both upstream")
        print("  AND in the encoder. Try the data-side fixes first since they're cheaper.")
    print("=" * 70)

    np.savez(args.out, **{f"{k}_{m}": v for k, d in results.items() for m, v in d.items()})
    print(f"[rank] saved to {args.out}")


if __name__ == "__main__":
    main()
