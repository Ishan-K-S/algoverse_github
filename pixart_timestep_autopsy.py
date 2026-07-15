"""PixArt timestep autopsy -- diagnose why PixArt heatmaps look uniform.

For ONE image this sweeps every cached PixArt timestep, encodes each through the
SAE, and measures how *localized* the resulting features are. It answers three
questions in one run:

  1. Is the uniform heatmap a WRONG-TIMESTEP artifact? -> some timestep will
     localize while others (near-pure-noise) stay flat.
  2. Is it a FEATURE-SELECTION artifact? -> ranking features by localization
     (peakiness) instead of by mean activation rescues sharp heatmaps.
  3. Is it a REAL encoding failure? -> every timestep, every ranking stays flat,
     i.e. PixArt latents genuinely carry no spatial structure through the SAE.

DinoV2 (which you confirmed localizes) is encoded as a reference so every number
has a "this is what good looks like" baseline.

Localization metric per feature: peakiness = max_token(|z|) / mean_token(|z|).
Uniform feature -> ~1.0.  Sharply localized feature -> large.

Outputs (into --output_dir):
  autopsy_<stem>.json                         - all per-timestep stats + verdict
  <stem>_DinoV2_reference.png                 - reference grid (localized target)
  <stem>_PixArt_t{idx}_localized.png          - per-timestep grid, localization-ranked
  <stem>_PixArt_t{best}_meanranked.png        - best timestep, mean-ranked (what the
                                                current viz shows) for contrast

Example:
    python pixart_timestep_autopsy.py --stem 000000562818 \
        --raw_image_dir /content/coco_data/val2017
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spatial_align import build_spatial_aligner_from_config, infer_grid_size
from data import CocoActivationDataset
from visualize_feature_activations import (
    DEFAULT_CACHE,
    DEFAULT_CONFIG,
    DEFAULT_WEIGHTS_DIR,
    find_latest_checkpoint,
    find_raw_image,
    load_checkpoint,
    make_feature_thumbnail,
    resolve_stem,
)

EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", required=True, help="One cache stem or image path, e.g. '000000562818'.")
    p.add_argument("--source", default="PixArt", help="Diffusion source to autopsy.")
    p.add_argument("--ref_source", default="DinoV2", help="Vision source used as localization baseline.")
    p.add_argument("--sources", nargs="+", default=["DinoV2", "PixArt"])
    p.add_argument("--top_k", type=int, default=8, help="Features per heatmap grid.")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--weights_dir", default=DEFAULT_WEIGHTS_DIR)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--cache_root", default=DEFAULT_CACHE)
    p.add_argument("--raw_image_dir", default=None)
    p.add_argument("--output_dir", default="pixart_timestep_autopsy")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def encode_slice(model, x_slice, source, sigma, aligner, device):
    """x_slice: (1, N, D) single-timestep activation -> latents z (1, N, K) on CPU."""
    x = x_slice.to(device)
    if aligner is not None:
        x = aligner.align(x, source=source)
    _z_pre, z = model.encode(x, source=source, sigma=sigma)
    return z.cpu()


def localization(z: torch.Tensor):
    """Per-feature stats from latents z (1, N, K).

    Returns dict with:
      mean_score : (K,) mean |z| over tokens  -- how the current viz ranks
      peak       : (K,) max/mean over tokens  -- localization (uniform ~1)
      active     : (K,) bool, features with non-trivial activation
    """
    absz = z[0].abs()                      # (N, K)
    mean_score = absz.mean(dim=0)          # (K,)
    max_score = absz.max(dim=0).values     # (K,)
    peak = max_score / (mean_score + EPS)  # (K,)
    active = mean_score > (mean_score.max() * 1e-3 + EPS)
    peak = torch.where(active, peak, torch.zeros_like(peak))
    return {"mean_score": mean_score, "peak": peak, "active": active}


def peakiness_of_topk(stats: dict, k: int, by: str) -> Tuple[float, List[int]]:
    """Mean peakiness of the top-k features, selected either 'mean' or 'localized'."""
    key = "mean_score" if by == "mean" else "peak"
    scores = stats[key]
    k = min(k, int(stats["active"].sum().item()) or 1)
    idx = torch.topk(scores, k=k).indices
    return float(stats["peak"][idx].mean().item()), idx.tolist()


def save_grid(base_image, z, feature_ids, grid_size, path, top_k):
    from visualize_feature_activations import build_grid_image
    thumbs = []
    for rank, fi in enumerate(feature_ids[:top_k], 1):
        thumbs.append((f"#{rank} feat {fi}", make_feature_thumbnail(base_image, z, fi, grid_size)))
    build_grid_image(thumbs, cols=min(4, len(thumbs))).save(path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = args.ckpt
    if not ckpt_path or not os.path.isfile(ckpt_path):
        ckpt_path = find_latest_checkpoint(args.weights_dir)
    print(f"[autopsy] checkpoint : {ckpt_path}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    g = cfg["global"]

    model, model_tokens_native = load_checkpoint(ckpt_path, cfg)
    model.eval().to(args.device)
    aligner = build_spatial_aligner_from_config(g, model_tokens_native)

    ds = CocoActivationDataset(
        cache_root=args.cache_root,
        sources=args.sources,
        combined_npz=True,
        standardize=bool(g.get("standardize", True)),
        return_metadata=True,
        diffusion_models=[s for s in args.sources if s in model.diffusion_models],
        use_class_tokens=False,
    )
    stem = resolve_stem(args.stem, ds.stems)
    (acts, meta), _ = ds[ds.stems.index(stem)]
    print(f"[autopsy] stem : {stem}")

    raw_path = find_raw_image(stem, args.raw_image_dir)
    base_image = Image.open(raw_path).convert("RGB") if raw_path else None
    if base_image is None:
        print("[autopsy] no raw image -- rendering heatmaps on black.")

    # ---- Reference: DinoV2 (single timestep, known to localize) ----
    ref = args.ref_source
    if ref not in acts:
        raise KeyError(f"ref_source {ref!r} not in {list(acts)}")
    z_ref = encode_slice(model, acts[ref].unsqueeze(0), ref, None, aligner, args.device)
    ref_grid = infer_grid_size(z_ref.shape[1])
    ref_stats = localization(z_ref)
    ref_peak, ref_ids = peakiness_of_topk(ref_stats, args.top_k, by="mean")
    save_grid(base_image, z_ref, ref_ids, ref_grid,
              os.path.join(args.output_dir, f"{stem}_{ref}_reference.png"), args.top_k)
    print(f"\n[autopsy] REFERENCE {ref}: top-k peakiness = {ref_peak:.2f} "
          f"(this is what 'localized' looks like)\n")

    # ---- PixArt: sweep every timestep ----
    src = args.source
    if src not in acts:
        raise KeyError(f"source {src!r} not in {list(acts)}")
    a = acts[src]
    if a.dim() != 3:
        raise ValueError(f"Expected diffusion activations (T,N,D) for {src!r}, got {tuple(a.shape)}. "
                         f"Is {src} actually a diffusion model in this cache?")
    T = a.shape[0]
    sig = meta["sigmas_by_model"].get(src, meta.get("sigmas"))
    sig_flat = sig.to(args.device).view(-1) if sig is not None else None
    print(f"[autopsy] {src} activations shape = {tuple(a.shape)}  (T={T} timesteps)\n")

    header = f"{'t':>3} {'sigma':>10} {'n_active':>9} {'peak(mean-rank)':>16} {'peak(localized)':>16}"
    print(header)
    print("-" * len(header))

    rows = []
    best = {"peak_localized": -1.0, "t": None}
    for t in range(T):
        sigma_t = sig_flat[t].view(1) if sig_flat is not None else None
        z_t = encode_slice(model, a[t].unsqueeze(0), src, sigma_t, aligner, args.device)
        grid = infer_grid_size(z_t.shape[1])
        stats = localization(z_t)
        peak_mean, mean_ids = peakiness_of_topk(stats, args.top_k, by="mean")
        peak_loc, loc_ids = peakiness_of_topk(stats, args.top_k, by="localized")
        n_active = int(stats["active"].sum().item())
        sigma_val = float(sigma_t.item()) if sigma_t is not None else float("nan")

        print(f"{t:>3} {sigma_val:>10.3f} {n_active:>9} {peak_mean:>16.2f} {peak_loc:>16.2f}")
        save_grid(base_image, z_t, loc_ids, grid,
                  os.path.join(args.output_dir, f"{stem}_{src}_t{t}_localized.png"), args.top_k)

        rows.append({
            "timestep": t, "sigma": sigma_val, "n_active": n_active,
            "peakiness_mean_ranked": peak_mean, "peakiness_localized": peak_loc,
            "top_mean_features": mean_ids, "top_localized_features": loc_ids,
        })
        if peak_loc > best["peak_localized"]:
            best = {"peak_localized": peak_loc, "peak_mean": peak_mean, "t": t,
                    "sigma": sigma_val, "loc_ids": loc_ids, "z": z_t, "grid": grid}

    # Contrast grid at the best timestep: mean-ranked (what the viz shows today).
    bt = best["t"]
    best_mean_ids = rows[bt]["top_mean_features"]
    save_grid(base_image, best["z"], best_mean_ids, best["grid"],
              os.path.join(args.output_dir, f"{stem}_{src}_t{bt}_meanranked.png"), args.top_k)

    # ---- Verdict ----
    ref_thresh = 0.5 * ref_peak  # "localized" = at least half the reference peakiness
    best_loc = best["peak_localized"]
    best_mean = best["peak_mean"]
    print("\n" + "=" * 60)
    print(f"[verdict] reference ({ref}) peakiness   : {ref_peak:.2f}")
    print(f"[verdict] best {src} timestep           : t={bt} (sigma={best['sigma']:.3f})")
    print(f"[verdict]   peakiness, localized-ranked : {best_loc:.2f}")
    print(f"[verdict]   peakiness, mean-ranked      : {best_mean:.2f}")

    if best_loc < ref_thresh:
        verdict = ("REAL encoding failure: even the best timestep with the best feature "
                   "selection stays flat. PixArt latents carry little spatial structure "
                   "through the SAE -> this is an alignment/training problem, not a viz bug.")
    elif best_mean < ref_thresh <= best_loc:
        verdict = (f"FEATURE-SELECTION artifact: localized-ranked features are sharp "
                   f"({best_loc:.2f}) but mean-ranked ones (what the viz shows) are flat "
                   f"({best_mean:.2f}). Fix the viz to rank by localization; PixArt is fine.")
    else:
        verdict = (f"WRONG-TIMESTEP artifact: timestep {bt} localizes well "
                   f"({best_loc:.2f}) while others are flat. Keep timestep {bt} "
                   f"(sigma={best['sigma']:.3f}) when collapsing PixArt to one timestep.")
    print(f"\n[verdict] {verdict}")
    print("=" * 60)

    with open(os.path.join(args.output_dir, f"autopsy_{stem}.json"), "w") as f:
        json.dump({
            "stem": stem, "checkpoint": ckpt_path, "source": src, "ref_source": ref,
            "T": T, "reference_peakiness": ref_peak,
            "best_timestep": bt, "best_sigma": best["sigma"],
            "verdict": verdict, "per_timestep": rows,
        }, f, indent=2)
    print(f"\n[autopsy] saved -> {args.output_dir}/  (grids + autopsy_{stem}.json)")


if __name__ == "__main__":
    main()
