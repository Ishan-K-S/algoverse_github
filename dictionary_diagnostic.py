"""
dictionary_diagnostic.py

Standalone analysis: given a trained Universal SAE checkpoint, answer
"is the dictionary partitioned between DinoV2 and PixArt, or shared?"

Outputs (printed and saved as a .npz + .png):

  features_used_by_dino_only    : # features that fire for DinoV2 but never PixArt
  features_used_by_pixart_only  : # features that fire for PixArt but never DinoV2
  features_used_by_both         : # features that fire for at least one image of each model
  features_used_by_neither      : # dead features

  partition_score = max(dino_only, pix_only) / (both + 1)
      Big number (>1) = badly partitioned. Near zero = healthy sharing.

  cofire_jaccard_per_feature    : per-feature, fraction of images where it fires
                                  for BOTH models / fraction where it fires for
                                  EITHER model. Histogram revealing whether
                                  shared features are coincidentally shared or
                                  genuinely co-firing.

  topk_scoring_jaccard          : the exact metric your overlap script computes,
                                  for comparison. If this is ~0 but features_used_by_both
                                  is large, your SCORING metric is the problem,
                                  not the dictionary.

Usage:
    python dictionary_diagnostic.py \
        --ckpt /path/to/usae_epoch_N.pth \
        --cache /content/combined_cache \
        --config /content/algoverse_github/config.yaml \
        --n_images 200 \
        --top_k 64
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from typing import Tuple

import numpy as np
import torch
import yaml


CHECKPOINT_PATH = "/content/algoverse_github/weights/ex16_bs8_topk512_LR0.0005_alignPixArt_30ep/usae_epoch_29.pth"
CACHE_ROOT      = "/content/combined_cache"
CONFIG_PATH     = "/content/algoverse_github/config.yaml"
REPO_ROOT       = "/content/algoverse_github"
OUT_PATH        = "/content/dict_diag.npz"
PLOT_PATH       = "/content/dict_diag.png"
N_IMAGES        = 500
TOP_K           = 64
PIXART_TIMESTEP = -1
DEVICE          = "cuda"
DRIVE_SAVE_DIR  = "/content/drive/MyDrive/DictionaryDiagnosticResults"
# --------------------


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",            default=CHECKPOINT_PATH, help="Path to usae_epoch_*.pth")
    p.add_argument("--cache",           default=CACHE_ROOT,      help="Path to combined activation cache dir")
    p.add_argument("--config",          default=CONFIG_PATH,     help="Path to config.yaml")
    p.add_argument("--repo_root",       default=REPO_ROOT,
                   help="Path to the algoverse repo (added to sys.path). Default: dirname(config).")
    p.add_argument("--n_images",        type=int,   default=N_IMAGES,        help="How many images to analyze")
    p.add_argument("--top_k",           type=int,   default=TOP_K,           help="K for the Jaccard scoring comparison")
    p.add_argument("--pixart_timestep", type=int,   default=PIXART_TIMESTEP, help="Which diffusion timestep to use")
    p.add_argument("--device",                      default=DEVICE)
    p.add_argument("--out",                         default=OUT_PATH,        help="Where to save raw arrays")
    p.add_argument("--plot",                        default=PLOT_PATH,       help="Where to save the diagnostic plot")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Drive helpers
# -----------------------------------------------------------------------------

def _mount_drive(retries: int = 3) -> bool:
    try:
        from google.colab import drive as _colab_drive
    except ImportError:
        return False
    for attempt in range(retries):
        try:
            _colab_drive.mount("/content/drive", force_remount=(attempt > 0))
            if os.path.isdir("/content/drive/MyDrive"):
                return True
        except Exception as e:
            print(f"[drive] Mount attempt {attempt + 1} failed: {e}")
        time.sleep(5)
    return False


def _robust_copy(src: str, dst: str, retries: int = 3) -> None:
    for attempt in range(retries):
        try:
            with open(src, "rb") as f:
                data = f.read()
            with open(dst, "wb") as f:
                f.write(data)
            return
        except OSError as e:
            if attempt < retries - 1:
                print(f"[drive] OSError on {os.path.basename(src)} (errno {e.errno}), remounting...")
                _mount_drive()
                time.sleep(5)
            else:
                raise


def save_to_drive(file_paths: list, drive_dir: str) -> None:
    if not _mount_drive():
        print("[drive] Not running in Colab — skipping Drive upload.")
        return
    os.makedirs(drive_dir, exist_ok=True)
    for path in file_paths:
        if not os.path.isfile(path):
            print(f"[drive] Skipping missing file: {path}")
            continue
        dst = os.path.join(drive_dir, os.path.basename(path))
        _robust_copy(path, dst)
        print(f"[drive] Saved: {dst}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = _parse_args()

    repo_root = args.repo_root or os.path.dirname(os.path.abspath(args.config))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from data import CocoActivationDataset
    from universal_sae import UniversalSAE
    from spatial_align import build_spatial_aligner_from_config

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    # ----- Load checkpoint and rebuild the model -----
    print(f"[diag] loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    with open(args.config) as f:
        cfg_file = yaml.safe_load(f)
    g_file = cfg_file.get("global", {})
    sae_p = cfg_file.get("sae_params", {})
    g_ckpt = (ckpt.get("config") or {}).get("global", {}) if isinstance(ckpt.get("config"), dict) else {}

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
        top_k=int(sae_p.get("top_k", pick("top_k", 256))),
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
    else:
        print(f"[diag] spatial alignment OFF")

    # ----- Dataset -----
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
    )
    n_use = min(args.n_images, len(ds))
    print(f"[diag] analyzing {n_use}/{len(ds)} images")

    K = model.latent_dim
    print(f"[diag] latent_dim K = {K}, top_k = {model.top_k}")

    # ----- Pass 1: per-image, per-feature firing pattern for each model -----
    # `fires_dino[k]` = number of images where feature k fires for DinoV2 at all
    # `fires_pixart[k]` = same for PixArt
    # `cofire[k]` = number of images where feature k fires for BOTH
    fires_dino = torch.zeros(K, dtype=torch.long, device=device)
    fires_pixart = torch.zeros(K, dtype=torch.long, device=device)
    cofire = torch.zeros(K, dtype=torch.long, device=device)

    # Top-k jaccard accumulators (matches your cross_model_overlap.py exactly)
    jaccard_scores = []
    count_dino_topk, count_pix_topk, count_both_topk = Counter(), Counter(), Counter()

    # Per-image bag-of-features cosine, computed at the image level (averaged over tokens)
    image_cosines = []

    with torch.no_grad():
        for i in range(n_use):
            (acts, meta), _ = ds[i]
            x_dino = acts["DinoV2"].to(device).unsqueeze(0).float()
            x_pix_full = acts["PixArt"].to(device).float()

            # Pick the same timestep as eval
            T = x_pix_full.shape[0]
            t_idx = args.pixart_timestep if args.pixart_timestep >= 0 else T + args.pixart_timestep
            x_pix = x_pix_full[t_idx].unsqueeze(0)

            sig_map = meta.get("sigmas_by_model", {})
            sigmas_pix = sig_map.get("PixArt", meta.get("sigmas"))
            sigma_pix = sigmas_pix.view(-1)[t_idx].to(device).float().view(1) if sigmas_pix is not None else None

            # Spatial align
            if aligner is not None:
                x_dino = aligner.align(x_dino, source="DinoV2")
                x_pix = aligner.align(x_pix, source="PixArt")

            # Encode both
            _, z_d = model.encode(x_dino, source="DinoV2", sigma=None)
            _, z_p = model.encode(x_pix, source="PixArt", sigma=sigma_pix)

            # Per-feature firing for this image (did feature k fire on ANY token?)
            fired_d = (z_d != 0).any(dim=(0, 1))  # (K,)
            fired_p = (z_p != 0).any(dim=(0, 1))  # (K,)

            fires_dino += fired_d.long()
            fires_pixart += fired_p.long()
            cofire += (fired_d & fired_p).long()

            # Top-K scoring metric (matches cross_model_overlap.py)
            scores_d = z_d.abs().mean(dim=(0, 1))
            scores_p = z_p.abs().mean(dim=(0, 1))
            _, top_d_idx = torch.topk(scores_d, k=min(args.top_k, K))
            _, top_p_idx = torch.topk(scores_p, k=min(args.top_k, K))
            sa, sb = set(top_d_idx.cpu().tolist()), set(top_p_idx.cpu().tolist())
            u = sa | sb
            jaccard_scores.append(len(sa & sb) / len(u) if u else 0.0)
            count_dino_topk.update(sa)
            count_pix_topk.update(sb)
            count_both_topk.update(sa & sb)

            # Bag-of-features cosine: are the two models lighting up similar feature mixes?
            # Average |z| across tokens to get a (K,) "feature usage" per image per model.
            vd = z_d.abs().mean(dim=(0, 1))
            vp = z_p.abs().mean(dim=(0, 1))
            cos = torch.nn.functional.cosine_similarity(vd.unsqueeze(0), vp.unsqueeze(0)).item()
            image_cosines.append(cos)

            if (i + 1) % 50 == 0:
                print(f"  processed {i+1}/{n_use}")

    fires_dino_np = fires_dino.cpu().numpy()
    fires_pix_np = fires_pixart.cpu().numpy()
    cofire_np = cofire.cpu().numpy()
    jaccard_arr = np.array(jaccard_scores)
    cosine_arr = np.array(image_cosines)

    # ----- Aggregate stats -----
    used_d = fires_dino_np > 0
    used_p = fires_pix_np > 0
    n_dino_only = int((used_d & ~used_p).sum())
    n_pix_only = int((~used_d & used_p).sum())
    n_both = int((used_d & used_p).sum())
    n_neither = int((~used_d & ~used_p).sum())
    partition_score = max(n_dino_only, n_pix_only) / max(n_both, 1)

    # Per-feature co-fire-Jaccard (only over features that fired at least once for either model)
    either_count = fires_dino_np + fires_pix_np - cofire_np
    cofire_jaccard = np.where(either_count > 0, cofire_np / either_count, 0.0)
    cofire_jaccard_active = cofire_jaccard[either_count > 0]

    print()
    print("=" * 70)
    print("DICTIONARY USAGE BREAKDOWN")
    print("=" * 70)
    print(f"  Total latent features (K)     : {K}")
    print(f"  Used by BOTH models           : {n_both:>6d}  ({100*n_both/K:5.1f}%)")
    print(f"  Used by DinoV2 ONLY           : {n_dino_only:>6d}  ({100*n_dino_only/K:5.1f}%)")
    print(f"  Used by PixArt ONLY           : {n_pix_only:>6d}  ({100*n_pix_only/K:5.1f}%)")
    print(f"  Dead (used by NEITHER)        : {n_neither:>6d}  ({100*n_neither/K:5.1f}%)")
    print()
    print(f"  Partition score              : {partition_score:.3f}")
    print(f"    (max model-exclusive / both. <0.3 healthy, >1.0 badly partitioned)")
    print()
    print("CO-FIRE-JACCARD PER FEATURE (over active features only)")
    print(f"  mean   : {cofire_jaccard_active.mean():.4f}")
    print(f"  median : {np.median(cofire_jaccard_active):.4f}")
    print(f"  p25    : {np.percentile(cofire_jaccard_active, 25):.4f}")
    print(f"  p75    : {np.percentile(cofire_jaccard_active, 75):.4f}")
    print()
    print("PER-IMAGE BAG-OF-FEATURES COSINE (Dino z vs PixArt z, averaged across tokens)")
    print(f"  mean   : {cosine_arr.mean():.4f}")
    print(f"  median : {np.median(cosine_arr):.4f}")
    print(f"  (0 = orthogonal feature usage, 1 = identical mix)")
    print()
    print("TOP-K JACCARD (matches your cross_model_overlap.py scoring)")
    print(f"  mean   : {jaccard_arr.mean():.4f}")
    print(f"  median : {np.median(jaccard_arr):.4f}")
    print(f"  fraction of images with Jaccard == 0 : {(jaccard_arr == 0).mean():.3f}")
    print()
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    if n_both / max(K, 1) < 0.2 and partition_score > 1.0:
        print("  PARTITION CONFIRMED: each model uses largely disjoint feature subsets.")
        print("  Cross-encoder alignment loss or shared-decoder architecture is required.")
    elif n_both / max(K, 1) > 0.6 and jaccard_arr.mean() < 0.05:
        print("  DICTIONARY IS SHARED but top-K Jaccard is still ~0.")
        print("  The problem is your SCORING metric — features that fire for both")
        print("  models aren't ranking the same way under z.abs().mean() scoring.")
        print("  Try ranking by total energy (sum |z|) or co-firing frequency instead.")
    elif n_both / max(K, 1) > 0.6 and jaccard_arr.mean() > 0.15:
        print("  HEALTHY: dictionary is shared and top-K Jaccard is meaningful.")
    else:
        print("  MIXED SIGNAL — see numbers above for the actual situation.")
    print("=" * 70)

    # ----- Save raw arrays -----
    np.savez(
        args.out,
        fires_dino=fires_dino_np,
        fires_pixart=fires_pix_np,
        cofire=cofire_np,
        cofire_jaccard_per_feature=cofire_jaccard,
        topk_jaccard_per_image=jaccard_arr,
        bag_of_features_cosine_per_image=cosine_arr,
        partition_score=partition_score,
        n_both=n_both, n_dino_only=n_dino_only, n_pix_only=n_pix_only, n_neither=n_neither,
        latent_dim=K,
    )
    print(f"[diag] saved arrays to: {args.out}")

    # ----- Diagnostic plot -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # (1) Bar: usage breakdown
        ax = axes[0, 0]
        labels = ["both", "dino only", "pixart only", "neither"]
        vals = [n_both, n_dino_only, n_pix_only, n_neither]
        colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#888888"]
        ax.bar(labels, vals, color=colors, edgecolor="black")
        ax.set_ylabel("# features (of K)")
        ax.set_title(f"Dictionary usage (partition score = {partition_score:.2f})")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{100*v/K:.0f}%", ha="center", va="bottom")

        # (2) Hist: co-fire-Jaccard per active feature
        ax = axes[0, 1]
        ax.hist(cofire_jaccard_active, bins=40, color="#2ca02c", edgecolor="black", alpha=0.85)
        ax.axvline(cofire_jaccard_active.mean(), color="red", ls="--",
                   label=f"mean = {cofire_jaccard_active.mean():.3f}")
        ax.set_xlabel("Per-feature co-fire Jaccard")
        ax.set_ylabel("# features")
        ax.set_title("How often shared features actually co-fire")
        ax.legend()

        # (3) Hist: top-K Jaccard per image (matches your script)
        ax = axes[1, 0]
        ax.hist(jaccard_arr, bins=40, color="#1f77b4", edgecolor="black", alpha=0.85)
        ax.axvline(jaccard_arr.mean(), color="red", ls="--",
                   label=f"mean = {jaccard_arr.mean():.3f}")
        ax.set_xlabel(f"Per-image top-{args.top_k} Jaccard")
        ax.set_ylabel("# images")
        ax.set_title("What your cross_model_overlap.py is measuring")
        ax.legend()

        # (4) Hist: bag-of-features cosine
        ax = axes[1, 1]
        ax.hist(cosine_arr, bins=40, color="#ff7f0e", edgecolor="black", alpha=0.85)
        ax.axvline(cosine_arr.mean(), color="red", ls="--",
                   label=f"mean = {cosine_arr.mean():.3f}")
        ax.set_xlabel("Bag-of-features cosine (Dino vs PixArt, per image)")
        ax.set_ylabel("# images")
        ax.set_title("Continuous feature-usage alignment")
        ax.legend()

        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"[diag] saved plot to: {args.plot}")
    except ImportError:
        print("[diag] matplotlib not available, skipping plot")

    save_to_drive([args.out, args.plot], DRIVE_SAVE_DIR)


if __name__ == "__main__":
    main()
