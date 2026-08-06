"""Visualize SAE feature activations on image patches.

Loads one cached image, finds the top-activating SAE features, and overlays
their patch activations on the original image. If semantic segmentation masks
are available, reports which semantic classes each feature activates on.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import print_top_features
from universal_sae import UniversalSAE
from spatial_align import build_spatial_aligner_from_config, infer_grid_size
from data import CocoActivationDataset
from pixart_timestep import resolve_pixart_timestep


DEFAULT_CONFIG = "/content/algoverse_github/config.yaml"
DEFAULT_CACHE = "/content/combined_cache"
DEFAULT_WEIGHTS_DIR = "/content/algoverse_github/weights"


def find_latest_checkpoint(weights_dir: str) -> str:
    candidates = sorted(
        glob.glob(os.path.join(weights_dir, "*.pt")) +
        glob.glob(os.path.join(weights_dir, "*.pth")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No .pt/.pth checkpoints found under {weights_dir}")
    return candidates[-1]


def native_tokens_from_zoo(cfg: dict, model_names: List[str]) -> Dict[str, int]:
    # model.model_tokens gets overwritten with post-alignment counts during
    # training, so pull native counts straight from config instead.
    zoo = cfg.get("model_zoo", {})
    return {name: int(zoo[name]["num_tokens"]) for name in model_names if name in zoo}


def load_checkpoint(ckpt_path: str, cfg: dict) -> Tuple[UniversalSAE, Dict[str, int]]:
    """Returns (model, model_tokens_native).

    Also sets model._standardization_stats and model._training_global (checkpoint's
    own training-time config, config.yaml as fallback) the same way
    top_activating_images.load_universal_sae does, so callers of this loader
    (pixart_timestep_autopsy.py, pixart_attn1_sanity_check.py) that already read those
    attributes via getattr(..., default) pick up the checkpoint's real training config
    instead of silently falling back to whatever config.yaml says today.
    """
    g = cfg["global"]
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(raw, UniversalSAE):
        model = raw
        model_tokens_native = native_tokens_from_zoo(cfg, model.model_names)
        return model, model_tokens_native
    elif isinstance(raw, dict) and "state_dict" in raw:
        diffusion_models = set(raw.get("diffusion_models", g.get("diffusion_models", [])))
        ckpt_cfg = raw.get("config")
        ckpt_global = ckpt_cfg.get("global", {}) if isinstance(ckpt_cfg, dict) else {}
        ckpt_sae_params = ckpt_cfg.get("sae_params", {}) if isinstance(ckpt_cfg, dict) else {}
        model = UniversalSAE(
            model_dims=raw["model_dims"],
            latent_dim=raw["latent_dim"],
            diffusion_models=diffusion_models,
            model_tokens=raw["model_tokens"],
            top_k=int(ckpt_sae_params.get("top_k", cfg["sae_params"].get("top_k", 64))),
            cls_pool_mode=str(g.get("cls_pool_mode", "none")),
            use_tide=bool(g.get("use_tide", False)),
        )
        missing, unexpected = model.load_state_dict(raw["state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint {ckpt_path} state_dict does not match the model built from its "
                f"own config -- refusing to silently run on a partially-random encoder. "
                f"missing={missing} unexpected={unexpected}"
            )
        model_tokens_native = raw.get("model_tokens_native") or native_tokens_from_zoo(cfg, model.model_names)
        model._standardization_stats = raw.get("standardization_stats")
        # Checkpoint values win over live config.yaml, which may have moved on since
        # the run that produced this checkpoint. ckpt_global already carries
        # spatial_align_to (uni_demo.py persists the whole global block).
        model._training_global = {**g, **ckpt_global}
        return model, model_tokens_native
    else:
        raise TypeError(f"Unrecognized checkpoint at {ckpt_path}: {type(raw)}")


def discover_stems(cache_root: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(cache_root, "*_combined.npz")))
    return [os.path.basename(p)[: -len("_combined.npz")] for p in paths]


def resolve_stem(query: str, stems: List[str]) -> str:
    stem_set = set(stems)
    bare = os.path.splitext(os.path.basename(query))[0]
    if bare in stem_set:
        return bare
    if bare.isdigit():
        padded = f"{int(bare):012d}"
        if padded in stem_set:
            return padded
    raise KeyError(f"No cached stem matching {query!r}. First few available: {stems[:5]}")


def find_raw_image(stem: str, raw_image_dir: Optional[str]) -> Optional[str]:
    if not raw_image_dir:
        return None
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = os.path.join(raw_image_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def find_semantic_mask(stem: str, semantic_mask_dir: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not semantic_mask_dir:
        return None, None
    labelmap = os.path.join(semantic_mask_dir, f"{stem}_labelmap.png")
    legend = os.path.join(semantic_mask_dir, f"{stem}_legend.json")
    labelmap = labelmap if os.path.isfile(labelmap) else None
    legend = legend if os.path.isfile(legend) else None
    return labelmap, legend


@torch.no_grad()
def encode_image_tokens(
    model: UniversalSAE,
    acts: Dict[str, torch.Tensor],
    meta: dict,
    source: str,
    aligner,
    device: str,
    config_global: Optional[dict] = None,
    pixart_timestep_override: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Returns (per-feature score (K,), raw per-token latents (1,N,K), grid_size).

    config_global should be the checkpoint's own training-time global config
    (model._training_global, already merged over live config.yaml) so
    fixed_timestep_idx reflects what this checkpoint actually trained on.
    """
    if source not in acts:
        raise KeyError(f"Source {source!r} not in loaded activations {list(acts)}.")

    x = acts[source].unsqueeze(0).to(device)  # vision:(1,N,D)  diffusion:(1,T,N,D)
    sigma = None
    if source in model.diffusion_models:
        sig = meta["sigmas_by_model"].get(source, meta.get("sigmas"))
        if sig is None:
            raise ValueError(f"Diffusion source {source!r} has no sigmas in metadata.")
        t_idx = resolve_pixart_timestep(
            x.shape[1], config_global=config_global, override=pixart_timestep_override
        )
        sigma = sig.to(device).view(-1)[t_idx].view(1)
        x = x[:, t_idx]

    if aligner is not None:
        x = aligner.align(x, source=source)

    _z_pre, z = model.encode(x, source=source, sigma=sigma)  # (1, N, K)
    scores = z.abs().amax(dim=(0, 1))
    grid_size = infer_grid_size(z.shape[1])
    return scores.cpu(), z.cpu(), grid_size


def feature_heatmap(z: torch.Tensor, feature_idx: int, grid_size: int) -> np.ndarray:
    vals = z[0, :, feature_idx].abs().numpy()
    return vals.reshape(grid_size, grid_size)


def upsample_heatmap(heat: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    # Nearest-neighbor so each patch fills the exact region it covers,
    # instead of blending across patch borders.
    h_img = Image.fromarray((heat * 255.0 / max(heat.max(), 1e-12)).astype(np.uint8))
    h_img = h_img.resize((out_size[1], out_size[0]), resample=Image.NEAREST)
    return np.asarray(h_img).astype(np.float32) / 255.0


def colorize_heatmap(heat_norm: np.ndarray) -> np.ndarray:
    """Black -> red -> yellow, avoiding a matplotlib dependency."""
    h = np.clip(heat_norm, 0.0, 1.0)
    r = np.clip(h * 2.0, 0.0, 1.0)
    g = np.clip(h * 2.0 - 1.0, 0.0, 1.0)
    b = np.zeros_like(h)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap_on_image(image: Image.Image, heat_grid: np.ndarray, alpha: float = 0.55) -> Image.Image:
    w, h = image.size
    heat_px = upsample_heatmap(heat_grid, (h, w))
    color = colorize_heatmap(heat_px) * 255.0
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    blended = base * (1 - alpha * heat_px[..., None]) + color * (alpha * heat_px[..., None])
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def concept_selectivity_scores(
    z: torch.Tensor,
    grid_size: int,
    label_map: np.ndarray,
    id_to_name: Dict[int, str],
    min_percentile: float = 50.0,
) -> Tuple[np.ndarray, List[dict]]:
    """Score each feature by top-concept-mean minus second-concept-mean.

    Gated features (dead on this image, or below the image-wide magnitude
    floor) keep their entry but get margin=-inf so they drop out of ranking
    without losing the record of why.
    """
    h, w = label_map.shape
    label_small = np.asarray(
        Image.fromarray(label_map).resize((grid_size, grid_size), resample=Image.NEAREST)
    )
    labels_flat = label_small.reshape(-1)
    vals = z[0].abs().numpy()  # (N, K)
    K = vals.shape[1]

    unique_ids = np.unique(labels_flat)
    if len(unique_ids) < 2:
        return np.full(K, -np.inf), [
            {"gated": True, "gate_reason": "fewer than 2 concepts present in this image"}
            for _ in range(K)
        ]

    # Mean |activation| per concept per feature: (num_concepts, K)
    concept_means = np.stack([vals[labels_flat == cid].mean(axis=0) for cid in unique_ids])
    order = np.argsort(-concept_means, axis=0)
    feat_idx = np.arange(K)
    top1_row, top2_row = order[0], order[1]
    top1_val = concept_means[top1_row, feat_idx]
    top2_val = concept_means[top2_row, feat_idx]
    raw_margins = top1_val - top2_val

    # Threshold is pooled across all features' nonzero per-token activations,
    # not per-feature -- a per-feature percentile is a no-op since a
    # feature's top-concept mean is always one of its own higher values.
    nonzero_pool = vals[vals > 0]
    if nonzero_pool.size >= 2:
        magnitude_threshold = float(np.percentile(nonzero_pool, min_percentile))
    else:
        magnitude_threshold = float("inf")

    has_signal = (vals > 0).any(axis=0)
    eligible = has_signal & (top1_val >= magnitude_threshold)
    margins = np.where(eligible, raw_margins, -np.inf)

    concept_info = []
    for k in range(K):
        entry = {
            "top_concept": id_to_name.get(int(unique_ids[top1_row[k]]), f"class_{unique_ids[top1_row[k]]}"),
            "top_concept_mean": float(top1_val[k]),
            "second_concept": id_to_name.get(int(unique_ids[top2_row[k]]), f"class_{unique_ids[top2_row[k]]}"),
            "second_concept_mean": float(top2_val[k]),
            "margin": float(raw_margins[k]),
            "magnitude_threshold": magnitude_threshold,
            "gated": not bool(eligible[k]),
        }
        if not has_signal[k]:
            entry["gate_reason"] = "feature never fires on this image"
        elif not eligible[k]:
            entry["gate_reason"] = (
                f"top-concept mean {top1_val[k]:.4g} below image-wide p{min_percentile:.0f} "
                f"activation threshold {magnitude_threshold:.4g}"
            )
        concept_info.append(entry)

    return margins, concept_info


def make_feature_thumbnail(
    base_image: Optional[Image.Image],
    z: torch.Tensor,
    feature_idx: int,
    grid_size: int,
    thumb_size: int = 224,
) -> Image.Image:
    heat = feature_heatmap(z, feature_idx, grid_size)
    if base_image is not None:
        img = base_image.resize((thumb_size, thumb_size), resample=Image.BICUBIC)
        return overlay_heatmap_on_image(img, heat)
    heat_norm = heat / max(heat.max(), 1e-12)
    color = (colorize_heatmap(heat_norm) * 255.0).astype(np.uint8)
    return Image.fromarray(color).resize((thumb_size, thumb_size), resample=Image.NEAREST)


def build_grid_image(thumbnails: List[Tuple[str, Image.Image]], cols: int = 4, pad: int = 8) -> Image.Image:
    if not thumbnails:
        raise ValueError("No thumbnails to lay out.")
    thumb_w, thumb_h = thumbnails[0][1].size
    label_h = 24
    rows = math.ceil(len(thumbnails) / cols)
    cell_w, cell_h = thumb_w + pad, thumb_h + label_h + pad
    canvas = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), (24, 24, 24))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
    except ImportError:
        draw = None
        font = None

    for i, (label, thumb) in enumerate(thumbnails):
        row, col = divmod(i, cols)
        x = pad + col * cell_w
        y = pad + row * cell_h
        canvas.paste(thumb, (x, y))
        if draw is not None:
            draw.text((x, y + thumb_h + 2), label, fill=(255, 255, 255), font=font)
    return canvas


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", nargs="+", required=True,
                   help="One or more cache stems or image paths, e.g. "
                        "'000000562818 000000001000'. Each produces its own heatmap grid.")
    p.add_argument("--source", default="DinoV2")
    p.add_argument("--sources", nargs="+", default=["DinoV2", "PixArt"])
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--pixart_timestep", type=int, default=None,
                   help="Override the diffusion timestep index. Defaults to whatever "
                        "the checkpoint (or config.yaml as a last resort) trained on.")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--weights_dir", default=DEFAULT_WEIGHTS_DIR)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--cache_root", default=DEFAULT_CACHE)
    p.add_argument("--raw_image_dir", default=None,
                   help="Directory containing the original image for --stem. If omitted, "
                        "heatmaps render without the underlying photo.")
    p.add_argument("--semantic_mask_dir", default=None,
                   help="Directory of *_labelmap.png / *_legend.json from semantic_mask.py.")
    p.add_argument("--min_percentile", type=float, default=50.0,
                   help="Magnitude gate for concept-margin ranking: a feature's top-concept "
                        "mean must clear this percentile of the image's pooled nonzero "
                        "activations to be eligible. Only used when --semantic_mask_dir is set.")
    p.add_argument("--output_dir", default="feature_visualizations")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-feature text output; only print saved file paths.")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def visualize_stem(args, model, aligner, ds, ckpt_path, stem_query, config_global):
    """Encode one image, render its top-feature heatmaps, and save grid + report."""
    stem = resolve_stem(stem_query, ds.stems)
    if not args.quiet:
        print(f"\n{'=' * 60}\n[viz] resolved stem : {stem}")

    idx = ds.stems.index(stem)
    (acts, meta), _ = ds[idx]

    _amax_scores, z, grid_size = encode_image_tokens(
        model, acts, meta, args.source, aligner, args.device,
        config_global=config_global, pixart_timestep_override=args.pixart_timestep,
    )

    raw_image_path = find_raw_image(stem, args.raw_image_dir)
    base_image = Image.open(raw_image_path).convert("RGB") if raw_image_path else None
    if base_image is None and not args.quiet:
        print("[viz] no raw image found -- rendering heatmaps only.")

    label_map = None
    id_to_name: Dict[int, str] = {}
    labelmap_path, legend_path = find_semantic_mask(stem, args.semantic_mask_dir)
    if labelmap_path and legend_path:
        label_map = np.asarray(Image.open(labelmap_path))
        with open(legend_path) as f:
            legend = json.load(f)
        id_to_name = {int(k_): v for k_, v in legend.get("id_to_name", {}).items()}
        if not args.quiet:
            print(f"[viz] semantic mask found -> {labelmap_path}")
    elif args.semantic_mask_dir and not args.quiet:
        print(f"[viz] no semantic mask found for stem={stem!r} in {args.semantic_mask_dir}. "
              f"Run semantic_mask.py on this image first.")

    # With a semantic mask, rank by concept-margin (magnitude-gated).
    # Without one there's no concept to be selective for, so fall back
    # to raw per-token max activation.
    concept_info: Optional[List[dict]] = None
    if label_map is not None:
        margins, concept_info = concept_selectivity_scores(
            z, grid_size, label_map, id_to_name, min_percentile=args.min_percentile
        )
        scores, score_label = torch.from_numpy(margins), "concept margin"
        n_eligible = int(np.isfinite(margins).sum())
        n_gated = scores.numel() - n_eligible
        if not args.quiet and n_gated:
            print(f"[viz] {n_gated}/{scores.numel()} features gated out (dead-on-this-image or "
                  f"below their own p{args.min_percentile:.0f} magnitude threshold) -- excluded from ranking.")
        k = min(args.top_k, n_eligible)
    else:
        scores, score_label = _amax_scores, "max activation"
        k = min(args.top_k, scores.numel())

    if k == 0:
        print(f"[viz] SKIPPED {stem!r}: no eligible features (all gated) for source={args.source}.")
        return

    top_scores, top_idx = torch.topk(scores, k=k)
    top_features = [
        {"feature_idx": int(i), "score": float(s)}
        for i, s in zip(top_idx.tolist(), top_scores.tolist())
    ]

    if not args.quiet:
        print(f"[viz] top {k} features for {stem} (source={args.source}, grid={grid_size}x{grid_size}, "
              f"ranked by {score_label}):")
        print_top_features(top_features)

    os.makedirs(args.output_dir, exist_ok=True)
    thumbnails: List[Tuple[str, Image.Image]] = []
    feature_report = []

    for rank, feat in enumerate(top_features, 1):
        fi = feat["feature_idx"]
        thumb = make_feature_thumbnail(base_image, z, fi, grid_size)
        thumbnails.append((f"#{rank} feat {fi} ({feat['score']:.3f})", thumb))

        entry = {"rank": rank, "feature_idx": fi, "score": feat["score"], "score_type": score_label}
        if concept_info is not None:
            entry["concept_selectivity"] = concept_info[fi]
        feature_report.append(entry)

    grid_img = build_grid_image(thumbnails, cols=args.cols)
    grid_path = os.path.join(args.output_dir, f"{stem}_{args.source}_top{k}_features.png")
    grid_img.save(grid_path)

    json_path = os.path.join(args.output_dir, f"{stem}_{args.source}_top{k}_features.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "stem": stem,
                "source": args.source,
                "checkpoint": ckpt_path,
                "grid_size": grid_size,
                "raw_image": raw_image_path,
                "semantic_mask": labelmap_path,
                "features": feature_report,
            },
            f,
            indent=2,
        )

    print(f"[viz] saved image grid -> {grid_path}")
    print(f"[viz] saved report     -> {json_path}")

    if label_map is not None and not args.quiet:
        print("[viz] top concept per feature (margin = top concept mean - 2nd concept mean):")
        for entry in feature_report:
            sel = entry.get("concept_selectivity")
            if sel:
                print(f"  feature {entry['feature_idx']:>5}: "
                      f"{sel['top_concept']:<20} "
                      f"(top_mean={sel['top_concept_mean']:.3f}, margin={sel['margin']:.3f}, "
                      f"2nd={sel['second_concept']})")


def main():
    args = parse_args()

    ckpt_path = args.ckpt
    if not ckpt_path or not os.path.isfile(ckpt_path):
        ckpt_path = find_latest_checkpoint(args.weights_dir)
    print(f"[viz] checkpoint : {ckpt_path}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    g = cfg["global"]

    model, model_tokens_native = load_checkpoint(ckpt_path, cfg)
    model.eval().to(args.device)

    if model.cls_pool_mode != "none":
        raise ValueError(
            f"cls_pool_mode={model.cls_pool_mode!r} pools all tokens into one vector "
            "per image; per-patch visualization requires cls_pool_mode='none'."
        )

    # Prefer the checkpoint's own training-time config over live config.yaml, which may
    # have moved on since the run that produced this checkpoint (V12).
    eval_g = getattr(model, "_training_global", g)
    aligner = build_spatial_aligner_from_config(eval_g, model_tokens_native)
    if aligner is not None:
        print(f"[viz] spatial alignment ON -> target grid "
              f"{aligner.target_grid_size}x{aligner.target_grid_size}")

    ds = CocoActivationDataset(
        cache_root=args.cache_root,
        sources=args.sources,
        combined_npz=True,
        standardize=bool(eval_g.get("standardize", True)),
        return_metadata=True,
        diffusion_models=[s for s in args.sources if s in model.diffusion_models],
        use_class_tokens=False,
        standardization_stats=getattr(model, "_standardization_stats", None),
        stats_seed=eval_g.get("stats_seed", 0),
        # Only used if the checkpoint has no persisted stats and this falls back to
        # recomputing: training fits stats to the POOLED distribution
        # (REPAIR_PLAN.md V6/Fix 2.2), so the fallback has to pool too, or heatmaps
        # get rendered from inputs standardised on a different scale than training.
        spatial_aligner=aligner,
    )
    failures = []
    for stem_query in args.stem:
        try:
            visualize_stem(args, model, aligner, ds, ckpt_path, stem_query, eval_g)
        except Exception as e:
            print(f"[viz] SKIPPED {stem_query!r}: {e}")
            failures.append(stem_query)

    done = len(args.stem) - len(failures)
    print(f"\n[viz] finished {done}/{len(args.stem)} stems -> {args.output_dir}")
    if failures:
        print(f"[viz] skipped: {', '.join(map(str, failures))}")


if __name__ == "__main__":
    main()