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


DEFAULT_CONFIG = "/content/algoverse_github/config.yaml"
DEFAULT_CACHE = "/content/combined_cache"
DEFAULT_WEIGHTS_DIR = "/content/weights"


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
    """Pre-alignment token counts per model, from config.yaml's model_zoo.

    model.model_tokens is mutated in place to post-alignment counts during
    training (see train.py), so it can't be used to recover native grid sizes.
    """
    zoo = cfg.get("model_zoo", {})
    return {name: int(zoo[name]["num_tokens"]) for name in model_names if name in zoo}


def load_checkpoint(ckpt_path: str, cfg: dict) -> Tuple[UniversalSAE, Dict[str, int]]:
    g = cfg["global"]
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(raw, UniversalSAE):
        model = raw
        model_tokens_native = native_tokens_from_zoo(cfg, model.model_names)
    elif isinstance(raw, dict) and "state_dict" in raw:
        diffusion_models = set(raw.get("diffusion_models", g.get("diffusion_models", [])))
        model = UniversalSAE(
            model_dims=raw["model_dims"],
            latent_dim=raw["latent_dim"],
            diffusion_models=diffusion_models,
            model_tokens=raw["model_tokens"],
            top_k=int(cfg["sae_params"].get("top_k", 64)),
            cls_pool_mode=str(g.get("cls_pool_mode", "none")),
            use_tide=bool(g.get("use_tide", False)),
        )
        model.load_state_dict(raw["state_dict"], strict=False)
        model_tokens_native = raw.get("model_tokens_native") or native_tokens_from_zoo(cfg, model.model_names)
    else:
        raise TypeError(f"Unrecognized checkpoint at {ckpt_path}: {type(raw)}")

    return model, model_tokens_native


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
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Returns (per-feature score (K,), raw per-token latents (1,N,K), grid_size)."""
    if source not in acts:
        raise KeyError(f"Source {source!r} not in loaded activations {list(acts)}.")

    x = acts[source].unsqueeze(0).to(device)  # vision:(1,N,D)  diffusion:(1,T,N,D)
    sigma = None
    if source in model.diffusion_models:
        sig = meta["sigmas_by_model"].get(source, meta.get("sigmas"))
        if sig is None:
            raise ValueError(f"Diffusion source {source!r} has no sigmas in metadata.")
        t_idx = x.shape[1] - 1  # final timestep, matches convention elsewhere in repo
        sigma = sig.to(device).view(-1)[t_idx].view(1)
        x = x[:, t_idx]

    if aligner is not None:
        x = aligner.align(x, source=source)

    _z_pre, z = model.encode(x, source=source, sigma=sigma)  # (1, N, K)
    scores = z.abs().mean(dim=(0, 1))
    grid_size = infer_grid_size(z.shape[1])
    return scores.cpu(), z.cpu(), grid_size


def feature_heatmap(z: torch.Tensor, feature_idx: int, grid_size: int) -> np.ndarray:
    vals = z[0, :, feature_idx].abs().numpy()
    return vals.reshape(grid_size, grid_size)


def upsample_heatmap(heat: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """out_size is (height, width). Nearest-neighbor so each patch fills the
    exact image region it covers, rather than blending across patch borders."""
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


def semantic_labels_for_heatmap(
    heat_grid: np.ndarray,
    label_map: np.ndarray,
    id_to_name: Dict[int, str],
    top_fraction: float = 0.2,
) -> List[dict]:
    """Class breakdown among the hottest top_fraction of pixels for one feature."""
    h, w = label_map.shape
    heat_px = upsample_heatmap(heat_grid, (h, w))
    flat_heat = heat_px.flatten()
    flat_labels = label_map.flatten()

    n_top = max(1, int(len(flat_heat) * top_fraction))
    top_idx = np.argpartition(flat_heat, -n_top)[-n_top:]
    top_labels = flat_labels[top_idx]

    ids, counts = np.unique(top_labels, return_counts=True)
    order = np.argsort(-counts)
    rows = []
    for i in order:
        cls_id = int(ids[i])
        rows.append({
            "class_id": cls_id,
            "class_name": id_to_name.get(cls_id, f"class_{cls_id}"),
            "fraction_of_hot_patches": float(counts[i]) / n_top,
        })
    return rows


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
    p.add_argument("--stem", required=True, help="Cache stem or image path/filename, e.g. '000000562818'.")
    p.add_argument("--source", default="DinoV2")
    p.add_argument("--sources", nargs="+", default=["DinoV2", "PixArt"])
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--weights_dir", default=DEFAULT_WEIGHTS_DIR)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--cache_root", default=DEFAULT_CACHE)
    p.add_argument("--raw_image_dir", default=None,
                   help="Directory containing the original image for --stem. If omitted, "
                        "heatmaps render without the underlying photo.")
    p.add_argument("--semantic_mask_dir", default=None,
                   help="Directory of *_labelmap.png / *_legend.json from semantic_mask.py.")
    p.add_argument("--output_dir", default="feature_visualizations")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


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

    aligner = build_spatial_aligner_from_config(g, model_tokens_native)
    if aligner is not None:
        print(f"[viz] spatial alignment ON -> target grid "
              f"{aligner.target_grid_size}x{aligner.target_grid_size}")

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
    print(f"[viz] resolved stem : {stem}")

    idx = ds.stems.index(stem)
    (acts, meta), _ = ds[idx]

    scores, z, grid_size = encode_image_tokens(model, acts, meta, args.source, aligner, args.device)
    k = min(args.top_k, scores.numel())
    top_scores, top_idx = torch.topk(scores, k=k)
    top_features = [
        {"feature_idx": int(i), "score": float(s)}
        for i, s in zip(top_idx.tolist(), top_scores.tolist())
    ]

    print(f"\n[viz] top {k} features for {stem} (source={args.source}, grid={grid_size}x{grid_size}):")
    print_top_features(top_features)

    raw_image_path = find_raw_image(stem, args.raw_image_dir)
    base_image = Image.open(raw_image_path).convert("RGB") if raw_image_path else None
    if base_image is None:
        print("[viz] no raw image found -- rendering heatmaps only.")

    label_map = None
    id_to_name: Dict[int, str] = {}
    labelmap_path, legend_path = find_semantic_mask(stem, args.semantic_mask_dir)
    if labelmap_path and legend_path:
        label_map = np.asarray(Image.open(labelmap_path))
        with open(legend_path) as f:
            legend = json.load(f)
        id_to_name = {int(k_): v for k_, v in legend.get("id_to_name", {}).items()}
        print(f"[viz] semantic mask found -> {labelmap_path}")
    elif args.semantic_mask_dir:
        print(f"[viz] no semantic mask found for stem={stem!r} in {args.semantic_mask_dir}. "
              f"Run semantic_mask.py on this image first.")

    os.makedirs(args.output_dir, exist_ok=True)
    thumbnails: List[Tuple[str, Image.Image]] = []
    feature_report = []

    for rank, feat in enumerate(top_features, 1):
        fi = feat["feature_idx"]
        thumb = make_feature_thumbnail(base_image, z, fi, grid_size)
        thumbnails.append((f"#{rank} feat {fi} ({feat['score']:.3f})", thumb))

        entry = {"rank": rank, "feature_idx": fi, "score": feat["score"]}
        if label_map is not None:
            heat = feature_heatmap(z, fi, grid_size)
            entry["semantic_classes"] = semantic_labels_for_heatmap(heat, label_map, id_to_name)
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

    print(f"\n[viz] saved image grid -> {grid_path}")
    print(f"[viz] saved report     -> {json_path}")

    if label_map is not None:
        print("\n[viz] top semantic class per feature:")
        for entry in feature_report:
            classes = entry.get("semantic_classes", [])
            if classes:
                top_cls = classes[0]
                print(f"  feature {entry['feature_idx']:>5}: "
                      f"{top_cls['class_name']:<20} "
                      f"({top_cls['fraction_of_hot_patches']*100:.0f}% of hot patches)")


if __name__ == "__main__":
    main()