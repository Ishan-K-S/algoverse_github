"""Probe SAE feature sensitivity by masking cached image patch activations.

This operates on the combined activation cache, not raw pixels. For a chosen
image/cache stem it masks selected patch tokens before UniversalSAE.encode()
and reports how feature scores change.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from top_activating_images import (
    CACHE_ROOT,
    CHECKPOINT_PATH,
    COCO_ANNOTATIONS_DIR,
    COCO_SPLIT,
    CONFIG_PATH,
    FeaturePoolMode,
    coco_annotation_path,
    load_activation_for_image,
    load_universal_sae,
    pool_feature_scores,
)


MaskFillMode = str
Box = Tuple[float, float, float, float]
RGB = Tuple[int, int, int]


def infer_grid_size(n_tokens: int) -> int:
    side = int(round(math.sqrt(n_tokens)))
    if side * side != n_tokens:
        raise ValueError(
            f"Cannot infer patch grid from {n_tokens} tokens. "
            "Use --use_cls only if the model was trained with CLS tokens, otherwise "
            "strip special tokens before probing."
        )
    return side


def parse_feature_ids(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or not raw.strip():
        return None
    feature_ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            feature_ids.extend(range(int(start), int(end) + 1))
        else:
            feature_ids.append(int(part))
    return sorted(set(feature_ids))


def parse_int_ranges(raw: Optional[str], field_name: str) -> List[int]:
    values = parse_feature_ids(raw)
    if values is None:
        raise ValueError(f"{field_name} is required.")
    return values


def parse_rgb_values(raw_values: Sequence[str]) -> List[RGB]:
    colors: List[RGB] = []
    for raw in raw_values:
        for part in raw.split(";"):
            part = part.strip()
            if not part:
                continue
            values = [int(v.strip()) for v in part.split(",")]
            if len(values) != 3:
                raise ValueError(f"RGB mask value must be r,g,b, got {part!r}")
            if any(v < 0 or v > 255 for v in values):
                raise ValueError(f"RGB values must be in [0, 255], got {part!r}")
            colors.append((values[0], values[1], values[2]))
    return colors


def parse_patch_indices(raw: Optional[str], grid_size: int) -> Set[int]:
    if raw is None or not raw.strip():
        return set()

    indices: Set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            row_s, col_s = part.split(":", 1)
            row, col = int(row_s), int(col_s)
            if not (0 <= row < grid_size and 0 <= col < grid_size):
                raise ValueError(f"Patch {part!r} is outside {grid_size}x{grid_size}.")
            indices.add(row * grid_size + col)
        elif "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))

    max_idx = grid_size * grid_size - 1
    bad = [idx for idx in indices if idx < 0 or idx > max_idx]
    if bad:
        raise ValueError(f"Patch indices outside [0, {max_idx}]: {bad[:10]}")
    return indices


def parse_box(raw: str) -> Box:
    parts = [float(p.strip()) for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Box must be x0,y0,x1,y1, got {raw!r}")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Box has non-positive area: {raw!r}")
    return x0, y0, x1, y1


def normalize_box(box: Box, image_width: Optional[float], image_height: Optional[float]) -> Box:
    x0, y0, x1, y1 = box
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
        return box
    if image_width is None or image_height is None:
        raise ValueError("Pixel-space boxes require --image_width and --image_height.")
    return x0 / image_width, y0 / image_height, x1 / image_width, y1 / image_height


def patches_for_box(box: Box, grid_size: int, min_overlap: float = 0.0) -> Set[int]:
    x0, y0, x1, y1 = [max(0.0, min(1.0, v)) for v in box]
    if x1 <= x0 or y1 <= y0:
        return set()

    selected: Set[int] = set()
    patch_area = 1.0 / (grid_size * grid_size)
    for row in range(grid_size):
        py0, py1 = row / grid_size, (row + 1) / grid_size
        oy = max(0.0, min(y1, py1) - max(y0, py0))
        if oy <= 0:
            continue
        for col in range(grid_size):
            px0, px1 = col / grid_size, (col + 1) / grid_size
            ox = max(0.0, min(x1, px1) - max(x0, px0))
            if ox <= 0:
                continue
            if (ox * oy) / patch_area >= min_overlap:
                selected.add(row * grid_size + col)
    return selected


def load_label_mask(path: str):
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Reading semantic mask images requires Pillow. Install pillow or pass "
            "COCO annotations via --coco_category."
        ) from e

    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("Reading semantic mask images requires numpy.") from e

    with Image.open(path) as image:
        return np.asarray(image)


def semantic_patches_from_mask(
    mask,
    grid_size: int,
    label_ids: Optional[Sequence[int]] = None,
    rgb_values: Optional[Sequence[RGB]] = None,
    min_fraction: float = 0.0,
) -> Tuple[Set[int], List[List[float]]]:
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("Semantic mask processing requires numpy.") from e

    if label_ids is None and rgb_values is None:
        raise ValueError("Pass --mask_labels for label maps or --mask_rgb for RGB masks.")
    if label_ids is not None and rgb_values is not None:
        raise ValueError("Use either --mask_labels or --mask_rgb, not both.")

    arr = np.asarray(mask)
    if rgb_values is not None:
        if arr.ndim < 3 or arr.shape[-1] < 3:
            raise ValueError("--mask_rgb requires an RGB/RGBA semantic mask image.")
        rgb = arr[..., :3]
        selected_pixels = np.zeros(rgb.shape[:2], dtype=bool)
        for color in rgb_values:
            selected_pixels |= np.all(rgb == np.asarray(color, dtype=rgb.dtype), axis=-1)
    else:
        if arr.ndim == 3:
            if arr.shape[-1] >= 3 and np.all(arr[..., 0] == arr[..., 1]) and np.all(arr[..., 1] == arr[..., 2]):
                arr = arr[..., 0]
            else:
                raise ValueError(
                    "The semantic mask looks RGB. Use --mask_rgb r,g,b instead of --mask_labels."
                )
        selected_pixels = np.isin(arr, np.asarray(label_ids, dtype=arr.dtype))

    height, width = selected_pixels.shape
    selected: Set[int] = set()
    fractions: List[List[float]] = []

    for row in range(grid_size):
        y0 = int(round(row * height / grid_size))
        y1 = int(round((row + 1) * height / grid_size))
        y1 = max(y1, y0 + 1)
        fraction_row: List[float] = []
        for col in range(grid_size):
            x0 = int(round(col * width / grid_size))
            x1 = int(round((col + 1) * width / grid_size))
            x1 = max(x1, x0 + 1)
            patch = selected_pixels[y0:y1, x0:x1]
            fraction = float(patch.mean()) if patch.size else 0.0
            fraction_row.append(fraction)
            if fraction > 0.0 and fraction >= min_fraction:
                selected.add(row * grid_size + col)
        fractions.append(fraction_row)

    return selected, fractions


def point_in_polygon(x: float, y: float, polygon: Sequence[float]) -> bool:
    inside = False
    n = len(polygon) // 2
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[2 * i], polygon[2 * i + 1]
        xj, yj = polygon[2 * j], polygon[2 * j + 1]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def patches_for_polygons(
    polygons: Sequence[Sequence[float]],
    image_width: float,
    image_height: float,
    grid_size: int,
    min_fraction: float,
    samples_per_patch: int,
) -> Set[int]:
    selected: Set[int] = set()
    samples = max(1, samples_per_patch)
    denom = samples * samples

    normalized_polygons = []
    for polygon in polygons:
        normalized = []
        for idx, value in enumerate(polygon):
            normalized.append(value / image_width if idx % 2 == 0 else value / image_height)
        normalized_polygons.append(normalized)

    for row in range(grid_size):
        for col in range(grid_size):
            hits = 0
            for sy in range(samples):
                y = (row + (sy + 0.5) / samples) / grid_size
                for sx in range(samples):
                    x = (col + (sx + 0.5) / samples) / grid_size
                    if any(point_in_polygon(x, y, polygon) for polygon in normalized_polygons):
                        hits += 1
            fraction = hits / denom
            if fraction > 0.0 and fraction >= min_fraction:
                selected.add(row * grid_size + col)

    return selected


def patches_for_coco_rle(
    segmentation,
    image_height: int,
    image_width: int,
    grid_size: int,
    min_fraction: float,
) -> Optional[Set[int]]:
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None

    decoded = mask_utils.decode(segmentation)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    selected, _fractions = semantic_patches_from_mask(
        decoded.astype("uint8"),
        grid_size=grid_size,
        label_ids=[1],
        min_fraction=min_fraction,
    )
    return selected


def load_coco_category_segments(annotations_path: str) -> Dict[str, Dict[str, List[dict]]]:
    with open(annotations_path) as f:
        coco = json.load(f)

    category_by_id = {cat["id"]: cat["name"] for cat in coco["categories"]}
    image_by_id = {img["id"]: img for img in coco["images"]}
    segments_by_key: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))

    for ann in coco["annotations"]:
        image = image_by_id.get(ann["image_id"])
        category = category_by_id.get(ann["category_id"])
        if image is None or category is None:
            continue
        x, y, w, h = ann["bbox"]
        entry = {
            "category": category,
            "bbox_xywh": [x, y, w, h],
            "image_width": image["width"],
            "image_height": image["height"],
            "area": ann.get("area", w * h),
            "segmentation": ann.get("segmentation"),
        }
        filename = image["file_name"]
        stem, _ext = os.path.splitext(os.path.basename(filename))
        keys = {
            filename,
            os.path.basename(filename),
            stem,
            str(image["id"]),
            f"{image['id']:012d}",
            f"{image['id']:012d}.jpg",
        }
        for key in keys:
            segments_by_key[key][category].append(entry)

    return {key: dict(value) for key, value in segments_by_key.items()}


def coco_segments_for_stem(
    stem: str,
    category: str,
    segments_by_key: Dict[str, Dict[str, List[dict]]],
) -> List[dict]:
    basename = os.path.basename(stem)
    bare, ext = os.path.splitext(basename)
    keys = [stem, basename, bare]
    if not ext:
        keys.append(f"{basename}.jpg")
    if bare.isdigit():
        image_id = int(bare)
        keys.extend([str(image_id), f"{image_id:012d}", f"{image_id:012d}.jpg"])

    for key in keys:
        if key in segments_by_key and category in segments_by_key[key]:
            return segments_by_key[key][category]
    return []


def patch_coords(indices: Iterable[int], grid_size: int) -> List[dict]:
    return [
        {"index": int(idx), "row": int(idx) // grid_size, "col": int(idx) % grid_size}
        for idx in sorted(indices)
    ]


def apply_patch_mask(
    act: torch.Tensor,
    patch_indices: Sequence[int],
    fill_mode: MaskFillMode,
    noise_std: float = 1.0,
) -> torch.Tensor:
    masked = act.clone()
    if not patch_indices:
        return masked

    idx = torch.as_tensor(patch_indices, dtype=torch.long, device=masked.device)
    if fill_mode == "zero":
        masked[idx] = 0.0
    elif fill_mode == "mean":
        keep = torch.ones(masked.shape[0], dtype=torch.bool, device=masked.device)
        keep[idx] = False
        fill = masked[keep].mean(dim=0) if keep.any() else masked.mean(dim=0)
        masked[idx] = fill
    elif fill_mode == "noise":
        mean = masked.mean(dim=0, keepdim=True)
        std = masked.std(dim=0, keepdim=True).clamp_min(1e-6)
        masked[idx] = mean + torch.randn_like(masked[idx]) * std * noise_std
    else:
        raise ValueError(f"Unknown fill mode: {fill_mode!r}")
    return masked


@torch.no_grad()
def encode_scores(
    sae_model,
    act: torch.Tensor,
    source: str,
    sigma: Optional[torch.Tensor],
    device: str,
    feature_pool: FeaturePoolMode,
) -> torch.Tensor:
    x = act.unsqueeze(0).to(device)
    sigma_batch = sigma.to(device).view(-1) if sigma is not None else None
    _z_pre, z = sae_model.encode(x, source=source, sigma=sigma_batch)
    return pool_feature_scores(z, feature_pool).squeeze(0).detach().cpu()


def top_drops(
    baseline: torch.Tensor,
    masked: torch.Tensor,
    feature_ids: Optional[List[int]],
    top_k: int,
) -> List[dict]:
    delta = baseline - masked
    if feature_ids is None:
        k = min(top_k, delta.numel())
        ids = torch.topk(delta, k=k, largest=True).indices.tolist()
    else:
        ids = feature_ids

    rows = []
    for feat_id in ids:
        base = float(baseline[feat_id].item())
        masked_value = float(masked[feat_id].item())
        drop = base - masked_value
        rows.append(
            {
                "feature": int(feat_id),
                "baseline": base,
                "masked": masked_value,
                "drop": drop,
                "relative_drop": drop / base if abs(base) > 1e-12 else None,
            }
        )
    rows.sort(key=lambda item: item["drop"], reverse=True)
    return rows


def build_mask_indices(args, stem: str, grid_size: int) -> Tuple[Set[int], List[dict]]:
    patch_indices = parse_patch_indices(args.patches, grid_size)
    mask_sources: List[dict] = []

    if args.semantic_mask:
        label_ids = parse_int_ranges(args.mask_labels, "--mask_labels") if args.mask_labels else None
        rgb_values = parse_rgb_values(args.mask_rgb) if args.mask_rgb else None
        selected, fractions = semantic_patches_from_mask(
            load_label_mask(args.semantic_mask),
            grid_size=grid_size,
            label_ids=label_ids,
            rgb_values=rgb_values,
            min_fraction=args.semantic_min_fraction,
        )
        patch_indices.update(selected)
        mask_sources.append(
            {
                "type": "semantic_mask",
                "path": args.semantic_mask,
                "label_ids": label_ids,
                "rgb_values": [list(color) for color in rgb_values] if rgb_values else None,
                "min_fraction": args.semantic_min_fraction,
                "n_patches": len(selected),
                "max_patch_fraction": max(max(row) for row in fractions) if fractions else 0.0,
            }
        )

    for raw_box in args.box:
        box = normalize_box(parse_box(raw_box), args.image_width, args.image_height)
        selected = patches_for_box(box, grid_size, args.min_patch_overlap)
        patch_indices.update(selected)
        mask_sources.append(
            {
                "type": "box",
                "box_xyxy_normalized": list(box),
                "n_patches": len(selected),
            }
        )

    if args.coco_category:
        annotations_path = coco_annotation_path(args.coco_annotations_dir, args.coco_split)
        segments_by_key = load_coco_category_segments(annotations_path)
        coco_segments = coco_segments_for_stem(stem, args.coco_category, segments_by_key)
        if not coco_segments:
            raise ValueError(f"No COCO masks for category {args.coco_category!r} on stem {stem!r}.")
        for entry in coco_segments:
            segmentation = entry.get("segmentation")
            selected: Optional[Set[int]] = None
            source_type = "coco_segmentation"
            if isinstance(segmentation, list):
                selected = patches_for_polygons(
                    segmentation,
                    image_width=entry["image_width"],
                    image_height=entry["image_height"],
                    grid_size=grid_size,
                    min_fraction=args.semantic_min_fraction,
                    samples_per_patch=args.semantic_samples_per_patch,
                )
            elif isinstance(segmentation, dict):
                selected = patches_for_coco_rle(
                    segmentation,
                    image_height=entry["image_height"],
                    image_width=entry["image_width"],
                    grid_size=grid_size,
                    min_fraction=args.semantic_min_fraction,
                )

            if selected is None:
                x, y, w, h = entry["bbox_xywh"]
                box = normalize_box(
                    (x, y, x + w, y + h),
                    entry["image_width"],
                    entry["image_height"],
                )
                selected = patches_for_box(box, grid_size, args.min_patch_overlap)
                source_type = "coco_bbox_fallback"

            patch_indices.update(selected)
            mask_sources.append(
                {
                    "type": source_type,
                    "category": args.coco_category,
                    "bbox_xywh_pixels": entry["bbox_xywh"],
                    "area": entry["area"],
                    "min_fraction": args.semantic_min_fraction,
                    "n_patches": len(selected),
                }
            )

    return patch_indices, mask_sources


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", default=CACHE_ROOT)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--source", required=True)
    parser.add_argument("--stem", required=True, help="Cache stem without '_combined.npz'.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--timestep_idx", type=int, default=-1)
    parser.add_argument("--use_cls", action="store_true")
    parser.add_argument("--feature_ids", default=None, help="Comma/range list like '12,44,80-90'.")
    parser.add_argument("--top_k_drops", type=int, default=25)
    parser.add_argument("--feature_pool", default="max", choices=("max", "max_abs"))
    parser.add_argument("--fill", default="zero", choices=("zero", "mean", "noise"))
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--patches", default=None, help="Patch indices or row:col values, comma-separated.")
    parser.add_argument(
        "--box",
        action="append",
        default=[],
        help="Mask a box as x0,y0,x1,y1. Values <=1 are normalized; pixels require image size.",
    )
    parser.add_argument("--image_width", type=float, default=None)
    parser.add_argument("--image_height", type=float, default=None)
    parser.add_argument("--min_patch_overlap", type=float, default=0.0)
    parser.add_argument(
        "--semantic_mask",
        default=None,
        help="Path to a dense semantic label mask image aligned with the cached image.",
    )
    parser.add_argument(
        "--mask_labels",
        default=None,
        help="Integer labels to mask from --semantic_mask, e.g. '15' or '15,16,20-25'.",
    )
    parser.add_argument(
        "--mask_rgb",
        action="append",
        default=[],
        help="RGB color to mask from an RGB semantic mask, e.g. '128,0,0'. Can repeat.",
    )
    parser.add_argument(
        "--semantic_min_fraction",
        type=float,
        default=0.0,
        help="Minimum fraction of selected semantic pixels needed to mask a patch.",
    )
    parser.add_argument(
        "--semantic_samples_per_patch",
        type=int,
        default=5,
        help="Sampling resolution per patch for polygon masks such as COCO segmentations.",
    )
    parser.add_argument(
        "--coco_category",
        default=None,
        help="COCO category name; masks COCO segmentation pixels for that concept.",
    )
    parser.add_argument("--coco_annotations_dir", default=COCO_ANNOTATIONS_DIR)
    parser.add_argument("--coco_split", default=COCO_SPLIT, choices=("train2017", "val2017"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 <= args.semantic_min_fraction <= 1.0):
        raise ValueError("--semantic_min_fraction must be between 0 and 1.")
    if args.semantic_samples_per_patch <= 0:
        raise ValueError("--semantic_samples_per_patch must be positive.")
    if args.semantic_mask and not args.mask_labels and not args.mask_rgb:
        raise ValueError("--semantic_mask requires --mask_labels or --mask_rgb.")
    if args.mask_labels and args.mask_rgb:
        raise ValueError("Use either --mask_labels or --mask_rgb, not both.")

    model, cfg = load_universal_sae(args.checkpoint, args.config, args.device)
    diffusion_models = set(cfg.get("global", {}).get("diffusion_models", []))
    is_diffusion = args.source in diffusion_models

    act, sigma = load_activation_for_image(
        args.cache_root,
        args.stem,
        args.source,
        is_diffusion,
        args.timestep_idx,
        args.use_cls,
    )
    grid_size = infer_grid_size(act.shape[0])
    patch_indices, mask_sources = build_mask_indices(args, args.stem, grid_size)
    if not patch_indices:
        raise ValueError(
            "No patches selected. Pass --semantic_mask, --coco_category, --patches, or --box."
        )

    masked_act = apply_patch_mask(
        act,
        sorted(patch_indices),
        fill_mode=args.fill,
        noise_std=args.noise_std,
    )
    baseline_scores = encode_scores(
        model, act, args.source, sigma, args.device, args.feature_pool
    )
    masked_scores = encode_scores(
        model, masked_act, args.source, sigma, args.device, args.feature_pool
    )
    feature_ids = parse_feature_ids(args.feature_ids)

    result = {
        "metadata": {
            "source": args.source,
            "stem": args.stem,
            "checkpoint": args.checkpoint,
            "is_diffusion": is_diffusion,
            "timestep_idx": args.timestep_idx if is_diffusion else None,
            "sigma": float(sigma.item()) if sigma is not None and sigma.numel() == 1 else None,
            "feature_pool": args.feature_pool,
            "fill": args.fill,
            "grid_size": grid_size,
            "n_patches_total": grid_size * grid_size,
            "n_patches_masked": len(patch_indices),
            "masked_fraction": len(patch_indices) / (grid_size * grid_size),
            "semantic_min_fraction": args.semantic_min_fraction,
            "mask_sources": mask_sources,
        },
        "masked_patches": patch_coords(patch_indices, grid_size),
        "feature_results": top_drops(
            baseline_scores,
            masked_scores,
            feature_ids=feature_ids,
            top_k=args.top_k_drops,
        ),
    }

    output = args.output
    if output is None:
        safe_stem = os.path.basename(args.stem)
        output = f"patch_mask_probe_{safe_stem}_{args.source}.json"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[mask-probe] masked {len(patch_indices)}/{grid_size * grid_size} patches")
    print(f"[mask-probe] saved -> {output}")
    for row in result["feature_results"][: min(10, len(result["feature_results"]))]:
        rel = row["relative_drop"]
        rel_s = "n/a" if rel is None else f"{rel:.3f}"
        print(
            f"  feature {row['feature']:>5}: "
            f"{row['baseline']:.6g} -> {row['masked']:.6g} "
            f"drop={row['drop']:.6g} rel={rel_s}"
        )


if __name__ == "__main__":
    main()
