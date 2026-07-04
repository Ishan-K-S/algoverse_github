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
    WEIGHTS_DIR,
    FeaturePoolMode,
    build_coco_label_lookup,
    coco_annotation_path,
    discover_stems,
    find_latest_checkpoint,
    labels_for_stem,
    load_activation_for_image,
    load_universal_sae,
    pool_feature_scores,
)


MaskFillMode = str
Box = Tuple[float, float, float, float]


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


def load_coco_category_boxes(annotations_path: str) -> Dict[str, Dict[str, List[dict]]]:
    with open(annotations_path) as f:
        coco = json.load(f)

    category_by_id = {cat["id"]: cat["name"] for cat in coco["categories"]}
    image_by_id = {img["id"]: img for img in coco["images"]}
    boxes_by_key: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))

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
            boxes_by_key[key][category].append(entry)

    return {key: dict(value) for key, value in boxes_by_key.items()}


def coco_boxes_for_stem(
    stem: str,
    category: str,
    boxes_by_key: Dict[str, Dict[str, List[dict]]],
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
        if key in boxes_by_key and category in boxes_by_key[key]:
            return boxes_by_key[key][category]
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
        boxes_by_key = load_coco_category_boxes(annotations_path)
        coco_boxes = coco_boxes_for_stem(stem, args.coco_category, boxes_by_key)
        if not coco_boxes:
            raise ValueError(f"No COCO boxes for category {args.coco_category!r} on stem {stem!r}.")
        for entry in coco_boxes:
            x, y, w, h = entry["bbox_xywh"]
            box = normalize_box(
                (x, y, x + w, y + h),
                entry["image_width"],
                entry["image_height"],
            )
            selected = patches_for_box(box, grid_size, args.min_patch_overlap)
            patch_indices.update(selected)
            mask_sources.append(
                {
                    "type": "coco_box",
                    "category": args.coco_category,
                    "bbox_xywh_pixels": entry["bbox_xywh"],
                    "box_xyxy_normalized": list(box),
                    "area": entry["area"],
                    "n_patches": len(selected),
                }
            )

    return patch_indices, mask_sources


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", default=CACHE_ROOT)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--weights_dir", default=WEIGHTS_DIR,
                        help="Searched for the latest .pth when --checkpoint is not given.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--stem", default=None,
                        help="Cache stem without '_combined.npz' (e.g. '000000001000'). "
                             "Omit to use the first available stem. Run --list_stems to see all.")
    parser.add_argument("--list_stems", action="store_true",
                        help="Print all available stems in --cache_root and exit.")
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
    parser.add_argument("--coco_category", default=None)
    parser.add_argument("--coco_annotations_dir", default=COCO_ANNOTATIONS_DIR)
    parser.add_argument("--coco_split", default=COCO_SPLIT, choices=("train2017", "val2017"))
    parser.add_argument("--skip_coco_labels", action="store_true",
                        help="Skip loading COCO category labels for the image.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve stem — list or auto-pick if not provided.
    stems = discover_stems(args.cache_root)
    if args.list_stems:
        print(f"[mask-probe] {len(stems)} stems in {args.cache_root}:")
        for s in stems:
            print(f"  {s}")
        return

    stem = args.stem
    if stem is None:
        stem = stems[0]
        print(f"[mask-probe] No --stem given, using first available: {stem}")
    else:
        expected = os.path.join(args.cache_root, f"{stem}_combined.npz")
        if not os.path.isfile(expected):
            raise FileNotFoundError(
                f"File not found: {expected}\n"
                f"  --stem must be a bare image stem, not a directory path.\n"
                f"  Example: --stem {stems[0]}\n"
                f"  Run with --list_stems to see all available stems."
            )

    # Load COCO labels for this image.
    image_labels: List[str] = []
    if not args.skip_coco_labels:
        annotations_path = coco_annotation_path(args.coco_annotations_dir, args.coco_split)
        coco_lookup = build_coco_label_lookup(annotations_path)
        image_labels = labels_for_stem(stem, coco_lookup)

    checkpoint = args.checkpoint
    if checkpoint is None or not os.path.isfile(checkpoint):
        checkpoint = find_latest_checkpoint(args.weights_dir)

    model, cfg = load_universal_sae(checkpoint, args.config, args.device)
    diffusion_models = set(cfg.get("global", {}).get("diffusion_models", []))
    is_diffusion = args.source in diffusion_models

    act, sigma = load_activation_for_image(
        args.cache_root,
        stem,
        args.source,
        is_diffusion,
        args.timestep_idx,
        args.use_cls,
    )
    grid_size = infer_grid_size(act.shape[0])
    patch_indices, mask_sources = build_mask_indices(args, stem, grid_size)
    if not patch_indices:
        raise ValueError("No patches selected. Pass --patches, --box, or --coco_category.")

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
            "stem": stem,
            "checkpoint": checkpoint,
            "is_diffusion": is_diffusion,
            "timestep_idx": args.timestep_idx if is_diffusion else None,
            "sigma": float(sigma.item()) if sigma is not None and sigma.numel() == 1 else None,
            "feature_pool": args.feature_pool,
            "fill": args.fill,
            "grid_size": grid_size,
            "n_patches_total": grid_size * grid_size,
            "n_patches_masked": len(patch_indices),
            "masked_fraction": len(patch_indices) / (grid_size * grid_size),
            "mask_sources": mask_sources,
            "coco_labels": image_labels,
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
        safe_stem = os.path.basename(stem)
        output = f"patch_mask_probe_{safe_stem}_{args.source}.json"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[mask-probe] masked {len(patch_indices)}/{grid_size * grid_size} patches")
    print(f"[mask-probe] coco labels : {', '.join(image_labels) if image_labels else '(none found)'}")
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
