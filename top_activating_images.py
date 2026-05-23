"""Rank cached images by SAE feature activation and write top-K per feature to JSON."""

import argparse
import glob
import json
import os
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Set

import numpy as np
import torch
import yaml
from tqdm import tqdm


# THESE ARE WHAT TO FILL IN

CACHE_ROOT = "/content/combined_cache"
CONFIG_PATH = "/content/algoverse_github/config.yaml"
WEIGHTS_DIR = "/content/algoverse_github/weights"    # searched automatically for the latest checkpoint
CHECKPOINT_PATH = "/content/algoverse_github/weights/ex16_bs16_topk64_LR0.0005_alignDinoV2_30ep/usae_epoch_29.pth"  
SOURCE = "DinoV2"
OUTPUT_PATH = "/content/top_activations.json"
DRIVE_SAVE_DIR = "/content/drive/My Drive/algoverse_results"

COCO_ANNOTATIONS_DIR = "/content/coco_annotations"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_SPLIT = "val2017"

MODELS_WITH_CLS = {"DinoV2", "ViT", "CLIP"}
FeaturePoolMode = Literal["max", "max_abs"]


def find_latest_checkpoint(models_dir: str) -> str:
    """Return the most recently modified .pt or .pth file under models_dir."""
    candidates = glob.glob(os.path.join(models_dir, "**", "*.pt"), recursive=True)
    candidates += glob.glob(os.path.join(models_dir, "**", "*.pth"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No .pt/.pth checkpoints found under {models_dir}")
    latest = max(candidates, key=os.path.getmtime)
    print(f"[ckpt] Auto-selected ({os.path.getsize(latest)/1_048_576:.1f} MB): {latest}")
    return latest


def save_to_drive(output_path: str, checkpoint_path: str, drive_dir: str) -> None:
    """Copy the results JSON and SAE checkpoint to Google Drive."""
    try:
        from google.colab import drive as _colab_drive
        if not os.path.isdir("/content/drive/My Drive"):
            print("[drive] Mounting Google Drive...")
            _colab_drive.mount("/content/drive")
    except ImportError:
        print("[drive] Not running in Colab - skipping Drive upload.")
        return

    os.makedirs(drive_dir, exist_ok=True)
    for src in (output_path, checkpoint_path):
        if not os.path.isfile(src):
            print(f"[drive] Warning: {src} not found - skipping.")
            continue
        dst = os.path.join(drive_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"[drive] Saved: {dst}")


def load_combined_npz(path: str) -> Dict[str, np.ndarray]:
    npz = np.load(path, mmap_mode="r", allow_pickle=False)
    return {k: npz[k].copy() for k in npz.files}


def discover_stems(cache_root: str) -> List[str]:
    suffix = "_combined.npz"
    files = [f for f in os.listdir(cache_root) if f.endswith(suffix)]
    if not files:
        raise RuntimeError(
            f"No '*_combined.npz' files in {cache_root}. "
            "Run combine_cached_acts.py first."
        )
    return sorted(f[: -len(suffix)] for f in files)


def download_coco_annotations(annotations_dir: str) -> None:
    """Download and extract COCO 2017 annotations if they are not already present."""
    os.makedirs(annotations_dir, exist_ok=True)
    zip_path = os.path.join(annotations_dir, "annotations_trainval2017.zip")

    if not os.path.isfile(zip_path):
        print(f"[coco] Downloading annotations from {COCO_ANNOTATIONS_URL}")
        urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zip_path)

    expected = os.path.join(annotations_dir, "annotations")
    if not os.path.isdir(expected):
        print(f"[coco] Extracting annotations to {annotations_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(annotations_dir)


def coco_annotation_path(annotations_dir: str, split: str) -> str:
    """Return the local instances annotation path, downloading COCO 2017 if needed."""
    split = split.replace("instances_", "").replace(".json", "")
    path = os.path.join(annotations_dir, "annotations", f"instances_{split}.json")
    if not os.path.isfile(path):
        download_coco_annotations(annotations_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find COCO annotations for split {split!r} at {path}"
        )
    return path


def _coco_lookup_keys(filename: str, image_id: Optional[int]) -> Set[str]:
    basename = os.path.basename(filename)
    stem, ext = os.path.splitext(basename)
    keys = {filename, basename, stem}
    if ext:
        keys.add(basename.lower())
    if image_id is not None:
        keys.update({
            str(image_id),
            f"{image_id:012d}",
            f"{image_id:012d}.jpg",
        })
    return keys


def build_coco_label_lookup(annotations_path: str) -> Dict[str, List[str]]:
    """Build a cache-stem-friendly lookup from COCO image filename/id to labels."""
    print(f"[coco] Loading labels from {annotations_path}")
    with open(annotations_path) as f:
        coco = json.load(f)

    category_by_id = {cat["id"]: cat["name"] for cat in coco["categories"]}
    image_by_id = {img["id"]: img for img in coco["images"]}
    labels_by_image_id: Dict[int, Set[str]] = defaultdict(set)

    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        category = category_by_id.get(ann["category_id"])
        if category is not None:
            labels_by_image_id[image_id].add(category)

    lookup: Dict[str, List[str]] = {}
    for image_id, image in image_by_id.items():
        labels = sorted(labels_by_image_id.get(image_id, set()))
        for key in _coco_lookup_keys(image["file_name"], image_id):
            lookup[key] = labels

    print(f"[coco] Indexed labels for {len(image_by_id)} images")
    return lookup


def labels_for_stem(stem: str, coco_labels: Optional[Dict[str, List[str]]]) -> List[str]:
    if coco_labels is None:
        return []

    basename = os.path.basename(stem)
    bare, ext = os.path.splitext(basename)
    keys = [stem, basename, bare]
    if not ext:
        keys.append(f"{basename}.jpg")
    if bare.isdigit():
        image_id = int(bare)
        keys.extend([str(image_id), f"{image_id:012d}", f"{image_id:012d}.jpg"])

    for key in keys:
        if key in coco_labels:
            return coco_labels[key]
    return []


def load_universal_sae(checkpoint_path: str, config_path: str, device: str):
    try:
        from universal_sae import UniversalSAE
    except ImportError as e:
        raise ImportError(
            "Could not import UniversalSAE. "
            "Make sure universal_sae.py is on the Python path."
        ) from e

    print(f"[sae] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    g = cfg.get("global", {})
    sp = cfg.get("sae_params", {})
    gc = (ckpt.get("config") or {}).get("global", {})

    def pick(key, default=None):
        return gc.get(key, g.get(key, default))

    model = UniversalSAE(
        model_dims=ckpt["model_dims"],
        latent_dim=ckpt["latent_dim"],
        diffusion_models=set(ckpt.get("diffusion_models", g.get("diffusion_models", []))),
        model_tokens=ckpt["model_tokens"],
        top_k=int(sp.get("top_k", pick("top_k", 64))),
        cls_pool_mode=str(pick("cls_pool_mode", "none")),
        use_tide=bool(pick("use_tide", False)),
        timestep_dim=int(pick("timestep_dim", 256)),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model.to(device), cfg


def load_activation_for_image(
    cache_root: str,
    stem: str,
    source: str,
    is_diffusion: bool,
    timestep_idx: int,
    use_cls: bool,
):
    path = os.path.join(cache_root, f"{stem}_combined.npz")
    data = load_combined_npz(path)

    if source not in data:
        raise KeyError(f"Source '{source}' missing in {path}")

    act = torch.from_numpy(data[source]).float()
    sigma = None

    if is_diffusion:
        if act.dim() != 3:
            raise ValueError(
                f"Diffusion source '{source}' should be (T, N, D), got {act.shape}"
            )
        T = act.shape[0]
        idx = timestep_idx if timestep_idx >= 0 else T + timestep_idx
        idx = max(0, min(idx, T - 1))

        sigma_key = f"{source}__sigmas"
        if sigma_key in data:
            sigma_arr = data[sigma_key]
            sigma = torch.from_numpy(sigma_arr).float().view(-1)[idx:idx + 1]

        act = act[idx]
    elif source in MODELS_WITH_CLS and not use_cls:
        act = act[1:]

    return act, sigma


def pool_feature_scores(z: torch.Tensor, mode: FeaturePoolMode) -> torch.Tensor:
    """
    Collapse token-level SAE activations into one feature vector per image.

    The SAE returns one latent vector per token/canonical token. This function
    answers "did this image activate the feature?" instead of "which token did?"
    by reducing over the token axis.
    """
    if z.dim() == 2:
        return z.abs() if mode.endswith("_abs") else z
    if z.dim() != 3:
        raise ValueError(f"Unexpected latent shape from SAE: {z.shape}")

    values = z.abs() if mode.endswith("_abs") else z
    if mode in {"max", "max_abs"}:
        return values.max(dim=1).values
    raise ValueError(f"Unknown feature pool mode: {mode!r}")


@torch.no_grad()
def compute_top_activations(
    cache_root: str,
    sae_model,
    source: str,
    diffusion_models: set,
    output_path: str,
    top_pct: float = 0.1,
    timestep_idx: int = -1,
    use_cls: bool = False,
    batch_size: int = 32,
    device: str = "cuda",
    max_images: Optional[int] = None,
    coco_labels: Optional[Dict[str, List[str]]] = None,
    feature_pool: FeaturePoolMode = "max",
):
    is_diffusion = source in diffusion_models
    stems = discover_stems(cache_root)
    if max_images is not None:
        stems = stems[:max_images]

    n_images = len(stems)
    top_n = max(1, int(round(n_images * top_pct / 100.0)))
    print(f"[top-act] {n_images} images, source={source}, "
          f"top_pct={top_pct}% -> keeping top {top_n} images per feature")

    scores_matrix: Optional[torch.Tensor] = None
    K: Optional[int] = None

    for start in tqdm(range(0, n_images, batch_size), desc="encoding"):
        batch_stems = stems[start:start + batch_size]

        acts_list, sigmas_list = [], []
        for stem in batch_stems:
            act, sigma = load_activation_for_image(
                cache_root, stem, source, is_diffusion, timestep_idx, use_cls
            )
            acts_list.append(act)
            if sigma is not None:
                sigmas_list.append(sigma)

        x = torch.stack(acts_list, dim=0).to(device)

        sigma_batch = None
        if is_diffusion and sigmas_list:
            sigma_batch = torch.stack(sigmas_list, dim=0).to(device).view(-1)

        _, z = sae_model.encode(x, source=source, sigma=sigma_batch)

        per_image_scores = pool_feature_scores(z, feature_pool).cpu()

        if scores_matrix is None:
            K = per_image_scores.shape[1]
            scores_matrix = torch.full(
                (n_images, K), float("-inf"), dtype=torch.float32
            )
            print(f"[top-act] Latent dim K = {K}, allocating "
                  f"({n_images} x {K}) score matrix "
                  f"({n_images * K * 4 / 1e6:.1f} MB)")

        scores_matrix[start:start + len(batch_stems)] = per_image_scores

    if scores_matrix is None:
        raise RuntimeError("No images processed.")

    print(f"[top-act] Ranking top {top_n} images for each of {K} features...")
    top_values, top_indices = torch.topk(scores_matrix, k=top_n, dim=0, largest=True)

    results: Dict[str, List[Dict[str, float]]] = {}
    for feat_id in range(K):
        entries = []
        for rank in range(top_n):
            img_idx = int(top_indices[rank, feat_id].item())
            score = float(top_values[rank, feat_id].item())
            entry = {"filename": stems[img_idx], "score": score}
            if rank == 0:
                entry["coco_labels"] = labels_for_stem(stems[img_idx], coco_labels)
            entries.append(entry)
        results[str(feat_id)] = entries

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    metadata = {
        "source": source,
        "is_diffusion": is_diffusion,
        "timestep_idx": timestep_idx if is_diffusion else None,
        "use_cls": use_cls,
        "n_images": n_images,
        "n_features": K,
        "top_pct": top_pct,
        "top_n_per_feature": top_n,
        "scoring": f"{feature_pool}_across_tokens",
        "coco_labels": coco_labels is not None,
        "coco_labels_scope": "rank_0_entry_per_feature",
    }
    output_obj = {"metadata": metadata, "top_activations": results}
    with open(output_path, "w") as f:
        json.dump(output_obj, f, indent=2)
    print(f"[top-act] Saved -> {output_path}")

    return output_obj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", default=CACHE_ROOT)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    p.add_argument("--source", default=SOURCE)
    p.add_argument("--output", default=OUTPUT_PATH)
    p.add_argument("--top_pct", type=float, default=0.1)
    p.add_argument("--timestep_idx", type=int, default=-1)
    p.add_argument("--use_cls", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument(
        "--feature_pool",
        default="max",
        choices=("max", "max_abs"),
        help=(
            "How to collapse token-level SAE activations into one image-level "
            "feature score. The default max finds images where a feature fires "
            "strongly anywhere."
        ),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--coco_annotations_dir", default=COCO_ANNOTATIONS_DIR)
    p.add_argument("--coco_split", default=COCO_SPLIT, choices=("train2017", "val2017"))
    p.add_argument("--skip_coco_labels", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[top-act] ---- Starting top_activating_images ----")
    print(f"[top-act] cache_root  : {args.cache_root}  exists={os.path.isdir(args.cache_root)}")
    print(f"[top-act] config      : {args.config}  exists={os.path.isfile(args.config)}")
    print(f"[top-act] checkpoint  : {args.checkpoint}  exists={os.path.isfile(args.checkpoint)}")
    print(f"[top-act] source      : {args.source}")
    print(f"[top-act] output      : {args.output}")
    print(f"[top-act] device      : {args.device}")
    if not os.path.isdir(args.cache_root):
        raise FileNotFoundError(
            f"[top-act] Combined cache not found: {args.cache_root}\n"
            "  Run combine_cached_acts.py first."
        )
    if not os.path.isfile(args.config):
        raise FileNotFoundError(
            f"[top-act] config.yaml not found: {args.config}"
        )

    checkpoint = args.checkpoint
    if not os.path.isfile(checkpoint):
        print(f"[top-act] Checkpoint not found at {checkpoint!r}, searching {WEIGHTS_DIR}...")
        checkpoint = find_latest_checkpoint(WEIGHTS_DIR)

    coco_labels = None
    if not args.skip_coco_labels:
        annotations_path = coco_annotation_path(args.coco_annotations_dir, args.coco_split)
        coco_labels = build_coco_label_lookup(annotations_path)

    model, cfg = load_universal_sae(checkpoint, args.config, args.device)
    diffusion_models = set(cfg.get("global", {}).get("diffusion_models", []))

    if args.source in diffusion_models:
        print(f"[main] '{args.source}' is a diffusion model "
              f"(timestep_idx={args.timestep_idx})")
    else:
        print(f"[main] '{args.source}' is a vision encoder")

    compute_top_activations(
        cache_root=args.cache_root,
        sae_model=model,
        source=args.source,
        diffusion_models=diffusion_models,
        output_path=args.output,
        top_pct=args.top_pct,
        timestep_idx=args.timestep_idx,
        use_cls=args.use_cls,
        batch_size=args.batch_size,
        device=args.device,
        max_images=args.max_images,
        coco_labels=coco_labels,
        feature_pool=args.feature_pool,
    )

    save_to_drive(args.output, checkpoint, DRIVE_SAVE_DIR)


if __name__ == "__main__":
    main()
