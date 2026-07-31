"""Run the Universal SAE on a chosen list of images and report top features.

Unlike top_activating_images.py (feature -> images across the whole cache),
this goes image -> features: for each image you pass, it encodes the cached
activation through the SAE and prints the features that fire most. Optionally
it then scans the cache to show which other images share those features.

The SAE operates on cached activations, not raw pixels, so every image you
pass must already have a '<stem>_combined.npz' in --cache_root. An "image path"
is resolved to its stem = basename without extension (e.g.
'/data/000000562818.jpg' -> '000000562818').

Examples:
    python run_inference_on_images.py 000000562818.jpg 000000001000.jpg
    python run_inference_on_images.py imgs/*.jpg --source PixArt --top_k 15
    python run_inference_on_images.py 000000562818 --top_images 0   # skip cache scan
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import print_top_features
from universal_sae import UniversalSAE
from spatial_align import build_spatial_aligner_from_config
from data import CocoActivationDataset
from pixart_timestep import resolve_pixart_timestep
from top_activating_images import (
    build_coco_label_lookup,
    coco_annotation_path,
    find_latest_checkpoint,
    labels_for_stem,
    COCO_ANNOTATIONS_DIR,
    COCO_SPLIT,
    WEIGHTS_DIR,
)

DEFAULT_CONFIG = "/content/algoverse_github/config.yaml"
DEFAULT_CACHE = "/content/combined_cache"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="*",
                   help="Image paths or bare stems. Each must have a matching "
                        "<stem>_combined.npz in --cache_root.")
    p.add_argument("--list_stems", action="store_true",
                   help="Print all stems available in --cache_root and exit.")
    p.add_argument("--source", default="DinoV2",
                   help="Which source model's activations to encode (must be in --sources).")
    p.add_argument("--sources", nargs="+", default=["DinoV2", "PixArt"],
                   help="Sources the dataset should load from the combined npz.")
    p.add_argument("--top_k", type=int, default=10, help="Top features to report per image.")
    p.add_argument("--top_images", type=int, default=5,
                   help="Top-activating images to show per feature (0 = skip the cache scan).")
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint path. If omitted or not a file, the newest "
                        ".pt/.pth under --weights_dir is auto-selected.")
    p.add_argument("--weights_dir", default=WEIGHTS_DIR,
                   help="Searched for the latest checkpoint when --ckpt is not given.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--cache_root", default=DEFAULT_CACHE)
    p.add_argument("--coco_annotations_dir", default=COCO_ANNOTATIONS_DIR)
    p.add_argument("--coco_split", default=COCO_SPLIT)
    p.add_argument("--skip_coco_labels", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def resolve_stem(path: str, stem_to_idx: dict) -> str:
    """Map an image path (or bare stem) to a stem present in the cache."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in stem_to_idx:
        return stem
    if stem.isdigit():
        padded = f"{int(stem):012d}"  # COCO ids are zero-padded to 12 digits
        if padded in stem_to_idx:
            return padded
    sample = ", ".join(list(stem_to_idx)[:5])
    raise KeyError(
        f"No cached activations for '{path}' (looked for stem '{stem}'). "
        f"Available stems start with: {sample} ..."
    )


@torch.no_grad()
def feature_scores(model, acts, meta, source, diffusion_models, aligner, device):
    """Encode one image's activations -> (per-feature score vector, raw latents).

    Score = mean |latent| over batch and tokens, matching inference._score_latents.
    Diffusion sources are sliced to the timestep the checkpoint trained on and
    scored with that timestep's sigma.
    """
    if source not in acts:
        raise KeyError(f"Source '{source}' not in loaded activations {list(acts)}. "
                       f"Add it to --sources.")
    x = acts[source].unsqueeze(0).to(device)  # vision:(1,N,D)  diffusion:(1,T,N,D)
    sigma = None
    if source in diffusion_models:
        sig = meta["sigmas_by_model"].get(source)
        if sig is None:
            sig = meta.get("sigmas")
        if sig is None:
            raise ValueError(f"Diffusion source '{source}' has no sigmas in metadata.")
        t_idx = resolve_pixart_timestep(
            x.shape[1], config_global=getattr(model, "_training_global", None)
        )
        sigma = sig.to(device).view(-1)[t_idx].view(1)
        x = x[:, t_idx]  # -> (1,N,D)
    if aligner is not None:
        x = aligner.align(x, source=source)
    _z_pre, z = model.encode(x, source=source, sigma=sigma)
    scores = z.abs().mean(dim=tuple(range(z.dim() - 1)))  # -> (K,)
    return scores.cpu(), z


def main():
    args = parse_args()

    if args.list_stems:
        import glob
        paths = sorted(glob.glob(os.path.join(args.cache_root, "*_combined.npz")))
        stems = [os.path.basename(p)[: -len("_combined.npz")] for p in paths]
        print(f"[run-inference] {len(stems)} stems in {args.cache_root}:")
        for s in stems:
            print(f"  {s}")
        return

    if not args.images:
        raise SystemExit("No images given. Pass image paths/stems, or use --list_stems.")
    if args.source not in args.sources:
        raise SystemExit(f"--source {args.source!r} must be one of --sources {args.sources}")

    # ---- Load model ----
    ckpt_path = args.ckpt
    if not ckpt_path or not os.path.isfile(ckpt_path):
        ckpt_path = find_latest_checkpoint(args.weights_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    g = cfg["global"]
    saved_cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    saved_global = saved_cfg.get("global", {})
    saved_sae = saved_cfg.get("sae_params", {})
    diffusion_models = set(ckpt.get("diffusion_models", saved_global.get("diffusion_models", g.get("diffusion_models", []))))

    model = UniversalSAE(
        model_dims=ckpt["model_dims"],
        latent_dim=ckpt["latent_dim"],
        diffusion_models=diffusion_models,
        model_tokens=ckpt["model_tokens"],
        top_k=int(saved_sae.get("top_k", cfg["sae_params"].get("top_k", 64))),
        cls_pool_mode=str(saved_global.get("cls_pool_mode", g.get("cls_pool_mode", "none"))),
        use_tide=bool(saved_global.get("use_tide", g.get("use_tide", False))),
        timestep_dim=int(saved_global.get("timestep_dim", g.get("timestep_dim", 256))),
    )
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint architecture does not match its saved configuration: "
            f"missing={len(missing)}, unexpected={len(unexpected)}."
        )
    model.eval().to(args.device)

    eval_g = {**g, **saved_global}
    # Ride along on the model so feature_scores can pick the training timestep
    # without threading the config through every call.
    model._training_global = eval_g
    aligner = build_spatial_aligner_from_config(eval_g, ckpt.get("model_tokens_native", ckpt["model_tokens"]))

    # ---- Load dataset ----
    ds = CocoActivationDataset(
        cache_root=args.cache_root,
        sources=args.sources,
        combined_npz=True,
        standardize=bool(eval_g.get("standardize", True)),
        return_metadata=True,
        diffusion_models=[s for s in args.sources if s in diffusion_models],
        use_class_tokens=False,
        standardization_stats=ckpt.get("standardization_stats"),
        stats_seed=eval_g.get("stats_seed", 0),
    )
    stem_to_idx = {stem: i for i, stem in enumerate(ds.stems)}

    # ---- image -> top features, for each requested image ----
    requested = [(path, resolve_stem(path, stem_to_idx)) for path in args.images]

    per_image_features = {}   # stem -> features list
    target_indices = set()    # union of all feature ids across requested images

    print("=" * 72)
    for path, stem in requested:
        (acts, meta), _ = ds[stem_to_idx[stem]]
        scores, _z = feature_scores(model, acts, meta, args.source,
                                    diffusion_models, aligner, args.device)
        k = min(args.top_k, scores.numel())
        top_scores, top_idx = torch.topk(scores, k=k)
        features = [{"feature_idx": int(i), "score": float(s)}
                    for i, s in zip(top_idx.tolist(), top_scores.tolist())]
        per_image_features[stem] = features
        target_indices.update(f["feature_idx"] for f in features)

        print(f"\n{stem}   source={args.source}   (from {path})")
        print_top_features(features)
    print("=" * 72)

    if args.top_images <= 0:
        return

    # ---- feature -> top-activating images, scanning the whole cache once ----
    coco_labels = None
    if not args.skip_coco_labels:
        coco_labels = build_coco_label_lookup(
            coco_annotation_path(args.coco_annotations_dir, args.coco_split))

    scores_per_feature = defaultdict(list)
    target_list = sorted(target_indices)
    try:
        from tqdm import tqdm
        it = tqdm(range(len(ds)), desc="Scanning cache")
    except ImportError:
        it = range(len(ds))

    with torch.no_grad():
        for i in it:
            (acts, meta), _ = ds[i]
            stem = ds.stems[i]
            scores, _z = feature_scores(model, acts, meta, args.source,
                                        diffusion_models, aligner, args.device)
            for feat_idx in target_list:
                scores_per_feature[feat_idx].append((stem, scores[feat_idx].item()))

    def label_str(stem):
        if coco_labels is None:
            return ""
        labels = labels_for_stem(stem, coco_labels)
        return f"  [{', '.join(labels) or '(no labels)'}]"

    for path, stem in requested:
        print("\n" + "=" * 72)
        print(f"Top images per feature for {stem} (source={args.source})")
        for rank, feat in enumerate(per_image_features[stem], 1):
            fi = feat["feature_idx"]
            top = sorted(scores_per_feature[fi], key=lambda x: x[1], reverse=True)[:args.top_images]
            print(f"\n#{rank}  Feature {fi}  (score={feat['score']:.4f})")
            for s, score in top:
                print(f"    {s}  score={score:.4f}{label_str(s)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
