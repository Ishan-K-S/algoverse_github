"""Rank cached images by SAE feature activation and write top-K per feature to JSON."""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

# THESE ARE WHAT TO FILL IN

CACHE_ROOT = "/content/combined_cache"
CONFIG_PATH = "/content/algoverse_github/config.yaml"
CHECKPOINT_PATH = "/content/algoverse_github/weights/usae_run/usae_epoch_30.pth"
SOURCE = "DinoV2"
OUTPUT_PATH = "/top_activations.json"

MODELS_WITH_CLS = {"DinoV2", "ViT", "CLIP"}


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


def load_universal_sae(checkpoint_path: str, config_path: str, device: str):
    try:
        from universal_sae import UniversalSAE
    except ImportError as e:
        raise ImportError(
            "Could not import UniversalSAE. "
            "Make sure universal_sae.py is on the Python path."
        ) from e

    print(f"[sae] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

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
        diffusion_models=set(ckpt.get(
            "diffusion_models", g.get("diffusion_models", []))),
        model_tokens=ckpt["model_tokens"],
        shared_latent_tokens=ckpt["shared_latent_tokens"],
        timestep_dim=int(pick("timestep_dim", 256)),
        top_k=int(sp.get("top_k", pick("top_k", 64))),
        topk_temperature=float(pick("topk_temperature", 0.1)),
        use_soft_topk=bool(pick("use_soft_topk", False)),
        interpolation_mode=str(pick("interpolation_mode", "bilinear")),
        token_reshape_mode=str(pick("token_reshape_mode", "attention")),
        attention_heads=int(pick("attention_heads", 8)),
        attention_dropout=float(pick("attention_dropout", 0.0)),
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

        # Per-image, per-feature score: mean |z| across tokens.
        # Matches inference.py's z.abs().mean(dim=(0, 1)) reduction when B=1.
        if z.dim() == 3:
            per_image_scores = z.abs().mean(dim=1)
        elif z.dim() == 2:
            per_image_scores = z.abs()
        else:
            raise ValueError(f"Unexpected latent shape from SAE: {z.shape}")

        per_image_scores = per_image_scores.cpu()

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
            entries.append({"filename": stems[img_idx], "score": score})
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
        "scoring": "mean_abs_across_tokens",
    }
    output_obj = {"metadata": metadata, "top_activations": results}
    with open(output_path, "w") as f:
        json.dump(output_obj, f, indent=2)
    print(f"[top-act] Saved -> {output_path}")

    return output_obj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top_pct", type=float, default=0.1)
    p.add_argument("--timestep_idx", type=int, default=-1)
    p.add_argument("--use_cls", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    model, cfg = load_universal_sae(args.checkpoint, args.config, args.device)
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
    )


if __name__ == "__main__":
    main()
