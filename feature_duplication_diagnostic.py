"""
Is the dictionary actually made of distinct features, or a handful of directions
copied across many indices?

Two ways a TopK SAE fakes a large dictionary:
  1. Decoder columns point the same way, so different indices decode to the
     same thing. Caught here by cosine similarity between columns.
  2. Features fire on the same images regardless of content, which makes them
     look strong in a top-features listing while carrying no information.
     Caught here by how concentrated each feature's top images are.

The second one is what prompted this script: three different COCO images came
back with an identical top-8 feature list, in near-identical order.

Part 1 needs only the checkpoint. Part 2 needs the cache; skip it with --no_images.

    python feature_duplication_diagnostic.py \
        --ckpt /content/ckpt29/usae_epoch_29.pth \
        --cache /content/diag_cache --n_images 300
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
    p.add_argument("--cache", default=None, help="Combined cache dir. Omit with --no_images.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--repo_root", default=None)
    p.add_argument("--n_images", type=int, default=300)
    p.add_argument("--no_images", action="store_true", help="Decoder geometry only, no cache needed.")
    p.add_argument("--dup_threshold", type=float, default=0.9,
                   help="Cosine above which two decoder columns count as duplicates.")
    p.add_argument("--top_per_feature", type=int, default=5,
                   help="How many top images to keep per feature when measuring concentration.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="feature_duplication.npz")
    return p.parse_args()


def decoder_duplication(w_dec: torch.Tensor, threshold: float, chunk: int = 512):
    """
    w_dec: (in_dim, latent_dim). Each latent feature is one column.

    Returns (max_cos_per_feature, n_pairs_over_threshold, worst_pairs).

    The full latent x latent matrix is 12288^2, so this walks it in column
    chunks and only keeps the per-feature maximum and the pair count instead of
    materialising ~600MB.
    """
    w = torch.nn.functional.normalize(w_dec.float(), dim=0)   # unit columns
    K = w.shape[1]

    max_cos = torch.full((K,), -1.0)
    argmax = torch.zeros(K, dtype=torch.long)
    n_pairs = 0

    for start in range(0, K, chunk):
        stop = min(start + chunk, K)
        block = w[:, start:stop].T @ w                     # (chunk, K)

        # Ignore each column's similarity with itself.
        rows = torch.arange(start, stop)
        block[torch.arange(stop - start), rows] = -1.0

        block_max, block_arg = block.max(dim=1)
        max_cos[start:stop] = block_max
        argmax[start:stop] = block_arg

        # Count each unordered pair once.
        n_pairs += int((block[:, :] > threshold).sum().item())

    n_pairs //= 2

    order = torch.argsort(max_cos, descending=True)[:10]
    worst_pairs = [(int(i), int(argmax[i]), float(max_cos[i])) for i in order]
    return max_cos, n_pairs, worst_pairs


def main():
    args = _parse_args()

    repo_root = args.repo_root or os.path.dirname(os.path.abspath(args.config))
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    K = ckpt["latent_dim"]

    manifest = ckpt.get("manifest")
    if manifest:
        print(f"[dup] checkpoint from code {manifest.get('code_version')}, "
              f"cache {manifest.get('cache_root')}, "
              f"timestep {manifest.get('fixed_timestep_idx')}")

    model_names = list(ckpt["model_dims"].keys())

    # ---------------------------------------------------------------- part 1
    print("\n" + "=" * 66)
    print("DECODER COLUMN DUPLICATION")
    print("=" * 66)
    print(f"  latent_dim = {K}, duplicate threshold = cos > {args.dup_threshold}")

    results = {}
    for name in model_names:
        key = f"saes.{name}.W_dec.weight"
        if key not in state:
            print(f"  {name}: no {key} in checkpoint, skipping")
            continue
        max_cos, n_pairs, worst = decoder_duplication(
            state[key], args.dup_threshold
        )
        results[f"max_cos_{name}"] = max_cos.numpy()

        print(f"\n  {name}")
        print(f"    max off-diagonal cosine : mean {max_cos.mean():.4f}  "
              f"median {max_cos.median():.4f}  p95 {np.percentile(max_cos.numpy(), 95):.4f}")
        print(f"    duplicate pairs         : {n_pairs}")
        print(f"    worst pairs             : "
              + ", ".join(f"{a}~{b} ({c:.3f})" for a, b, c in worst[:5]))

    if args.no_images:
        np.savez(args.out, **results)
        print(f"\n[dup] saved -> {args.out}")
        return

    if not args.cache:
        print("\n[dup] no --cache given, stopping after decoder geometry. "
              "Pass --cache or --no_images.")
        return

    # ---------------------------------------------------------------- part 2
    from data import CocoActivationDataset
    from universal_sae import UniversalSAE
    from spatial_align import build_spatial_aligner_from_config
    from pixart_timestep import resolve_pixart_timestep

    with open(args.config) as f:
        cfg_file = yaml.safe_load(f)
    g_file = cfg_file.get("global", {})
    sae_p = cfg_file.get("sae_params", {})
    g_ckpt = (ckpt.get("config") or {}).get("global", {})
    g_ckpt_sae = (ckpt.get("config") or {}).get("sae_params", {})

    def pick(k, default=None):
        if k in g_ckpt: return g_ckpt[k]
        if k in g_file: return g_file[k]
        return default

    model = UniversalSAE(
        model_dims=ckpt["model_dims"],
        latent_dim=K,
        diffusion_models=set(ckpt.get("diffusion_models", [])),
        model_tokens=ckpt["model_tokens"],
        top_k=int(g_ckpt_sae.get("top_k", sae_p.get("top_k", 128))),
        cls_pool_mode=str(pick("cls_pool_mode", "none")),
        use_tide=bool(pick("use_tide", False)),
        timestep_dim=int(pick("timestep_dim", 256)),
    )
    model.load_state_dict(state, strict=False)
    model = model.to(args.device).eval()

    aligner = build_spatial_aligner_from_config(
        {"spatial_align_to": ckpt.get("spatial_align_to", pick("spatial_align_to"))},
        ckpt.get("model_tokens_native", ckpt["model_tokens"]),
    )

    ds = CocoActivationDataset(
        cache_root=args.cache,
        sources=model_names,
        combined_npz=True,
        standardize=bool(pick("standardize", True)),
        divide_norm=bool(pick("divide_norm", False)),
        use_class_tokens=bool(pick("use_class_tokens", False)),
        return_metadata=True,
        diffusion_models=list(model.diffusion_models),
        standardization_stats=ckpt.get("standardization_stats"),
    )
    n_use = min(args.n_images, len(ds))

    # per-feature, per-image peak activation
    peaks = {name: torch.zeros(n_use, K) for name in model_names}

    with torch.no_grad():
        for i in range(n_use):
            (acts, meta), _ = ds[i]
            for name in model_names:
                x = acts[name].to(args.device).float()
                sigma = None
                if name in model.diffusion_models:
                    t_idx = resolve_pixart_timestep(
                        x.shape[0], ckpt=ckpt, config_global=g_file
                    )
                    sig = meta.get("sigmas_by_model", {}).get(name, meta.get("sigmas"))
                    if sig is not None:
                        sigma = sig.view(-1)[t_idx].to(args.device).float().view(1)
                    x = x[t_idx]
                x = x.unsqueeze(0)
                if aligner is not None:
                    x = aligner.align(x, source=name)
                _z_pre, z = model.encode(x, source=name, sigma=sigma)
                peaks[name][i] = z.abs().amax(dim=(0, 1)).cpu()
            if (i + 1) % 50 == 0:
                print(f"  processed {i + 1}/{n_use}")

    print("\n" + "=" * 66)
    print(f"TOP-IMAGE CONCENTRATION  ({n_use} images)")
    print("=" * 66)
    print("  If a few images own most features' top slots, those features are")
    print("  responding to global structure rather than to image content.\n")

    for name in model_names:
        P = peaks[name]
        alive = P.amax(dim=0) > 0
        n_alive = int(alive.sum())
        if n_alive == 0:
            print(f"  {name}: no alive features")
            continue

        Pa = P[:, alive]
        top_img = Pa.argmax(dim=0)                      # best image per feature
        counts = torch.bincount(top_img, minlength=n_use).float()
        share = counts.max() / n_alive
        top5 = counts.sort(descending=True).values[:5].sum() / n_alive

        active_frac = (Pa > 0).float().mean(dim=0)      # how often it fires at all

        results[f"active_frac_{name}"] = active_frac.numpy()
        results[f"top_image_counts_{name}"] = counts.numpy()

        print(f"  {name}  ({n_alive} alive features)")
        print(f"    single most-claimed image owns : {share:6.1%} of features")
        print(f"    top 5 images own               : {top5:6.1%} of features")
        print(f"    expected if uniform            : {5.0 / n_use:6.1%}")
        print(f"    active fraction per feature    : mean {active_frac.mean():.3f}  "
              f"median {active_frac.median():.3f}")
        if share > 0.10:
            print(f"    ^ CONCENTRATED: one image is the peak for >10% of features.")

    np.savez(args.out, **results)
    print(f"\n[dup] saved -> {args.out}")


if __name__ == "__main__":
    main()
