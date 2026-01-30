"""
Minimal entrypoint replacing uni_demo.py (no overcomplete dependency).

Expected config structure is compatible with your existing config.yaml's `global` section,
with two additions:
  - global.sources: list of sources in combined NPZ files (e.g., [ViT, CLIP, FLUX])
  - global.sigma_source: which diffusion source to use for sigma schedule (e.g., FLUX)

You can keep your existing config.yaml and just add these fields.
"""

import yaml
from usae.training.train_temporal import train


def main(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    g = cfg.get("global", {})

    imagenet_root = g["imagenet_root"]
    activation_root = g["path_to_cache"]
    sources = g.get("sources") or list(cfg.get("model_zoo", {}).keys())
    sigma_source = g.get("sigma_source", None)
    if sigma_source is None:
        # best-effort: pick the first source that looks like diffusion (present in metadata keys in NPZ)
        # user should set it explicitly.
        sigma_source = sources[-1]

    train(
        imagenet_root=imagenet_root,
        activation_root=activation_root,
        sources=sources,
        sigma_source=sigma_source,
        latent_dim=int(g.get("latent_dim", 16384)),
        batch_size=int(g.get("batch_size", 8)),
        num_workers=int(g.get("num_workers", 8)),
        lr=float(g.get("lr", 1e-4)),
        nb_epochs=int(g.get("nb_epochs", 1)),
        device=str(g.get("device", "cuda")),
        use_class_tokens=bool(g.get("use_class_tokens", False)),
        standardize=bool(g.get("standardize", True)),
        divide_norm=bool(g.get("divide_norm", False)),
        top_k=int(cfg.get("sae_params", {}).get("top_k", 32)),
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    args = p.parse_args()
    main(args.config)
