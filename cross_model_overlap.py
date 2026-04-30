"""cross-model feature overlap: dinov2 vs pixart in universalsae latent space."""

# ---- edit these ----
REPO_ROOT       = "/content/algoverse_github"
CACHE_ROOT      = "/content/cache"
CONFIG_PATH     = "/content/algoverse_github/config.yaml"
CHECKPOINT_PATH = "/content/checkpoints/usae_epoch_29.pth"
CKPT_SEARCH_ROOT = "/content"

SOURCES             = ["DinoV2", "PixArt"]
TOP_K               = 64
MAX_IMAGES          = 500
PIXART_TIMESTEP_IDX = -1
DEVICE           = "cuda"
OUT_DIR          = "/content/overlap_results"
# --------------------

import os, sys, time
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from collections import Counter

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import CocoActivationDataset
from universal_sae import UniversalSAE


# walk a tree and list every checkpoint file with size + mtime
def find_checkpoints(root, extensions=(".pt", ".ckpt", ".pth")):
    if not os.path.isdir(root):
        print(f"root does not exist: {root}")
        return []
    hits = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(extensions):
                hits.append(os.path.join(dirpath, f))
    hits.sort()
    if not hits:
        print(f"no .pt/.ckpt/.pth files under {root}")
        return hits
    print(f"{len(hits)} candidate(s) under {root}:")
    print(f"  {'SIZE (MB)':>10}   {'MODIFIED':<20}   PATH")
    for p in hits:
        st = os.stat(p)
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
        print(f"  {st.st_size / 1048576:>10.2f}   {mtime:<20}   {p}")
    return hits


# rebuild universalsae from ckpt fields, fill rest from config.yaml
def load_universal_sae(ckpt_path, config_path, device):
    print(f"loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    with open(config_path, "r") as f:
        cfg_file = yaml.safe_load(f)
    g_file = cfg_file.get("global", {})
    sae_p  = cfg_file.get("sae_params", {})
    g_ckpt = (ckpt.get("config") or {}).get("global", {}) if isinstance(ckpt.get("config"), dict) else {}

    # ckpt config wins, then yaml, then default
    def pick(key, default=None):
        if key in g_ckpt: return g_ckpt[key]
        if key in g_file: return g_file[key]
        return default

    model = UniversalSAE(
        model_dims=ckpt["model_dims"],
        latent_dim=ckpt["latent_dim"],
        diffusion_models=set(ckpt.get("diffusion_models", g_file.get("diffusion_models", []))),
        model_tokens=ckpt["model_tokens"],
        shared_latent_tokens=ckpt["shared_latent_tokens"],
        timestep_dim=int(pick("timestep_dim", 256)),
        top_k=int(sae_p.get("top_k", pick("top_k", TOP_K))),
        topk_temperature=float(pick("topk_temperature", 0.1)),
        use_soft_topk=bool(pick("use_soft_topk", False)),
        interpolation_mode=str(pick("interpolation_mode", "bilinear")),
        token_reshape_mode=str(pick("token_reshape_mode", "attention")),
        attention_heads=int(pick("attention_heads", 8)),
        attention_dropout=float(pick("attention_dropout", 0.0)),
    )

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:    print(f"missing keys: {len(missing)}")
    if unexpected: print(f"unexpected keys: {len(unexpected)}")

    return model.to(device).eval(), cfg_file


# encode one image, score features by mean |z| across tokens, return top-k indices
@torch.no_grad()
def top_feature_set(model, x, source, sigma, top_k, device):
    x = x.to(device).unsqueeze(0).float()
    if sigma is not None:
        sigma = sigma.to(device).float().view(1)
    _z_pre, z = model.encode(x, source=source, sigma=sigma)
    scores = z.abs().mean(dim=(0, 1))
    _, idx = torch.topk(scores, k=min(top_k, scores.numel()))
    return idx.cpu().numpy()


# set jaccard over two index arrays
def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    u = sa | sb
    return len(sa & sb) / len(u) if u else 0.0


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    # show available checkpoints, then load the configured one
    print("=" * 60)
    find_checkpoints(CKPT_SEARCH_ROOT)
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"CHECKPOINT_PATH not found: {CHECKPOINT_PATH}")

    device = DEVICE if (DEVICE != "cuda" or torch.cuda.is_available()) else "cpu"
    model, cfg = load_universal_sae(CHECKPOINT_PATH, CONFIG_PATH, device)
    g = cfg.get("global", {})

    # combined npz so we can pull pixart sigmas from metadata
    ds = CocoActivationDataset(
        cache_root=CACHE_ROOT,
        sources=SOURCES,
        combined_npz=True,
        standardize=bool(g.get("standardize", True)),
        divide_norm=bool(g.get("divide_norm", False)),
        use_class_tokens=bool(g.get("use_class_tokens", False)),
        return_metadata=True,
        diffusion_models=list(model.diffusion_models),
    )

    n_use = len(ds) if MAX_IMAGES is None else min(MAX_IMAGES, len(ds))
    print(f"using {n_use}/{len(ds)} images")

    jaccard_scores = []
    count_dino, count_pixart, count_both = Counter(), Counter(), Counter()

    t0 = time.time()
    for i in range(n_use):
        (acts, meta), _ = ds[i]
        x_dino   = acts["DinoV2"]
        x_pixart = acts["PixArt"]

        # pick one diffusion timestep slice
        T = x_pixart.shape[0]
        t_idx = PIXART_TIMESTEP_IDX if PIXART_TIMESTEP_IDX >= 0 else T + PIXART_TIMESTEP_IDX
        x_pixart_slice = x_pixart[t_idx]

        # matching sigma for that timestep
        sig_map = meta.get("sigmas_by_model", {})
        sigmas_pixart = sig_map.get("PixArt", meta.get("sigmas"))
        if sigmas_pixart is None:
            raise KeyError("No PixArt sigma in metadata")
        sigma_pixart = sigmas_pixart.view(-1)[t_idx]

        top_dino   = top_feature_set(model, x_dino,         "DinoV2", None,         TOP_K, device)
        top_pixart = top_feature_set(model, x_pixart_slice, "PixArt", sigma_pixart, TOP_K, device)

        # per-image overlap + running per-feature counts
        jaccard_scores.append(jaccard(top_dino, top_pixart))
        count_dino.update(top_dino.tolist())
        count_pixart.update(top_pixart.tolist())
        count_both.update(set(top_dino.tolist()) & set(top_pixart.tolist()))

        if (i + 1) % 25 == 0 or (i + 1) == n_use:
            print(f"  [{i+1}/{n_use}] mean Jaccard = {np.mean(jaccard_scores):.4f}  "
                  f"({time.time()-t0:.1f}s)")

    jaccard_scores = np.array(jaccard_scores)
    print(f"mean={jaccard_scores.mean():.4f}  median={np.median(jaccard_scores):.4f}  "
          f"std={jaccard_scores.std():.4f}  min={jaccard_scores.min():.4f}  max={jaccard_scores.max():.4f}")

    top20 = count_both.most_common(20)
    print("\nTop-20 co-active features:")
    for rank, (fid, cnt) in enumerate(top20, 1):
        print(f"  {rank:2d}. feat {fid:<6d}  both={cnt:4d}  "
              f"dino={count_dino.get(fid,0):4d}  pixart={count_pixart.get(fid,0):4d}")

    # raw arrays for downstream use
    np.savez(
        os.path.join(OUT_DIR, "overlap_stats.npz"),
        jaccard_scores=jaccard_scores,
        both_features=np.array([f for f, _ in top20], dtype=np.int64),
        both_counts=np.array([c for _, c in top20], dtype=np.int64),
    )

    # histogram of per-image jaccard
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.hist(jaccard_scores, bins=40, edgecolor="black", alpha=0.85)
    ax1.axvline(jaccard_scores.mean(), color="red", linestyle="--",
                label=f"mean = {jaccard_scores.mean():.3f}")
    ax1.set_xlabel(f"Per-image Jaccard(top-{TOP_K} DinoV2, top-{TOP_K} PixArt)")
    ax1.set_ylabel("Image count")
    ax1.set_title("Cross-model top-k feature overlap")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUT_DIR, "jaccard_histogram.png"), dpi=150)
    plt.show()

    # top-20 universally co-active features
    if top20:
        ids, cnts = [str(f) for f, _ in top20], [c for _, c in top20]
        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        ax2.bar(ids, cnts, edgecolor="black", alpha=0.85)
        ax2.set_xlabel("SAE feature index")
        ax2.set_ylabel("Images with feature in BOTH top-k")
        ax2.set_title(f"Top-20 universally co-activated features (N={len(jaccard_scores)})")
        ax2.tick_params(axis="x", rotation=45)
        fig2.tight_layout()
        fig2.savefig(os.path.join(OUT_DIR, "top20_coactive_features.png"), dpi=150)
        plt.show()

    return jaccard_scores, top20


if __name__ == "__main__":
    run()
