"""cross-model feature overlap: dinov2 vs pixart in universalsae latent space."""

# ---- edit these ----
REPO_ROOT       = "/content/algoverse_github"
CACHE_ROOT      = "/content/combined_cache"
CONFIG_PATH     = "/content/algoverse_github/config.yaml"
WEIGHTS_DIR     = "/content/algoverse_github/weights"   # searched automatically for the latest checkpoint
CHECKPOINT_PATH = "/content/usae_epoch_29.pth"  # fallback if auto-search fails
CKPT_SEARCH_ROOT = "/content"

SOURCES             = ["DinoV2", "PixArt"]
TOP_K               = 64
MAX_IMAGES          = 2000
PIXART_TIMESTEP_IDX = None  # None = take it from the checkpoint, see pixart_timestep.py
DEVICE           = "cuda"
OUT_DIR          = "/content/overlap_results"
DRIVE_SAVE_DIR   = "/content/drive/My Drive/algoverse_inference_results"
# --------------------

import os, sys, time, glob, shutil
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from collections import Counter

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import CocoActivationDataset
from pixart_timestep import resolve_pixart_timestep
from universal_sae import UniversalSAE
from spatial_align import build_spatial_aligner_from_config
from feature_usage import compute_feature_usage


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


# return the most recently modified .pth file under a directory
def find_latest_checkpoint(weights_dir):
    candidates = glob.glob(os.path.join(weights_dir, "**", "*.pth"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoints found under {weights_dir}")
    latest = max(candidates, key=os.path.getmtime)
    print(f"[ckpt] Auto-selected: {latest}")
    return latest


# copy the overlap result files to google drive so they survive runtime resets
def save_to_drive(out_dir, drive_dir):
    try:
        from google.colab import drive as _colab_drive
        if not os.path.isdir("/content/drive/My Drive"):
            print("[drive] Mounting Google Drive...")
            _colab_drive.mount("/content/drive")
    except ImportError:
        print("[drive] Not running in Colab — skipping Drive upload.")
        return

    os.makedirs(drive_dir, exist_ok=True)
    if not os.path.isdir(out_dir):
        print(f"[drive] Warning: {out_dir} not found — skipping.")
        return
    for fname in os.listdir(out_dir):
        src = os.path.join(out_dir, fname)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(drive_dir, fname)
        shutil.copy2(src, dst)
        print(f"[drive] Saved: {dst}")


# rebuild universalsae from ckpt fields, fill rest from config.yaml
def load_universal_sae(ckpt_path, config_path, device):
    print(f"loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    with open(config_path, "r") as f:
        cfg_file = yaml.safe_load(f)
    g_file = cfg_file.get("global", {})
    sae_p  = cfg_file.get("sae_params", {})
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    g_ckpt = ckpt_cfg.get("global", {})
    sae_p_ckpt = ckpt_cfg.get("sae_params", {})

    # ckpt config wins, then yaml, then default
    def pick(key, default=None):
        if key in g_ckpt: return g_ckpt[key]
        if key in g_file: return g_file[key]
        return default

    # The ckpt stores POST-alignment token counts under "model_tokens" and
    # pre-alignment counts under "model_tokens_native" (if it was saved by
    # the patched uni_demo.py). Falls back to "model_tokens" for old ckpts.
    model_tokens_effective = ckpt["model_tokens"]
    model_tokens_native = ckpt.get("model_tokens_native", model_tokens_effective)

    model = UniversalSAE(
        model_dims=ckpt["model_dims"],
        latent_dim=ckpt["latent_dim"],
        diffusion_models=set(ckpt.get("diffusion_models", g_file.get("diffusion_models", []))),
        model_tokens=model_tokens_effective,
        timestep_dim=int(pick("timestep_dim", 256)),
        # Checkpoint's own sae_params wins -- this sets the model's actual TopK width,
        # which must match what it was trained with.
        top_k=int(sae_p_ckpt.get("top_k", sae_p.get("top_k", pick("top_k", TOP_K)))),
        cls_pool_mode=str(pick("cls_pool_mode", "none")),
        use_tide=bool(pick("use_tide", False)),
    )

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:    print(f"missing keys: {len(missing)}")
    if unexpected: print(f"unexpected keys: {len(unexpected)}")

    # Build the spatial aligner from saved config so eval matches training.
    align_to = ckpt.get("spatial_align_to", pick("spatial_align_to", None))
    aligner_cfg = {"spatial_align_to": align_to}
    aligner = build_spatial_aligner_from_config(aligner_cfg, model_tokens_native)
    if aligner is not None:
        print(f"[overlap] spatial alignment ON: target grid {aligner.target_grid_size}x{aligner.target_grid_size}")
    else:
        print(f"[overlap] spatial alignment OFF")

    return model.to(device).eval(), cfg_file, aligner


# encode one image, score features by mean |z| across tokens, return top-k indices
@torch.no_grad()
def top_feature_set(model, x, source, sigma, top_k, device, aligner=None):
    x = x.to(device).unsqueeze(0).float()
    if aligner is not None:
        x = aligner.align(x, source=source)
    if sigma is not None:
        sigma = sigma.to(device).float().view(1)
    _z_pre, z = model.encode(x, source=source, sigma=sigma)
    scores = z.abs().amax(dim=(0, 1))
    # feature_usage.compute_feature_usage's "top_k_per_sample" criterion
    # (REPAIR_PLAN.md V16/Fix 3.2), applied to this single image treated as a
    # one-row batch -- same set of indices torch.topk would give (the caller
    # immediately converts to a set(), so index order doesn't matter), but
    # sharing one implementation with dictionary_diagnostic.py instead of two
    # independently-written top-k selections that could silently drift apart.
    used_mask = compute_feature_usage(scores.unsqueeze(0), criterion="top_k_per_sample", top_k=top_k)
    idx = used_mask.nonzero(as_tuple=True)[0]
    return idx.cpu().numpy()


# set jaccard over two index arrays
def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    u = sa | sb
    return len(sa & sb) / len(u) if u else 0.0


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    # path diagnostics
    print("=" * 60)
    print(f"[overlap] cache_root  : {CACHE_ROOT}  exists={os.path.isdir(CACHE_ROOT)}")
    print(f"[overlap] config      : {CONFIG_PATH}  exists={os.path.isfile(CONFIG_PATH)}")
    print(f"[overlap] checkpoint  : {CHECKPOINT_PATH}  exists={os.path.isfile(CHECKPOINT_PATH)}")
    if not os.path.isdir(CACHE_ROOT):
        raise FileNotFoundError(
            f"[overlap] Combined cache not found: {CACHE_ROOT}\n"
            "  Run combine_cached_acts.py first."
        )
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(f"[overlap] config.yaml not found: {CONFIG_PATH}")

    # show available checkpoints, then load the configured one (auto-search if missing)
    find_checkpoints(CKPT_SEARCH_ROOT)
    checkpoint = CHECKPOINT_PATH
    if not os.path.isfile(checkpoint):
        print(f"[overlap] Checkpoint not found at {checkpoint!r}, searching {WEIGHTS_DIR}...")
        checkpoint = find_latest_checkpoint(WEIGHTS_DIR)

    device = DEVICE if (DEVICE != "cuda" or torch.cuda.is_available()) else "cpu"
    model, cfg, aligner = load_universal_sae(checkpoint, CONFIG_PATH, device)
    g = cfg.get("global", {})

    # Reuse the training stats rather than recomputing them off 1000 npz files,
    # which is hours when the cache is on a Drive mount (and gives you slightly
    # different stats than the model was trained with).
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_stats = ckpt.get("standardization_stats")

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
        standardization_stats=ckpt_stats,
        # Only used if ckpt_stats is None and this falls back to recomputing:
        # training fits stats to the POOLED distribution (REPAIR_PLAN.md V6/Fix 2.2),
        # so the fallback has to pool too or it standardises with a different std.
        spatial_aligner=aligner,
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

        # pick one diffusion timestep slice, the same one the model trained on
        T = x_pixart.shape[0]
        t_idx = resolve_pixart_timestep(
            T, ckpt=ckpt, config_global=g, override=PIXART_TIMESTEP_IDX
        )
        if i == 0:
            print(f"[overlap] PixArt timestep {t_idx} of {T}")
        x_pixart_slice = x_pixart[t_idx]

        # matching sigma for that timestep
        sig_map = meta.get("sigmas_by_model", {})
        sigmas_pixart = sig_map.get("PixArt", meta.get("sigmas"))
        if sigmas_pixart is None:
            raise KeyError("No PixArt sigma in metadata")
        sigma_pixart = sigmas_pixart.view(-1)[t_idx]

        top_dino   = top_feature_set(model, x_dino,         "DinoV2", None,         TOP_K, device, aligner=aligner)
        top_pixart = top_feature_set(model, x_pixart_slice, "PixArt", sigma_pixart, TOP_K, device, aligner=aligner)

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

    save_to_drive(OUT_DIR, DRIVE_SAVE_DIR)

    return jaccard_scores, top20


if __name__ == "__main__":
    run()
