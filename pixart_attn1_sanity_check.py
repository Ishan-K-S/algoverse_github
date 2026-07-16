"""One-image sanity check: does hooking attn1 (self-attention) instead of
attn2 (cross-attention to the null prompt) fix PixArt's flat/uniform
per-patch activations?

The existing cache was built with the old attn2 hook, so this runs
PixArtActivationExtractor fresh on a single raw image (bypassing the cache
entirely) and feeds the resulting (T, N, D) activations through the same
localization diagnostic as pixart_timestep_autopsy.py. DinoV2's reference
activation is reused from the existing combined cache since its extraction
is unaffected by this change.

Example:
    python pixart_attn1_sanity_check.py --stem 000000562818 \
        --raw_image_dir /content/coco_data/val2017
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spatial_align import build_spatial_aligner_from_config, infer_grid_size
from data import CocoActivationDataset
from DiffusionActivationExtractor import PixArtActivationExtractor
from visualize_feature_activations import (
    DEFAULT_CACHE,
    DEFAULT_CONFIG,
    DEFAULT_WEIGHTS_DIR,
    find_latest_checkpoint,
    find_raw_image,
    load_checkpoint,
    resolve_stem,
)
from pixart_timestep_autopsy import encode_slice, localization, peakiness_of_topk, save_grid

EPS = 1e-8


def raw_token_stats(x: torch.Tensor) -> dict:
    """Localization stats computed directly on raw (pre-SAE) activations.

    x: (N, D) single-image activation, any source.
    Unlike `localization()` in pixart_timestep_autopsy.py, this does NOT rank
    features by their own score before reporting -- it averages over every
    channel, so there's no selection bias inflating the number.

    Returns:
      mean_pairwise_cos : average cosine similarity between distinct tokens.
                           ~1.0 means every patch is (almost) the same vector,
                           i.e. no spatial information reached this point.
      channel_peakiness : mean over channels of max_token(|v|)/mean_token(|v|).
                           ~1.0 means every channel is flat across tokens.
    """
    v = x.float()
    vn = torch.nn.functional.normalize(v, dim=-1)
    sim = vn @ vn.T
    n = sim.shape[0]
    off_diag = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    mean_pairwise_cos = sim[off_diag].mean().item()

    absv = v.abs()
    mean_c = absv.mean(dim=0)
    max_c = absv.max(dim=0).values
    channel_peakiness = (max_c / (mean_c + EPS)).mean().item()
    return {"mean_pairwise_cos": mean_pairwise_cos, "channel_peakiness": channel_peakiness}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", required=True, help="COCO stem, e.g. '000000562818'.")
    p.add_argument("--raw_image_dir", required=True,
                   help="Directory containing <stem>.jpg -- needed to re-run PixArt from scratch.")
    p.add_argument("--ref_source", default="DinoV2")
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--num_inference_steps", type=int, default=15,
                   help="Must match how the existing cache was built (cache_coco_diffusion_activations.py).")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--weights_dir", default=DEFAULT_WEIGHTS_DIR)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--cache_root", default=DEFAULT_CACHE,
                   help="Only used to fetch the DinoV2 reference activation.")
    p.add_argument("--output_dir", default="pixart_attn1_sanity")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = args.ckpt
    if not ckpt_path or not os.path.isfile(ckpt_path):
        ckpt_path = find_latest_checkpoint(args.weights_dir)
    print(f"[sanity] checkpoint : {ckpt_path}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    g = cfg["global"]

    model, model_tokens_native = load_checkpoint(ckpt_path, cfg)
    model.eval().to(args.device)
    aligner = build_spatial_aligner_from_config(g, model_tokens_native)

    # ---- DinoV2 reference: reused from the existing cache (unaffected by this change) ----
    ds = CocoActivationDataset(
        cache_root=args.cache_root,
        sources=[args.ref_source],
        combined_npz=True,
        standardize=bool(g.get("standardize", True)),
        return_metadata=True,
        diffusion_models=[],
        use_class_tokens=False,
    )
    stem = resolve_stem(args.stem, ds.stems)
    (acts, _meta), _ = ds[ds.stems.index(stem)]

    raw_path = find_raw_image(stem, args.raw_image_dir)
    if raw_path is None:
        raise FileNotFoundError(f"No raw image for stem {stem!r} under {args.raw_image_dir}")
    base_image = Image.open(raw_path).convert("RGB")

    z_ref = encode_slice(model, acts[args.ref_source].unsqueeze(0), args.ref_source, None, aligner, args.device)
    ref_grid = infer_grid_size(z_ref.shape[1])
    ref_stats = localization(z_ref)
    ref_peak, ref_ids = peakiness_of_topk(ref_stats, args.top_k, by="mean")
    save_grid(base_image, z_ref, ref_ids, ref_grid,
              os.path.join(args.output_dir, f"{stem}_{args.ref_source}_reference.png"), args.top_k)
    print(f"\n[sanity] REFERENCE {args.ref_source}: top-k peakiness = {ref_peak:.2f}\n")

    ref_raw = raw_token_stats(acts[args.ref_source])
    print(f"[sanity] REFERENCE {args.ref_source} RAW (pre-SAE): "
          f"mean_pairwise_cos={ref_raw['mean_pairwise_cos']:.3f}  "
          f"channel_peakiness={ref_raw['channel_peakiness']:.2f}\n")

    # ---- PixArt: extract fresh, with the attn1 hook ----
    print("[sanity] loading PixArt pipeline (this takes a while)...")
    extractor = PixArtActivationExtractor(device=args.device, num_inference_steps=args.num_inference_steps)
    assert extractor.HOOK_ATTN_NAME == "attn1", (
        f"Expected PixArtActivationExtractor.HOOK_ATTN_NAME == 'attn1', got {extractor.HOOK_ATTN_NAME!r}. "
        "Did the patch land?"
    )

    x = extractor.preprocess(base_image).unsqueeze(0)
    out = extractor.extract_activations(x)
    a = torch.stack([act.detach().cpu() for act in out.activations], dim=0)  # (T, B, N, D)
    a = a[:, 0]  # (T, N, D)
    sigmas = out.sigmas
    T = a.shape[0]
    print(f"[sanity] PixArt(attn1) activations shape = {tuple(a.shape)}  (T={T} timesteps)\n")

    header = (f"{'t':>3} {'sigma':>10} {'n_active':>9} {'peak(mean-rank)':>16} {'peak(localized)':>16} "
              f"{'raw_cos':>9} {'raw_chan_peak':>13}")
    print(header)
    print("-" * len(header))

    rows = []
    best = {"peak_localized": -1.0, "t": None}
    for t in range(T):
        sigma_t = torch.tensor([sigmas[t]], device=args.device)
        z_t = encode_slice(model, a[t].unsqueeze(0), "PixArt", sigma_t, aligner, args.device)
        grid = infer_grid_size(z_t.shape[1])
        stats = localization(z_t)
        peak_mean, mean_ids = peakiness_of_topk(stats, args.top_k, by="mean")
        peak_loc, loc_ids = peakiness_of_topk(stats, args.top_k, by="localized")
        n_active = int(stats["active"].sum().item())
        raw = raw_token_stats(a[t])

        print(f"{t:>3} {sigmas[t]:>10.3f} {n_active:>9} {peak_mean:>16.2f} {peak_loc:>16.2f} "
              f"{raw['mean_pairwise_cos']:>9.3f} {raw['channel_peakiness']:>13.2f}")
        save_grid(base_image, z_t, loc_ids, grid,
                  os.path.join(args.output_dir, f"{stem}_PixArt_attn1_t{t}_localized.png"), args.top_k)

        rows.append({
            "timestep": t, "sigma": sigmas[t], "n_active": n_active,
            "peakiness_mean_ranked": peak_mean, "peakiness_localized": peak_loc,
            "top_mean_features": mean_ids, "top_localized_features": loc_ids,
            "raw_mean_pairwise_cos": raw["mean_pairwise_cos"], "raw_channel_peakiness": raw["channel_peakiness"],
        })
        if peak_loc > best["peak_localized"]:
            best = {"peak_localized": peak_loc, "peak_mean": peak_mean, "t": t,
                    "sigma": sigmas[t], "loc_ids": loc_ids, "mean_ids": mean_ids,
                    "z": z_t, "grid": grid, "raw": raw}

    bt = best["t"]
    save_grid(base_image, best["z"], best["mean_ids"], best["grid"],
              os.path.join(args.output_dir, f"{stem}_PixArt_attn1_t{bt}_meanranked.png"), args.top_k)

    # ---- Verdict ----
    # peakiness has a mathematical floor of 1.0 (max_token/mean_token >= 1), so
    # compare excess-over-uniform rather than the raw value -- see the bug in
    # pixart_timestep_autopsy.py's ref_thresh (0.5 * ref_peak can fall below the
    # floor and make its middle branch unreachable).
    ref_thresh = 1.0 + 0.5 * (ref_peak - 1.0)
    best_mean = best["peak_mean"]
    best_raw = best["raw"]
    print("\n" + "=" * 60)
    print(f"[verdict] reference ({args.ref_source}) peakiness         : {ref_peak:.2f}")
    print(f"[verdict] reference ({args.ref_source}) raw mean_pairwise_cos : {ref_raw['mean_pairwise_cos']:.3f}")
    print(f"[verdict] best PixArt(attn1) timestep                    : t={bt} (sigma={best['sigma']:.3f})")
    print(f"[verdict]   peakiness, localized-ranked                  : {best['peak_localized']:.2f}")
    print(f"[verdict]   peakiness, mean-ranked                       : {best_mean:.2f}")
    print(f"[verdict]   raw mean_pairwise_cos                        : {best_raw['mean_pairwise_cos']:.3f}")
    print(f"[verdict]   raw channel_peakiness                        : {best_raw['channel_peakiness']:.2f}")

    sae_localizes = best_mean >= ref_thresh
    # Raw signal counts as "carrying spatial structure" if its token-to-token
    # variation is in the same ballpark as DinoV2's -- i.e. tokens are not
    # near-duplicates of each other. Pure cosine similarity has no natural
    # zero-floor issue (unlike peakiness), so compare directly to the reference.
    raw_carries_signal = best_raw["mean_pairwise_cos"] <= ref_raw["mean_pairwise_cos"] + 0.15

    if sae_localizes:
        verdict = ("IMPROVED: mean-ranked (real usage) features now localize close to the "
                   "DinoV2 baseline -- the attn2->attn1 hook change looks like the fix. "
                   "Proceed to re-cache PixArt with this hook and re-run cross-recon.")
    elif raw_carries_signal:
        verdict = ("SAE/ALIGNMENT ISSUE: the raw attn1 activation is NOT token-invariant "
                   f"(mean_pairwise_cos={best_raw['mean_pairwise_cos']:.3f} vs DinoV2's "
                   f"{ref_raw['mean_pairwise_cos']:.3f}) -- real per-patch structure exists "
                   "upstream. The SAE encoder / spatial_align path for PixArt is what's "
                   "collapsing it. Look at spatial_align.py's handling of diffusion sources "
                   "and the SAE's PixArt-specific training weight, not the extractor.")
    else:
        verdict = ("REAL UPSTREAM FAILURE: even the raw, pre-SAE activation is nearly "
                   f"token-invariant (mean_pairwise_cos={best_raw['mean_pairwise_cos']:.3f}, "
                   f"channel_peakiness={best_raw['channel_peakiness']:.2f}, both close to the "
                   "no-signal floor). Neither attn1 nor attn2 carries per-patch identity at "
                   f"block depth_frac={PixArtActivationExtractor.HOOK_DEPTH_FRAC:.3f}. Try a "
                   "shallower/later block depth, or treat this as a genuine PixArt/DiT "
                   "limitation and swap to SD-Turbo/SD1.5.")
    print(f"\n[verdict] {verdict}")
    print("=" * 60)

    with open(os.path.join(args.output_dir, f"sanity_{stem}.json"), "w") as f:
        json.dump({
            "stem": stem, "hook": "attn1", "checkpoint": ckpt_path,
            "reference_peakiness": ref_peak, "reference_raw": ref_raw,
            "best_timestep": bt, "best_sigma": best["sigma"], "best_raw": best_raw,
            "verdict": verdict, "per_timestep": rows,
        }, f, indent=2)
    print(f"\n[sanity] saved -> {args.output_dir}/  (grids + sanity_{stem}.json)")


if __name__ == "__main__":
    main()
