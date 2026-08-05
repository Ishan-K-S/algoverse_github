"""Cache-only activation health check: no SAE checkpoint, no GPU required.

Samples n cached images and reports min/max/mean/std and mean per-token L2
norm for DinoV2 (single activation) and PixArt (broken out per timestep,
since we already know scale/behavior varies a lot across the 15 timesteps).

Also reports an "outlier ratio" (max per-image L2 norm / median per-image L2
norm) per model/timestep -- this is the number that directly tests whether a
few loud images dominate every feature, rather than requiring someone to scan
the per-image column for outliers.

Example:
    python cache_activation_diagnostic.py --n 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import CocoActivationDataset
from visualize_feature_activations import DEFAULT_CACHE

# Sources whose cached activation is (T, N, D) rather than (N, D).
# Extend here if more diffusion sources get cached alongside PixArt.
DIFFUSION_SOURCES = {"PixArt"}


class RunningStats:
    """Elementwise min/max/mean/std (Welford, Chan et al. parallel merge -- same
    formula as data.py's _compute_standardization_stats) plus a running mean of
    per-token L2 norm, accumulated batch-by-batch so memory doesn't scale with
    how many slices get folded in.
    """

    def __init__(self):
        self.n_scalars = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = float("inf")
        self.max = float("-inf")
        self.norm_sum = 0.0
        self.norm_count = 0

    def update(self, x: torch.Tensor) -> None:
        """x: (N, D) activation slice."""
        flat = x.flatten().double()
        vals = flat[torch.isfinite(flat)]
        if vals.numel() > 0:
            batch_n = vals.numel()
            batch_mean = vals.mean().item()
            batch_var = vals.var(unbiased=False).item()
            new_n = self.n_scalars + batch_n
            delta = batch_mean - self.mean
            self.mean += delta * (batch_n / new_n)
            self.M2 += batch_var * batch_n + delta ** 2 * self.n_scalars * batch_n / new_n
            self.n_scalars = new_n
            self.min = min(self.min, vals.min().item())
            self.max = max(self.max, vals.max().item())

        token_norms = x.norm(dim=-1)
        token_norms = token_norms[torch.isfinite(token_norms)]
        if token_norms.numel() > 0:
            self.norm_sum += token_norms.sum().item()
            self.norm_count += token_norms.numel()

    @property
    def std(self) -> float:
        return (self.M2 / self.n_scalars) ** 0.5 if self.n_scalars else float("nan")

    @property
    def l2norm(self) -> float:
        return self.norm_sum / self.norm_count if self.norm_count else float("nan")

    def to_dict(self) -> dict:
        return {"min": self.min, "max": self.max, "mean": self.mean,
                "std": self.std, "l2norm": self.l2norm}


def outlier_ratio(records: List[Tuple[float, str]]) -> Optional[dict]:
    """records: list of (l2norm, label) pairs, one per sampled image (or
    image+timestep). Returns max/median plus which label produced the max --
    the number and evidence for "do a few images dominate."
    """
    finite = [r for r in records if r[0] == r[0]]  # drop NaN (NaN != NaN)
    if not finite:
        return None
    norms = [r[0] for r in finite]
    med = statistics.median(norms)
    max_norm, max_label = max(finite, key=lambda r: r[0])
    ratio = (max_norm / med) if med > 0 else float("inf")
    return {"ratio": ratio, "max_norm": max_norm, "median_norm": med, "max_label": max_label}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_root", default=DEFAULT_CACHE)
    p.add_argument("--n", type=int, default=50, help="Number of images to sample.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sources", nargs="+", default=["DinoV2", "PixArt"])
    p.add_argument("--output_dir", default="activation_diagnostics")
    return p.parse_args()


def print_per_image_table(rows: List[dict], has_timestep: bool):
    if has_timestep:
        header = (f"{'stem':>14} {'t':>3} {'sigma':>10} {'min':>10} {'max':>10} "
                   f"{'mean':>10} {'std':>10} {'l2norm':>10} {'nan':>4} {'inf':>4}")
    else:
        header = (f"{'stem':>14} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} "
                   f"{'l2norm':>10} {'nan':>4} {'inf':>4}")
    print(header)
    print("-" * len(header))
    for r in rows:
        if has_timestep:
            print(f"{r['stem']:>14} {r['timestep']:>3} {r['sigma']:>10.3f} {r['min']:>10.3f} "
                  f"{r['max']:>10.3f} {r['mean']:>10.3f} {r['std']:>10.3f} {r['l2norm']:>10.3f} "
                  f"{str(r['has_nan']):>4} {str(r['has_inf']):>4}")
        else:
            print(f"{r['stem']:>14} {r['min']:>10.3f} {r['max']:>10.3f} {r['mean']:>10.3f} "
                  f"{r['std']:>10.3f} {r['l2norm']:>10.3f} {str(r['has_nan']):>4} {str(r['has_inf']):>4}")


def print_pooled_row(label: str, stats: dict, ratio_info: Optional[dict]):
    print(f"[pooled] {label:<16} min={stats['min']:>10.3f} max={stats['max']:>10.3f} "
          f"mean={stats['mean']:>10.3f} std={stats['std']:>10.3f} l2norm={stats['l2norm']:>10.3f}")
    if ratio_info is not None:
        print(f"         outlier_ratio={ratio_info['ratio']:.2f}x  "
              f"(max={ratio_info['max_norm']:.3f} @ {ratio_info['max_label']}, "
              f"median={ratio_info['median_norm']:.3f})")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    diffusion_sources = [s for s in args.sources if s in DIFFUSION_SOURCES]

    ds = CocoActivationDataset(
        cache_root=args.cache_root,
        sources=args.sources,
        combined_npz=True,
        standardize=False,
        return_metadata=True,
        diffusion_models=diffusion_sources,
        use_class_tokens=False,
    )

    n = min(args.n, len(ds.stems))
    if n < args.n:
        print(f"[diag] Requested n={args.n} but only {len(ds.stems)} stems cached -- using n={n}.")
    sampled_stems = random.Random(args.seed).sample(ds.stems, n)
    sampled_indices = [ds.stems.index(s) for s in sampled_stems]
    print(f"[diag] cache_root={args.cache_root}  n={n}  seed={args.seed}  sources={args.sources}\n")

    per_image_rows: Dict[str, List[dict]] = {s: [] for s in args.sources}
    l2_records: Dict[str, List[Tuple[float, str]]] = {s: [] for s in args.sources}
    l2_records_by_t: Dict[str, Dict[int, List[Tuple[float, str]]]] = {s: {} for s in diffusion_sources}
    pooled: Dict[str, RunningStats] = {s: RunningStats() for s in args.sources}
    pooled_by_t: Dict[str, Dict[int, RunningStats]] = {s: {} for s in diffusion_sources}

    for idx, stem in zip(sampled_indices, sampled_stems):
        (acts, meta), _ = ds[idx]
        for source in args.sources:
            act = acts[source]
            if source in diffusion_sources:
                sigmas = meta["sigmas_by_model"][source].view(-1).tolist()
                for t in range(act.shape[0]):
                    sl = act[t]
                    rs = RunningStats()
                    rs.update(sl)
                    row = rs.to_dict()
                    row.update({
                        "stem": stem, "timestep": t, "sigma": sigmas[t],
                        "has_nan": bool(torch.isnan(sl).any()),
                        "has_inf": bool(torch.isinf(sl).any()),
                    })
                    per_image_rows[source].append(row)
                    label = f"{stem}@t{t}"
                    l2_records[source].append((row["l2norm"], label))
                    l2_records_by_t[source].setdefault(t, []).append((row["l2norm"], stem))
                    pooled[source].update(sl)
                    pooled_by_t[source].setdefault(t, RunningStats()).update(sl)
            else:
                rs = RunningStats()
                rs.update(act)
                row = rs.to_dict()
                row.update({
                    "stem": stem,
                    "has_nan": bool(torch.isnan(act).any()),
                    "has_inf": bool(torch.isinf(act).any()),
                })
                per_image_rows[source].append(row)
                l2_records[source].append((row["l2norm"], stem))
                pooled[source].update(act)

    json_out: dict = {"n": n, "seed": args.seed, "cache_root": args.cache_root, "models": {}}

    for source in args.sources:
        is_diffusion = source in diffusion_sources
        print("=" * 90)
        print(f"[{source}]  ({'diffusion, per-timestep' if is_diffusion else 'vision'})")
        print("=" * 90)
        print_per_image_table(per_image_rows[source], has_timestep=is_diffusion)
        print()

        model_json: dict = {"per_image": per_image_rows[source]}

        if is_diffusion:
            per_t_summary = {}
            for t in sorted(l2_records_by_t[source]):
                stats = pooled_by_t[source][t].to_dict()
                ratio_info = outlier_ratio(l2_records_by_t[source][t])
                print_pooled_row(f"t={t}", stats, ratio_info)
                per_t_summary[t] = {"stats": stats, "outlier_ratio": ratio_info}
            grand_stats = pooled[source].to_dict()
            grand_ratio = outlier_ratio(l2_records[source])
            print_pooled_row("ALL (grand total)", grand_stats, grand_ratio)
            model_json["pooled_by_timestep"] = per_t_summary
            model_json["pooled_grand_total"] = {"stats": grand_stats, "outlier_ratio": grand_ratio}
        else:
            stats = pooled[source].to_dict()
            ratio_info = outlier_ratio(l2_records[source])
            print_pooled_row(source, stats, ratio_info)
            model_json["pooled"] = {"stats": stats, "outlier_ratio": ratio_info}

        json_out["models"][source] = model_json
        print()

    out_path = os.path.join(args.output_dir, f"cache_diagnostics_n{n}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"[diag] saved -> {out_path}")


if __name__ == "__main__":
    main()
