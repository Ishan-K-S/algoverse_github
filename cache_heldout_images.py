"""Cache a held-out image set that the SAE has never seen, for honest evaluation.

The feature atlas scores cached activations, not JPEGs, so "test on new images"
means running the caching pipeline on images excluded from training:

    val2017 (5000) - selected_images.txt (the 2000 used for training) -> pick N

Nothing in the resulting cache was ever trained on, so `feature_atlas.py
--cache_root <out_cache> --split all` is a genuine held-out result without
needing a retrain or a train/val stems file.

CRITICAL: the new cache must be built the same way as the training cache, or
the activations won't correspond and every number will be quietly wrong. This
script inspects the reference (training) cache and mirrors its PixArt mode:

  T == 1  -> DIFT single-timestep mode. The raw timestep is read back out of
             the reference npz's `PixArt__timesteps`. n_noise is NOT recoverable
             from the npz (it's averaged away), so it defaults to the caching
             script's own default and is printed loudly as an assumption.
  T  > 1  -> full trajectory mode with num_inference_steps = T.

Runs chunked (cache -> combine -> drop raw) so peak disk stays bounded and a
crash mid-run doesn't lose everything: already-combined stems are skipped on
re-run.

Example:
    python cache_heldout_images.py --n 1000
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache_coco_activations import cache_model_activations
from cache_coco_diffusion_activations import cache_diffusion_activations, login_to_huggingface
from combine_cached_acts import combine_activations

COCO_ROOT = "/content/coco_data/val2017"
TRAIN_SELECTION = "/content/cache/selected_images.txt"
REFERENCE_CACHE = "/content/combined_cache"
RAW_CACHE = "/content/heldout_cache_raw"
OUT_CACHE = "/content/heldout_cache"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------- #
# Mirror the training cache's PixArt settings
# --------------------------------------------------------------------------- #

def detect_pixart_mode(reference_cache: str) -> Tuple[int, Optional[int]]:
    """Return (T, raw_timestep) from a sample combined npz in the training cache.

    raw_timestep is only meaningful when T == 1 (DIFT mode); it's read from the
    stored PixArt__timesteps so the held-out cache noises to the exact same
    level the model was trained on.
    """
    files = sorted(f for f in os.listdir(reference_cache) if f.endswith("_combined.npz"))
    if not files:
        raise RuntimeError(
            f"No *_combined.npz in reference cache {reference_cache!r}; cannot "
            "determine how the training cache was built. Pass --single_timestep / "
            "--num_inference_steps explicitly if you know them."
        )
    sample = os.path.join(reference_cache, files[0])
    with np.load(sample, allow_pickle=False) as data:
        if "PixArt" not in data.files:
            raise RuntimeError(f"{sample} has no 'PixArt' key (found {data.files}).")
        act = data["PixArt"]
        T = int(act.shape[0]) if act.ndim == 3 else 1
        raw_t = None
        if T == 1 and "PixArt__timesteps" in data.files:
            ts = np.asarray(data["PixArt__timesteps"]).reshape(-1)
            if ts.size:
                raw_t = int(ts[0])
    print(f"[heldout] reference cache {os.path.basename(sample)}: PixArt shape={act.shape} -> T={T}")
    return T, raw_t


# --------------------------------------------------------------------------- #
# Held-out image selection
# --------------------------------------------------------------------------- #

def pick_heldout_images(
    coco_root: str, train_selection: str, n: int, seed: int, save_path: str
) -> List[str]:
    """N images from coco_root that are NOT in the training selection.

    Idempotent like coco_dataset_setup.select_images: if save_path exists it's
    reused, so re-runs and resumes operate on the identical set.
    """
    if os.path.isfile(save_path):
        with open(save_path) as f:
            picked = [l.strip() for l in f if l.strip()]
        print(f"[heldout] reusing {len(picked)} previously selected images from {save_path}")
        return picked

    if not os.path.isfile(train_selection):
        raise FileNotFoundError(
            f"Training selection not found: {train_selection}\n"
            "  This is the selected_images.txt written by coco_dataset_setup.select_images "
            "when the training cache was built. Without it we can't guarantee the held-out "
            "set is disjoint from training -- point --train_selection at the right file."
        )
    with open(train_selection) as f:
        trained_on = {l.strip() for l in f if l.strip()}

    all_images = sorted(
        f for f in os.listdir(coco_root) if f.lower().endswith(IMAGE_EXTS)
    )
    available = [f for f in all_images if f not in trained_on]
    print(f"[heldout] {len(all_images)} images in {coco_root}, "
          f"{len(trained_on)} used for training, {len(available)} available")

    if len(available) < n:
        raise RuntimeError(
            f"Only {len(available)} unused images available but --n {n} requested. "
            "Lower --n, or download another COCO split."
        )

    picked = sorted(random.Random(seed).sample(available, n))
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(picked) + "\n")
    print(f"[heldout] selected {len(picked)} held-out images -> {save_path}")
    return picked


def already_combined(out_cache: str) -> set:
    if not os.path.isdir(out_cache):
        return set()
    return {
        f[: -len("_combined.npz")]
        for f in os.listdir(out_cache)
        if f.endswith("_combined.npz")
    }


def clear_raw(raw_cache: str, sources: List[str]) -> int:
    """Delete only this script's own per-source npz files from its scratch dir."""
    removed = 0
    for f in os.listdir(raw_cache):
        if any(f.endswith(f"_{s}.npz") for s in sources):
            os.remove(os.path.join(raw_cache, f))
            removed += 1
    return removed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=1000, help="How many held-out images to cache")
    p.add_argument("--coco_root", default=COCO_ROOT)
    p.add_argument("--train_selection", default=TRAIN_SELECTION,
                   help="selected_images.txt used to build the TRAINING cache")
    p.add_argument("--reference_cache", default=REFERENCE_CACHE,
                   help="Training combined cache, inspected to mirror its PixArt mode")
    p.add_argument("--raw_cache", default=RAW_CACHE, help="Scratch dir for per-source npz")
    p.add_argument("--out_cache", default=OUT_CACHE, help="Where combined npz land")
    p.add_argument("--selection_file", default=None,
                   help="Default: <out_cache>/heldout_images.txt")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--sources", nargs="+", default=["DinoV2", "PixArt"])
    p.add_argument("--chunk_size", type=int, default=100,
                   help="Images per cache->combine->cleanup cycle. Bounds peak disk.")
    p.add_argument("--keep_raw", action="store_true",
                   help="Keep per-source npz after combining (roughly doubles disk).")

    p.add_argument("--single_timestep", type=int, default=None,
                   help="Override the auto-detected DIFT raw timestep.")
    p.add_argument("--num_inference_steps", type=int, default=None,
                   help="Override the auto-detected trajectory length.")
    p.add_argument("--n_noise", type=int, default=8,
                   help="Noise draws averaged per image in DIFT mode. Not recoverable "
                        "from the reference cache -- must match how training was cached.")
    p.add_argument("--no_seed_from_filenames", action="store_true",
                   help="Disable deterministic per-image noise seeding.")

    p.add_argument("--dino_batch_size", type=int, default=64)
    p.add_argument("--pixart_batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # Guard: cleanup deletes from raw_cache, so it must not be a real cache dir.
    for name, path in (("reference_cache", args.reference_cache), ("out_cache", args.out_cache)):
        if os.path.abspath(args.raw_cache) == os.path.abspath(path):
            raise ValueError(
                f"--raw_cache must not be the same directory as --{name} "
                f"({path!r}); this script deletes per-source npz from raw_cache."
            )

    # cache_coco_activations.cache_model_activations does x.cuda() internally
    # (cache_coco_activations.py:85), so fail here rather than deep in the batch loop.
    if not args.device.startswith("cuda"):
        raise RuntimeError(
            f"--device {args.device!r}: caching requires a CUDA GPU "
            "(cache_model_activations calls .cuda() unconditionally). "
            "Switch the Colab runtime to a GPU."
        )

    os.makedirs(args.raw_cache, exist_ok=True)
    os.makedirs(args.out_cache, exist_ok=True)
    selection_file = args.selection_file or os.path.join(args.out_cache, "heldout_images.txt")

    # ---- Mirror the training cache's PixArt settings ----
    T, detected_raw_t = detect_pixart_mode(args.reference_cache)
    single_timestep = args.single_timestep if args.single_timestep is not None else (
        detected_raw_t if T == 1 else None
    )
    if T == 1:
        if single_timestep is None:
            raise RuntimeError(
                "Reference cache is single-timestep (T=1) but its raw timestep couldn't be "
                "read from PixArt__timesteps. Pass --single_timestep explicitly (see "
                "pixart_timestep.resolve_pixart_raw_timestep)."
            )
        num_inference_steps = args.num_inference_steps or 15
        print(f"[heldout] MODE: DIFT single-timestep, raw t={single_timestep}, "
              f"n_noise={args.n_noise}")
        print(f"[heldout] NOTE: n_noise is not recoverable from the cache. {args.n_noise} is "
              f"assumed -- if training used a different value, pass --n_noise to match.")
    else:
        num_inference_steps = args.num_inference_steps or T
        print(f"[heldout] MODE: full trajectory, num_inference_steps={num_inference_steps}")

    # ---- Select images ----
    picked = pick_heldout_images(
        args.coco_root, args.train_selection, args.n, args.seed, selection_file
    )
    done = already_combined(args.out_cache)
    todo = [f for f in picked if os.path.splitext(f)[0] not in done]
    if done:
        print(f"[heldout] {len(done)} already combined, {len(todo)} remaining")
    if not todo:
        print("[heldout] nothing to do -- cache is complete.")
        return

    login_to_huggingface()

    print(f"[heldout] loading extractors on {args.device} ...")
    from models import DinoV2
    from DiffusionActivationExtractor import PixArtActivationExtractor

    dino = DinoV2().to(args.device).eval()
    pixart = PixArtActivationExtractor(
        device=args.device, num_inference_steps=num_inference_steps
    )

    chunks = [todo[i:i + args.chunk_size] for i in range(0, len(todo), args.chunk_size)]
    print(f"[heldout] {len(todo)} images in {len(chunks)} chunk(s) of <= {args.chunk_size}")

    for ci, chunk in enumerate(chunks, 1):
        print(f"\n[heldout] ===== chunk {ci}/{len(chunks)} ({len(chunk)} images) =====")

        cache_model_activations(
            model=dino,
            model_name="DinoV2",
            coco_root=args.coco_root,
            path_to_cache=args.raw_cache,
            batch_size=args.dino_batch_size,
            image_list=chunk,
        )

        cache_diffusion_activations(
            extractor=pixart,
            source_name="PixArt",
            coco_root=args.coco_root,
            cache_root=args.raw_cache,
            batch_size=args.pixart_batch_size,
            num_workers=args.num_workers,
            image_list=chunk,
            single_timestep=single_timestep,
            seed_from_filenames=not args.no_seed_from_filenames,
            n_noise=args.n_noise,
        )

        combine_activations(
            cache_root=args.raw_cache,
            output_root=args.out_cache,
            sources=list(args.sources),
            num_workers=args.num_workers,
        )

        if not args.keep_raw:
            removed = clear_raw(args.raw_cache, list(args.sources))
            print(f"[heldout] cleaned {removed} per-source npz from {args.raw_cache}")

    n_final = len(already_combined(args.out_cache))
    print(f"\n[heldout] done: {n_final} combined npz in {args.out_cache}")
    print("[heldout] next:")
    print(f"  python feature_atlas.py --cache_root {args.out_cache} \\")
    print(f"      --image_dir {args.coco_root} --output_dir /content/results \\")
    print(f"      --split all --top_k 5 --only_shared --max_features 100")
    print("[heldout] (--split all is legitimately held out here: nothing in this "
          "cache was trained on.)")


if __name__ == "__main__":
    main()
