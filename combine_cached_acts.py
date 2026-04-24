# combine_cached_acts.py
"""
Combine per-model cached activation .npz files into one combined .npz per image.

Expects a FLAT cache directory (no class subdirectories), matching the layout
written by cache_coco_activations.py and cache_coco_diffusion_activations.py.

Input layout:
  <cache_root>/<img_stem>_ViT.npz
  <cache_root>/<img_stem>_DinoV2.npz
  <cache_root>/<img_stem>_SD3.npz
  ...

Output layout:
  <output_root>/<img_stem>_combined.npz

Combined npz keys:
  <MODEL>            – activation array (N,D) for vision or (T,N,D) for diffusion
  <MODEL>__sigmas    – (T,) float32 array, diffusion models only
  <MODEL>__timesteps – (T,) int64 array,  diffusion models only
"""

import os
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

DIFF_META_FIELDS = ("sigmas", "timesteps")


def _process_one_image(args):
    """Combine all source npz files for a single image stem."""
    stem, cache_root, output_root, sources = args

    combined = {}
    for source in sources:
        act_path = os.path.join(cache_root, f"{stem}_{source}.npz")
        if not os.path.exists(act_path):
            print(f"[warn] Missing: {act_path} — skipping this image.")
            return   # skip the whole image if any source is missing

        data = np.load(act_path, allow_pickle=False)

        # Store the main activation tensor under the model name key
        if "activation" in data.files:
            combined[source] = data["activation"]
        else:
            # Fallback: store whatever keys are present prefixed by source name
            for k in data.files:
                if k not in DIFF_META_FIELDS and k != "filename":
                    combined[f"{source}_{k}"] = data[k]

        # Store diffusion metadata with double-underscore separator
        for mf in DIFF_META_FIELDS:
            if mf in data.files:
                combined[f"{source}__{mf}"] = data[mf]

    out_path = os.path.join(output_root, f"{stem}_combined.npz")
    np.savez_compressed(out_path, **combined)


def combine_activations(
    cache_root: str,
    output_root: str,
    sources: list,
    num_workers: int = 8,
):
    """
    Combine per-source .npz files into one combined .npz per image.

    Parameters
    ----------
    cache_root  : directory containing all per-source .npz files
    output_root : directory where combined .npz files will be written
    sources     : ordered list of model names, e.g. ["ViT", "DinoV2", "SD3"]
    num_workers : multiprocessing worker count
    """
    os.makedirs(output_root, exist_ok=True)

    # Discover image stems from the first source
    anchor_suffix = f"_{sources[0]}.npz"
    all_files = os.listdir(cache_root)
    stems = sorted(
        f[: -len(anchor_suffix)]
        for f in all_files
        if f.endswith(anchor_suffix)
    )

    if not stems:
        raise RuntimeError(
            f"No files ending in '{anchor_suffix}' found in '{cache_root}'. "
            "Check that caching has been completed for the first source."
        )

    print(f"[combine] Found {len(stems)} images (anchor source: {sources[0]})")

    args_list = [(stem, cache_root, output_root, sources) for stem in stems]

    if num_workers > 1:
        with Pool(num_workers) as pool:
            list(tqdm(
                pool.imap(_process_one_image, args_list),
                total=len(stems),
                desc="Combining activations",
            ))
    else:
        for args in tqdm(args_list, desc="Combining activations"):
            _process_one_image(args)

    print("[combine] Done.")


if __name__ == "__main__":
    cache_root  = "/lambda/nfs/AlgoverseResearchAIJK/cache_path"
    output_root = "/lambda/nfs/AlgoverseResearchAIJK/combined_path"

    colab_cache_root = "/content/cache"
    colab_output_cache = "/content/combined_cache"

    # List every source you cached, in any order
    sources = ["ViT", "DinoV2", "SigLIP", "CLIP", "SD3", "FLUX"]

    combine_activations(
        cache_root=colab_cache_root,
        output_root=colab_output_cache,
        sources=sources,
        num_workers=12,
    )
