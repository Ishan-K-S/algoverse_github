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
import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
DRIVE_SAVE_DIR = "/content/drive/MyDrive/algoverse_results/combined_cache2000"



DIFF_META_FIELDS = ("sigmas", "timesteps")


def _mount_drive(retries: int = 3) -> bool:
    """Mount Google Drive, returning True on success."""
    try:
        from google.colab import drive as _colab_drive
    except ImportError:
        return False
    for attempt in range(retries):
        try:
            _colab_drive.mount("/content/drive", force_remount=(attempt > 0))
            if os.path.isdir("/content/drive/MyDrive"):
                return True
        except Exception as e:
            print(f"[drive] Mount attempt {attempt + 1} failed: {e}")
        time.sleep(5)
    return False


def _robust_copy(src: str, dst: str, retries: int = 3) -> None:
    """Copy a file using explicit read/write to avoid FUSE sendfile() failures."""
    for attempt in range(retries):
        try:
            with open(src, "rb") as fsrc:
                data = fsrc.read()
            with open(dst, "wb") as fdst:
                fdst.write(data)
            return
        except OSError as e:
            if attempt < retries - 1:
                print(f"\n[drive] OSError on {os.path.basename(src)} (errno {e.errno}), remounting...")
                _mount_drive()
                time.sleep(5)
            else:
                raise


def save_combined_cache_to_drive(combined_cache_dir: str, drive_dir: str) -> None:
    """Copy the entire combined cache directory to Google Drive."""
    if not _mount_drive():
        print("[drive] Not running in Colab — skipping Drive upload.")
        return

    if not os.path.isdir(combined_cache_dir):
        print(f"[drive] Warning: {combined_cache_dir} not found — skipping.")
        return

    files = [f for f in os.listdir(combined_cache_dir) if f.endswith("_combined.npz")]
    if not files:
        print(f"[drive] No combined npz files found in {combined_cache_dir} — skipping.")
        return

    os.makedirs(drive_dir, exist_ok=True)
    print(f"[drive] Copying {len(files)} combined npz files to {drive_dir} ...")
    for fname in tqdm(files, desc="Saving to Drive"):
        _robust_copy(os.path.join(combined_cache_dir, fname), os.path.join(drive_dir, fname))
    print(f"[drive] Done. {len(files)} files saved to {drive_dir}")


def load_combined_cache_from_drive(drive_dir: str, local_dir: str) -> None:
    """Copy combined npz files from Google Drive to a local Colab directory."""
    if not _mount_drive():
        print("[drive] Not running in Colab — skipping Drive download.")
        return

    if not os.path.isdir(drive_dir):
        print(f"[drive] Warning: {drive_dir} not found on Drive — skipping.")
        return

    files = [f for f in os.listdir(drive_dir) if f.endswith("_combined.npz")]
    if not files:
        print(f"[drive] No combined npz files found in {drive_dir} — skipping.")
        return

    os.makedirs(local_dir, exist_ok=True)
    print(f"[drive] Copying {len(files)} files from Drive to {local_dir} ...")
    for fname in tqdm(files, desc="Loading from Drive"):
        _robust_copy(os.path.join(drive_dir, fname), os.path.join(local_dir, fname))
    print(f"[drive] Done. {len(files)} files copied to {local_dir}")


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
    print(f"[combine] ---- Starting combine step ----")
    print(f"[combine] cache_root   : {cache_root}  exists={os.path.isdir(cache_root)}")
    print(f"[combine] output_root  : {output_root}")
    print(f"[combine] sources      : {sources}")
    print(f"[combine] num_workers  : {num_workers}")
    if not os.path.isdir(cache_root):
        raise RuntimeError(
            f"[combine] cache_root not found: {cache_root}\n"
            "  Run cache_coco_activations.py and cache_coco_diffusion_activations.py first."
        )
    for src in sources:
        n = len([f for f in os.listdir(cache_root) if f.endswith(f"_{src}.npz")])
        print(f"[combine]   {src}: {n} cached files found")
        if n == 0:
            raise RuntimeError(
                f"[combine] No '{src}' npz files found in {cache_root}.\n"
                f"  Expected files ending in '_{src}.npz'. "
                f"Did the caching script for '{src}' complete successfully?"
            )
    os.makedirs(output_root, exist_ok=True)
    already_done = len([f for f in os.listdir(output_root) if f.endswith("_combined.npz")])
    if already_done > 0:
        print(f"[combine] {already_done} combined files already exist in {output_root} (will overwrite)")

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

    n_out = len([f for f in os.listdir(output_root) if f.endswith("_combined.npz")])
    print(f"[combine] Done. Combined files in {output_root}: {n_out}")


if __name__ == "__main__":
    cache_root  = "/lambda/nfs/AlgoverseResearchAIJK/cache_path"
    output_root = "/lambda/nfs/AlgoverseResearchAIJK/combined_path"

    colab_cache_root = "/content/cache"
    colab_output_cache = "/content/combined_cache"

    # List every source you cached, in any order
    #sources = ["ViT", "DinoV2", "SigLIP", "CLIP", "SD3", "FLUX"]
    sources = ["DinoV2", "PixArt"]

    combine_activations(
        cache_root=colab_cache_root,
        output_root=colab_output_cache,
        sources=sources,
        num_workers=12,
    )

    save_combined_cache_to_drive(colab_output_cache, DRIVE_SAVE_DIR)

