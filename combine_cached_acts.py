# combine_cached_acts.py (patched)

import os
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

DIFF_META_FIELDS = ("sigmas", "timesteps")

def process_class_directory(args):
    class_dir, split_dir, output_split_dir, sources = args

    class_path = os.path.join(split_dir, class_dir)
    if not os.path.isdir(class_path):
        return

    output_class_dir = os.path.join(output_split_dir, class_dir)
    os.makedirs(output_class_dir, exist_ok=True)

    source_files = glob.glob(os.path.join(class_path, f"*_{sources[0]}.npz"))

    for source_file in source_files:
        base_name = os.path.basename(source_file).replace(f"_{sources[0]}.npz", "")

        combined = {}
        for source in sources:
            act_path = os.path.join(class_path, f"{base_name}_{source}.npz")
            if not os.path.exists(act_path):
                print(f"Missing activation file: {act_path}")
                continue

            data = np.load(act_path, allow_pickle=False)

            # vision cache: has 'activation'
            if "activation" in data.files:
                combined[source] = data["activation"]
            else:
                # If you ever save vision keys directly, support it too
                # (not used by your current pipeline)
                for k in data.files:
                    combined[f"{source}_{k}"] = data[k]

            # diffusion cache metadata (optional)
            for mf in DIFF_META_FIELDS:
                if mf in data.files:
                    combined[f"{source}__{mf}"] = data[mf]

        out_path = os.path.join(output_class_dir, f"{base_name}_combined.npz")
        np.savez(out_path, **combined)


def combine_activations(activation_root: str, output_root: str, split: str, sources: list, num_workers: int = 8):
    split_dir = os.path.join(activation_root, split)
    output_split_dir = os.path.join(output_root, split)

    class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    args_list = [(class_dir, split_dir, output_split_dir, sources) for class_dir in class_dirs]

    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_class_directory, args_list), total=len(class_dirs), desc="Combining activations"))


if __name__ == "__main__":
    activation_root = ""
    output_root = ""
    split = "train"

    # add diffusion sources as needed:
    sources = ["ViT", "DinoV2", "SigLIP", "CLIP", "SD3", "FLUX"]

    combine_activations(
        activation_root=activation_root,
        output_root=output_root,
        split=split,
        sources=sources,
        num_workers=12,
    )
