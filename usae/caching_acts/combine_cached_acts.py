"""
Combine per-source cached activations into a single NPZ per ImageNet image.

Supports:
  - vision encoders cached with keys: activation, label
  - diffusion models cached with keys: activation, timesteps, sigmas, label

Output file:
  <output_root>/<split>/<class>/<img_basename>_combined.npz

Combined NPZ keys:
  - for each source S: S -> activation array
  - for diffusion sources additionally:
        S__timesteps -> (T,)
        S__sigmas    -> (T,)
  - label -> int (copied from first available source)
"""

import os
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def _load_one(path: str):
    npz = np.load(path, allow_pickle=False)
    out = {k: npz[k] for k in npz.files}
    return out


def process_class_directory(args):
    class_dir, split_dir, output_split_dir, sources = args

    class_path = os.path.join(split_dir, class_dir)
    if not os.path.isdir(class_path):
        return

    out_class_dir = os.path.join(output_split_dir, class_dir)
    os.makedirs(out_class_dir, exist_ok=True)

    # anchor list by first source
    anchor = sources[0]
    anchor_files = glob.glob(os.path.join(class_path, f'*_{anchor}.npz'))

    for anchor_file in anchor_files:
        base_name = os.path.basename(anchor_file).replace(f'_{anchor}.npz', '')

        combined = {}
        label = None

        for source in sources:
            p = os.path.join(class_path, f'{base_name}_{source}.npz')
            if not os.path.exists(p):
                print(f"Missing activation file: {p}")
                continue

            data = _load_one(p)

            if label is None and 'label' in data:
                label = data['label']

            # activation always stored
            combined[source] = data['activation']

            # diffusion metadata
            if 'timesteps' in data:
                combined[f"{source}__timesteps"] = data['timesteps']
            if 'sigmas' in data:
                combined[f"{source}__sigmas"] = data['sigmas']

        if label is not None:
            combined['label'] = label

        out_path = os.path.join(out_class_dir, f'{base_name}_combined.npz')
        np.savez_compressed(out_path, **combined)


def combine_activations(activation_root: str, output_root: str, split: str, sources: list, num_workers: int = 8):
    split_dir = os.path.join(activation_root, split)
    out_split_dir = os.path.join(output_root, split)

    class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]

    args_list = [(class_dir, split_dir, out_split_dir, sources) for class_dir in class_dirs]

    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_class_directory, args_list), total=len(class_dirs), desc="Combining activations"))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--split', type=str, default='train', choices=['train','val'])
    parser.add_argument('--sources', type=str, nargs='+', required=True,
                        help='List of source names, must match filename suffixes (e.g., ViT DinoV2 FLUX)')
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()

    combine_activations(
        activation_root=args.activation_root,
        output_root=args.output_root,
        split=args.split,
        sources=args.sources,
        num_workers=args.num_workers,
    )
