import sys
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset


def select_images(coco_root: str, n: int, save_path: str, seed: int = 42) -> list:
    """
    Randomly select n images from coco_root and save filenames to save_path.
    If save_path already exists, load and return it (idempotent — safe to call
    from multiple caching scripts; they all get the same list).
    """
    if os.path.exists(save_path):
        with open(save_path) as f:
            images = [line.strip() for line in f if line.strip()]
        print(f"[select_images] Loaded {len(images)} pre-selected images from {save_path}")
        return images

    all_images = sorted([
        f for f in os.listdir(coco_root)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])
    if not all_images:
        raise RuntimeError(f"[select_images] No images found in {coco_root}")

    rng = random.Random(seed)
    selected = sorted(rng.sample(all_images, min(n, len(all_images))))

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(selected) + '\n')
    print(f"[select_images] Selected and saved {len(selected)} images to {save_path}")
    return selected


def split_train_val(
    identifiers: list,
    save_dir: str,
    val_fraction: float = 0.2,
    seed: int = 42,
    train_filename: str = "train_stems.txt",
    val_filename: str = "val_stems.txt",
) -> tuple:
    """
    Deterministically split a list of identifiers (image filenames or cache
    stems -- this function doesn't care which) into disjoint train/val sets,
    persisted to disk next to selected_images.txt (REPAIR_PLAN.md V7/Fix 2.3).

    Idempotent like select_images(): if both files already exist, loads and
    returns them unchanged rather than re-splitting -- so every diagnostic and
    every resumed run sees the exact same split.

    Every prior number in this project (including the fixed_timestep_idx
    choice itself) was measured on the same data used to train, so there was
    no generalization measurement anywhere. This exists to fix that; it will
    make the headline numbers look worse, which is the point.
    """
    train_path = os.path.join(save_dir, train_filename)
    val_path = os.path.join(save_dir, val_filename)

    if os.path.exists(train_path) and os.path.exists(val_path):
        with open(train_path) as f:
            train_list = [line.strip() for line in f if line.strip()]
        with open(val_path) as f:
            val_list = [line.strip() for line in f if line.strip()]

        # Fail loudly, not silently, if the cache this split was made from has
        # since changed (e.g. a Fix 2.1 re-cache under a different image set)
        # -- otherwise new stems belong to neither persisted list and quietly
        # vanish from training instead of landing in either split.
        persisted = set(train_list) | set(val_list)
        current = set(identifiers)
        if persisted != current:
            missing_from_cache = persisted - current
            new_in_cache = current - persisted
            raise RuntimeError(
                f"[split_train_val] Persisted split at {save_dir} no longer matches the "
                f"current cache: {len(missing_from_cache)} persisted stem(s) are no longer "
                f"in the cache, {len(new_in_cache)} cached stem(s) aren't in the persisted "
                f"split. The cache was likely re-generated since this split was created. "
                f"Delete {train_filename!r}/{val_filename!r} in {save_dir} to regenerate the "
                f"split against the current cache (this will produce a NEW split -- old "
                f"results measured on the old split will not be directly comparable)."
            )

        print(f"[split_train_val] Loaded persisted split: "
              f"{len(train_list)} train / {len(val_list)} val from {save_dir}")
        return train_list, val_list

    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    items = sorted(set(identifiers))
    if not items:
        raise RuntimeError("[split_train_val] Got an empty identifier list to split.")

    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    if n_val >= len(shuffled):
        raise RuntimeError(
            f"[split_train_val] val_fraction={val_fraction} leaves no training images "
            f"out of {len(shuffled)} total."
        )
    val_list = sorted(shuffled[:n_val])
    train_list = sorted(shuffled[n_val:])

    overlap = set(train_list) & set(val_list)
    assert not overlap, f"[split_train_val] train/val overlap (should be impossible): {overlap}"

    os.makedirs(save_dir, exist_ok=True)
    with open(train_path, 'w') as f:
        f.write('\n'.join(train_list) + '\n')
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_list) + '\n')
    print(f"[split_train_val] Created new split (seed={seed}): "
          f"{len(train_list)} train / {len(val_list)} val, saved to {save_dir}")
    return train_list, val_list


class CocoData(Dataset):

    def __init__(self, path_to_data, transform, image_list=None, max_images=None, seed=42):
        self.path_to_data = path_to_data
        self.transform = transform

        if not os.path.isdir(path_to_data):
            raise RuntimeError(
                f"[CocoData] Image directory not found: {path_to_data}\n"
                "  Make sure you downloaded COCO val2017 and extracted it to this path."
            )

        if image_list is not None:
            # Explicit list supplied — use it directly (no random sampling needed).
            self.images = sorted(list(image_list))
        else:
            self.images = sorted([
                f for f in os.listdir(path_to_data)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            if len(self.images) == 0:
                raise RuntimeError(
                    f"[CocoData] No images (.jpg/.png/.jpeg) found in: {path_to_data}"
                )
            if max_images is not None:
                rng = random.Random(seed)
                self.images = sorted(rng.sample(self.images, min(max_images, len(self.images))))

        print(f"[CocoData] Using {len(self.images)} images from {path_to_data}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]
        image_path = os.path.join(self.path_to_data, filename)

        with Image.open(image_path) as img:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, filename
