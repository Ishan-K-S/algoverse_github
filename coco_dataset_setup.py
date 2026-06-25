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
