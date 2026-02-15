# cache_imagenet_diffusion_activations.py
"""
Cache diffusion transformer activations (SD3 / FLUX) on ImageNet in the SAME
directory format as cache_coco_activations.py.

For each image, saves:
  activation: (T, N, D)
  sigmas:     (T,)
  timesteps:  (T,)
  label:      ()
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from coco_dataset_setup import CocoData


from DiffusionActivationExtractor import SD3ActivationExtractor, FLUXActivationExtractor

from huggingface_hub import login
login()


def save_diffusion_npz(path_no_ext: str, activation_tnd: torch.Tensor, sigmas, timesteps, filename: str):
    """
    path_no_ext: full path without ".npz"
    activation_tnd: (T, N, D) on CPU
    sigmas: list[float] length T
    timesteps: list[torch.Tensor] length T (each scalar tensor)
    """
    sigmas_arr = np.asarray(sigmas, dtype=np.float32)
    # timesteps might be torch scalar tensors; convert safely to int64
    ts_arr = np.asarray([int(t.item()) for t in timesteps], dtype=np.int64)
    if activation_tnd.dtype == torch.bfloat16:
        activation_tnd = activation_tnd.float()
    np.savez_compressed(
        path_no_ext + ".npz",
        activation=activation_tnd.numpy(),
        sigmas=sigmas_arr,
        timesteps=ts_arr,
        filename=filename,
    )


@torch.no_grad()
def cache_diffusion_activations(
    extractor,
    source_name: str,
    coco_root: str,
    cache_root: str,
    #split: str = "train",
    batch_size: int = 4,
    num_workers: int = 8,
):
    """
    extractor: SD3ActivationExtractor or FLUXActivationExtractor.
      Must expose `.preprocess` compatible with torchvision transforms, and
      `.extract_activations(image_batch)` that returns ActivationOutput with
      activations (list of (B,N,D)), sigmas(list[float]), timesteps(list[tensor]).
      (This matches your DiffusionActivationExtractor.py.) :contentReference[oaicite:4]{index=4}
    """
    #print(f"[diffusion-cache] source={source_name} split={split}")

    ds = CocoData(coco_root, transform=extractor.preprocess)

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    #os.makedirs(cache_root, exist_ok=True)

    for i, (x, y) in enumerate(tqdm(dl, desc=f"Caching {source_name}", dynamic_ncols=True)):
        # x is already preprocessed tensor from extractor.preprocess
        #x is images and y is filename
        x = x.to(extractor.device, non_blocking=True)
        #y = y.cpu()

        # match your vision script’s way of mapping batch indices -> file paths :contentReference[oaicite:5]{index=5}
        #image_paths = dl.dataset.samples[i * dl.batch_size : (i + 1) * dl.batch_size]
        

        out = extractor.extract_activations(x)
        # out.activations: list length T, each (B,N,D)
        # stack -> (T,B,N,D) -> (B,T,N,D)
        acts_t = torch.stack([a.detach().cpu() for a in out.activations], dim=0)
        acts_btnd = acts_t.permute(1, 0, 2, 3).contiguous()

        # out.sigmas: list[float], out.timesteps: list[tensor] length T
        sigmas = out.sigmas
        timesteps = out.timesteps

        for j in range(acts_btnd.shape[0]):
            #image_path = image_paths[j][0]    # full path
            filename = y[j]  
            # IMPORTANT: match ImageNet structure in cache root
            # ImageNet root structure is <imagenet_root>/<split>/<class>/<img>.JPEG
            """rel_path = os.path.relpath(image_path, dl.dataset.root)
            cache_path = os.path.join(cache_root, rel_path)
            cache_dir = os.path.dirname(cache_path)
            os.makedirs(cache_dir, exist_ok=True)

            # .../n014.../n014..._1234.JPEG -> .../n014.../n014..._1234_<source>.npz
            base = cache_path.replace(".JPEG", f"_{source_name}")"""

            cache_filename = filename.replace('.jpg', f'_{source_name}.npz')
            cache_path = os.path.join(cache_root, cache_filename)
            base = cache_path.replace('.npz', '')

            save_diffusion_npz(
                path_no_ext=base,
                activation_tnd=acts_btnd[j],
                sigmas=sigmas,
                timesteps=timesteps,
                filename=cache_filename,
            )
    print(f"caching complete")

if __name__ == "__main__":
    coco_root = "/content/coco_data/val2017"
    cache_root = "/content/cache_path"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Choose ONE:
    extractor = SD3ActivationExtractor(device=device, num_inference_steps=28)
    source_name = "SD3"

    #extractor = FLUXActivationExtractor(device=device, num_inference_steps=4)
    #source_name = "FLUX"

    cache_diffusion_activations(
        extractor=extractor,
        source_name=source_name,
        coco_root=coco_root,
        cache_root=cache_root,
        #split="train",
        batch_size=2,
        num_workers=8,
    )
