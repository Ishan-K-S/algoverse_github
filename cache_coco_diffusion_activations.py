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
from coco_dataset_setup import CocoData, select_images


from DiffusionActivationExtractor import SD3ActivationExtractor, FLUXActivationExtractor, PixArtActivationExtractor

from huggingface_hub import login
login(token="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")


def save_diffusion_npz(path_no_ext: str, activation_tnd: torch.Tensor, sigmas, timesteps, filename: str):
    """
    path_no_ext: full path without ".npz"
    activation_tnd: (T, N, D) on CPU
    sigmas: list[float] length T
    timesteps: list[torch.Tensor] length T (each scalar tensor)
    """
    sigmas_arr = np.asarray(sigmas, dtype=np.float32)[None, :]
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
    batch_size: int = 4,
    num_workers: int = 2,
    image_list=None,
):
    """
    extractor: SD3ActivationExtractor or FLUXActivationExtractor.
      Must expose `.preprocess` compatible with torchvision transforms, and
      `.extract_activations(image_batch)` that returns ActivationOutput with
      activations (list of (B,N,D)), sigmas(list[float]), timesteps(list[tensor]).
    image_list : list[str] or None
        Pre-selected image filenames (from select_images()). If None, all images
        in coco_root are used.
    """
    print(f"[diffusion-cache] ---- Starting diffusion caching ----")
    print(f"[diffusion-cache] source     : {source_name}")
    print(f"[diffusion-cache] coco_root  : {coco_root}  exists={os.path.isdir(coco_root)}")
    print(f"[diffusion-cache] cache_root : {cache_root}  exists={os.path.isdir(cache_root)}")
    print(f"[diffusion-cache] device     : {extractor.device}")
    if not os.path.isdir(coco_root):
        raise RuntimeError(f"[diffusion-cache] COCO root not found: {coco_root}\n"
                           "  Download COCO val2017 first: wget http://images.cocodataset.org/zips/val2017.zip")
    if not os.path.isdir(cache_root):
        raise RuntimeError(f"[diffusion-cache] Cache directory not found: {cache_root}\n"
                           "  Create it first: os.makedirs('/content/cache', exist_ok=True)")

    ds = CocoData(coco_root, transform=extractor.preprocess, image_list=image_list)

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

        if i == 0:
            print(f"[diffusion-cache] First batch image shape : {tuple(x.shape)}")

        out = extractor.extract_activations(x)
        # out.activations: list length T, each (B,N,D)
        # stack -> (T,B,N,D) -> (B,T,N,D)
        acts_t = torch.stack([a.detach().cpu() for a in out.activations], dim=0)
        acts_btnd = acts_t.permute(1, 0, 2, 3).contiguous()

        if i == 0:
            print(f"[diffusion-cache] First batch activation shape (B,T,N,D) : {tuple(acts_btnd.shape)}")
            print(f"[diffusion-cache] Timesteps ({len(out.timesteps)}): {[int(t.item()) for t in out.timesteps]}")
            print(f"[diffusion-cache] Sigmas    ({len(out.sigmas)}): {[round(s, 4) for s in out.sigmas]}")

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
    n_written = len([f for f in os.listdir(cache_root) if f.endswith(f"_{source_name}.npz")])
    print(f"[diffusion-cache] Caching complete. Files written: {n_written} in {cache_root}")

if __name__ == "__main__":
    coco_root = "/lambda/nfs/AlgoverseResearchAIJK/coco_data/train2017"
    cache_root = "/lambda/nfs/AlgoverseResearchAIJK/cache_path"

    colab_coco_root = "/content/coco_data/val2017"
    colab_path_to_cache = "/content/cache"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[diffusion-cache] GPU available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[diffusion-cache] GPU           : {torch.cuda.get_device_name(0)}")

    # Choose ONE:
    #extractor = SD3ActivationExtractor(device=device, num_inference_steps=4)
    #source_name = "SD3"

    extractor = PixArtActivationExtractor(device = device, num_inference_steps=15)
    source_name = "PixArt"

    #extractor = FLUXActivationExtractor(device=device, num_inference_steps=4)
    #source_name = "FLUX"

    # Load the same selection file written by cache_coco_activations.py.
    # If it doesn't exist yet (running diffusion first), it will be created here.
    selection_file = os.path.join(colab_path_to_cache, "selected_images.txt")
    image_list = select_images(colab_coco_root, 2000, selection_file)

    cache_diffusion_activations(
        extractor=extractor,
        source_name=source_name,
        coco_root=colab_coco_root,
        cache_root=colab_path_to_cache,
        batch_size=2,
        num_workers=2,
        image_list=image_list,
    )
