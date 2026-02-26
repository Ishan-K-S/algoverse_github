"""
Cache forward feature pass of various models on Imagenet dataset (train set).
    - could be modified to cache multiple models at once.
    
Model output shapes:
    - ViT:    (batch_size, 197, 768)  - 196 patches (14x14) + 1 CLS token
    - CLIP:   (batch_size, 197, 768)  - 196 patches (14x14) + 1 CLS token
    - SigLIP: (batch_size, 196, 768)  - 196 patches (14x14), no CLS token
    - DinoV2: (batch_size, 257, 384)  - 256 patches (16x16) + 1 CLS token
"""

import sys; sys.path.insert(0, '..')
from models import ViT, DinoV2, SigLIP, CLIP
from torchvision.datasets import ImageNet
import torch
import numpy as np
import os
from tqdm import tqdm
import gzip

# Save a tensor with gzip compression
def save_tensor_compressed(tensors, path):
    with gzip.open(path, 'wb') as f:
        torch.save(tensors, f)

# Load a compressed tensor
def load_tensor_compressed(path):
    with gzip.open(path, 'rb') as f:
        tensor = torch.load(f)
    return tensor

# Save tensor in NPY format with compression
def save_tensor_npz(tensors, path):
    np.savez_compressed(path, activation=tensors[0].numpy(), label=tensors[1].numpy())

# Load tensor from NPZ format
def load_tensor_npz(path):
    data = np.load(path)
    activation = torch.tensor(data['activation'])
    label = torch.tensor(data['label'])
    return activation, label


def cache_model_activations(model, model_name, imagenet_root, path_to_cache, batch_size=512):
    """
    Cache activations for a single model.
    
    Parameters
    ----------
    model : BaseModel
        The model instance (already on CUDA).
    model_name : str
        Name of the model for file naming.
    imagenet_root : str
        Path to ImageNet dataset.
    path_to_cache : str
        Output directory for cached activations.
    batch_size : int
        Batch size for processing.
    """
    print(f"Caching model activations: {model_name}")
    print("Loading ImageNet")

    imagenet_data = ImageNet(imagenet_root, split='train',transform=model.preprocess)

    print("ImageNet Loaded")

    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    print(f"Model: {model_name}, Cache path: {path_to_cache}")

    # Iterate through the dataloader
    for i, batch in enumerate(tqdm(data_loader)):
        x, y = batch
        x = x.cuda()
        y = y.cpu()

        # Retrieve batch filenames from the dataset
        image_paths = data_loader.dataset.samples[i * data_loader.batch_size:(i + 1) * data_loader.batch_size]

        # Get output features based on model type
        with torch.no_grad():
            if model_name == "DinoV2":
                # DinoV2: returns dict with 'x_norm_clstoken' and 'x_norm_patchtokens'
                output_dict = model.model.forward_features(x)
                cls = output_dict['x_norm_clstoken'].unsqueeze(1)
                output_features = torch.cat((cls, output_dict['x_norm_patchtokens']), dim=1)
            elif model_name in {"ViT", "CLIP"}:
                # ViT and CLIP: use model.model.forward_features to include CLS token
                output_features = model.model.forward_features(x)
            elif model_name == "SigLIP":
                # SigLIP: no CLS token
                output_features = model.forward_features(x)
            else:
                # Generic fallback
                output_features = model.forward_features(x)

        output_features = output_features.cpu()

        # Loop through each sample in the batch
        for j in range(x.size(0)):
            # Get the corresponding image path
            image_path = image_paths[j][0]

            # Convert ImageNet image path to the corresponding cache path
            relative_path = os.path.relpath(image_path, data_loader.dataset.root)
            activation_path = os.path.join(path_to_cache, relative_path)

            # Get only the directory part
            activation_dir = os.path.dirname(activation_path)

            # Ensure the directory exists
            os.makedirs(activation_dir, exist_ok=True)

            # Save both the activations and label
            activation_filename = activation_path.replace('.JPEG', f"_{model_name}")

            save_tensor_npz(
                path=activation_filename,
                tensors=(output_features[j].detach().clone(), y[j].detach().clone())
            )


if __name__ == '__main__':
    # Configuration
    imagenet_root = "./imagenet_sample"
    path_to_cache = "./cache_output"
    
    # Select which model to cache (uncomment one)
    
    # Option 1: ViT - Output shape: Bx197x768
    model = ViT().cuda()
    model_name = "ViT"
    
    # Option 2: CLIP - Output shape: Bx197x768 (same as ViT)
    # model = CLIP().cuda()
    # model_name = "CLIP"
    
    # Option 3: SigLIP - Output shape: Bx196x768 (no cls token)
    # model = SigLIP().cuda()
    # model_name = "SigLIP"
    
    # Option 4: DinoV2 - Output shape: Bx257x384
    # model = DinoV2().cuda()
    # model_name = "DinoV2"

    model.eval()
    
    cache_model_activations(
        model=model,
        model_name=model_name,
        imagenet_root=imagenet_root,
        path_to_cache=path_to_cache,
        batch_size=512
    )
