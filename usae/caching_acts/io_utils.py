import numpy as np
import torch

def save_npz(path: str, **arrays):
    # Convert torch tensors to numpy
    np_arrays = {}
    for k,v in arrays.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        np_arrays[k] = v
    np.savez_compressed(path, **np_arrays)
