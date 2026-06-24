import sys
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

class CocoData(Dataset):

    def __init__(self, path_to_data, transform, max_images=None):
        self.path_to_data = path_to_data
        self.transform = transform

        if not os.path.isdir(path_to_data):
            raise RuntimeError(
                f"[CocoData] Image directory not found: {path_to_data}\n"
                "  Make sure you downloaded COCO val2017 and extracted it to this path."
            )

        self.images = [
                    f for f in os.listdir(path_to_data)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                       ]

        if len(self.images) == 0:
            raise RuntimeError(
                f"[CocoData] No images (.jpg/.png/.jpeg) found in: {path_to_data}"
            )

        if max_images is not None:
            self.images = random.sample(self.images, min(max_images, len(self.images)))
            self.images.sort()
        else:
            self.images.sort()

        print(f"[CocoData] Found {len(self.images)} images in {path_to_data}"
              + (f" (capped at {max_images})" if max_images is not None else ""))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        """Returns a single image as a tuple in the form of (tensor, filename)"""
        

        filename = self.images[index]
        image_path = os.path.join(self.path_to_data, filename)

        with Image.open(image_path) as img:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, filename

"""
        ImageTensor = Image.open(image_path).convert('RGB')

        if self.transform:
            ImageTensor = self.transform(ImageTensor)
        
        return (ImageTensor, filename)"""
