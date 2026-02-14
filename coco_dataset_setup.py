import sys
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CocoData(Dataset):

    def __init__(self, path_to_data, transform):
        self.path_to_data = path_to_data
        self.transform = transform

        self.images = [
                    f for f in os.listdir(path_to_data)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                       ]
        
        self.images.sort()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        """Returns a single image as a tuple in the form of (tensor, filename)"""
        with Image.open(image_path) as img:
            img = img.convert("RGB")

        filename = self.images[index]
        image_path = os.path.join(self.path_to_data, filename)

        if self.transform:
            img = self.transform(img)

        return img, filename

"""
        ImageTensor = Image.open(image_path).convert('RGB')

        if self.transform:
            ImageTensor = self.transform(ImageTensor)
        
        return (ImageTensor, filename)"""
