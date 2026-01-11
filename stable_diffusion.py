"""
Stable Diffusion CLIP Image Encoder for feature extraction.

This module provides the StableDiffusion model following the overcomplete
library's BaseModel interface pattern.
"""

import torch
from torch import nn
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from overcomplete.models import BaseModel


class CLIPVisionWrapper(nn.Module):
    """
    Wrapper around CLIP vision model to match DINOv2's interface.
    Provides forward_features() method that returns dict with tokens.
    """
    
    def __init__(self, vision_model, use_half=False):
        super().__init__()
        self.vision_model = vision_model
        self.use_half = use_half
    
    def forward_features(self, x):
        """
        Forward pass returning dict with CLS and patch tokens.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'x_norm_clstoken': CLS token of shape (batch_size, hidden_dim)
            - 'x_norm_patchtokens': Patch tokens of shape (batch_size, num_patches, hidden_dim)
        """
        if self.use_half:
            x = x.half()
        
        # Forward through CLIP vision model
        outputs = self.vision_model(
            pixel_values=x,
            return_dict=True
        )
        
        # CLIP structure: [CLS, patch1, patch2, ..., patchN]
        hidden_states = outputs.last_hidden_state  # (B, 1+N, D)
        
        cls_token = hidden_states[:, 0, :]         # (B, D)
        patch_tokens = hidden_states[:, 1:, :]     # (B, N, D)
        
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens
        }
    
    def eval(self):
        """Set to evaluation mode."""
        self.vision_model.eval()
        return self
    
    def to(self, device):
        """Move to device."""
        self.vision_model.to(device)
        return self
    
    def half(self):
        """Convert to half precision."""
        self.vision_model.half()
        return self


class StableDiffusion(BaseModel):
    """
    Concrete class for Stable Diffusion CLIP image encoder.

    Parameters
    ----------
    use_half : bool, optional
        Whether to use half-precision (float16), by default False.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.

    Methods
    -------
    forward_features(x):
        Perform a forward pass on the input tensor.

    preprocess(img):
        Preprocess input images for the CLIP model.
    """

    def __init__(self, use_half=False, device='cpu'):
        super().__init__(use_half, device)
        
        # Use CLIP ViT-L/14 from OpenAI (used in SD 1.x and 2.x)
        model_name = "openai/clip-vit-large-patch14"
        
        # Load the CLIP vision model
        vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # Wrap it to match DINOv2's interface (self.model.forward_features())
        self.model = CLIPVisionWrapper(vision_model.vision_model, use_half=use_half).eval().to(self.device)
        
        if self.use_half:
            self.model = self.model.half()

        # Create preprocessing pipeline matching CLIP specifications
        self.preprocess = transforms.Compose([
            transforms.Resize(
                processor.size["shortest_edge"],
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.CenterCrop(processor.size["shortest_edge"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=processor.image_mean,
                std=processor.image_std
            )
        ])

    def forward_features(self, x):
        """
        Perform a forward pass on the input tensor.
        Assume input is in the same device as the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output patch tokens of shape (batch_size, num_patches, hidden_dim).
        """
        with torch.no_grad():
            if self.use_half:
                x = x.half()
            return self.model.forward_features(x)['x_norm_patchtokens']
