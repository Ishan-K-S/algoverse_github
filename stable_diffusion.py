"""
Stable Diffusion 3 DiT Transformer for feature extraction.

This module extracts features from the diffusion transformer during the 
denoising process, capturing the model's internal representations.
"""

import torch
from torch import nn
from torchvision import transforms
from diffusers import StableDiffusion3Pipeline, DDPMScheduler
from PIL import Image

from overcomplete.models import BaseModel


class SD3TransformerWrapper(nn.Module):
    """
    Wrapper around SD3's DiT transformer to extract features during denoising.
    Provides forward_features() method that returns transformer activations.
    """
    
    def __init__(self, transformer, vae, scheduler, text_encoder, tokenizer, use_half=False, device='cpu'):
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.use_half = use_half
        self.device = device
        
        # Denoising parameters
        self.num_inference_steps = 5  # Quick denoising, adjust as needed
        self.timestep_to_extract = 2  # Which denoising step to extract from
        
    @torch.no_grad()
    def forward_features(self, x):
        """
        Extract transformer features during denoising process.
        
        Process:
        1. Encode images to latent space
        2. Add noise (forward diffusion)
        3. Denoise through transformer
        4. Extract intermediate activations
        """
        batch_size = x.shape[0]
        
        # Step 1: Encode images to latent space
        if self.use_half:
            x = x.half()
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # Step 2: Add noise (forward diffusion)
        noise = torch.randn_like(latents)
        timestep = torch.tensor([self.scheduler.config.num_train_timesteps // 2], device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)
        
        # Step 3: Create null text embeddings (unconditional)
        null_prompt = [""] * batch_size
        text_inputs = self.tokenizer(
            null_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        # Step 4: Run through transformer and extract features
        # We'll hook into the transformer to extract intermediate activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        # Register hook on the last transformer block
        hook_handle = None
        if hasattr(self.transformer, 'transformer_blocks'):
            last_block = self.transformer.transformer_blocks[-1]
            hook_handle = last_block.register_forward_hook(hook_fn('last_block'))
        
        # Run denoising step
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        for t in self.scheduler.timesteps[:self.timestep_to_extract + 1]:
            # Prepare timestep
            timestep_tensor = torch.tensor([t], device=self.device).expand(batch_size)
            
            # Predict noise
            noise_pred = self.transformer(
                noisy_latents,
                timestep_tensor,
                encoder_hidden_states=text_embeddings,
                return_dict=False
            )[0]
            
            # Denoise
            noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents, return_dict=False)[0]
        
        # Remove hook
        if hook_handle:
            hook_handle.remove()
        
        # Extract features from activations
        if 'last_block' in activations:
            features = activations['last_block']
        else:
            # Fallback: use the noisy latents themselves
            features = noisy_latents
        
        # Reshape to (B, N, D) format
        B, C, H, W = features.shape
        features_flat = features.reshape(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        # Create CLS token as spatial average
        cls_token = features_flat.mean(dim=1)  # (B, C)
        patch_tokens = features_flat  # (B, H*W, C)
        
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens
        }
    
    def eval(self):
        """Set to evaluation mode."""
        self.transformer.eval()
        self.vae.eval()
        self.text_encoder.eval()
        return self
    
    def to(self, device):
        """Move to device."""
        self.transformer.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)
        self.device = device
        return self
    
    def half(self):
        """Convert to half precision."""
        self.transformer.half()
        self.vae.half()
        self.text_encoder.half()
        return self


class StableDiffusion(BaseModel):
    """
    Concrete class for Stable Diffusion 3 DiT transformer feature extraction.

    Extracts features from the diffusion transformer during the denoising process.
    This captures the model's internal understanding of images through the 
    generative modeling process.
    """

    def __init__(self, use_half=False, device='cpu'):
        super().__init__(use_half, device)
        
        print("Loading Stable Diffusion 3 pipeline...")
        
        # Load SD3 pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16 if use_half else torch.float32
        )
        
        # Extract components
        transformer = pipe.transformer
        vae = pipe.vae
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        
        # Create wrapper
        self.model = SD3TransformerWrapper(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            use_half=use_half,
            device=device
        ).eval().to(self.device)
        
        if self.use_half:
            self.model = self.model.half()

        # SD3 preprocessing (512x512 by default)
        self.preprocess = transforms.Compose([
            transforms.Resize(
                512,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def forward_features(self, x):
        """
        Perform a forward pass on the input tensor.
        Runs through the diffusion process and extracts transformer features.
        """
        with torch.no_grad():
            if self.use_half:
                x = x.half()
            return self.model.forward_features(x)['x_norm_patchtokens']
