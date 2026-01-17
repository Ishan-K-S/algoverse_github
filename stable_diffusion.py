"""
Stable Diffusion 3 DiT Transformer for feature extraction.

This module extracts features from the diffusion transformer during the 
denoising process, implementing TIDE-inspired improvements:
- Token sampling (1/16 of spatial tokens)
- Timestep-dependent modulation
"""

import torch
from torch import nn
from torchvision import transforms
from diffusers import StableDiffusion3Pipeline, DDPMScheduler
import torch.nn.functional as F

from overcomplete.models import BaseModel


class TimestepModulation(nn.Module):
    """
    Timestep-dependent modulation operation inspired by TIDE.
    Uses adaptive layer normalization conditioned on timestep.
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        # Learnable scale and shift parameters conditioned on timestep
        self.scale_net = nn.Linear(1, hidden_dim)
        self.shift_net = nn.Linear(1, hidden_dim)
        
        # Initialize from adaptive LayerNorm-like behavior
        nn.init.ones_(self.scale_net.weight)
        nn.init.zeros_(self.scale_net.bias)
        nn.init.zeros_(self.shift_net.weight)
        nn.init.zeros_(self.shift_net.bias)
    
    def forward(self, x, timestep):
        """
        Apply timestep-dependent modulation.
        """
        # Normalize timestep to [0, 1] range
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(-1)  # (B, 1)
        timestep_normalized = timestep.float() / 1000.0  # Assuming max timestep ~1000
        
        # Compute scale and shift
        scale = self.scale_net(timestep_normalized).unsqueeze(1)  # (B, 1, D)
        shift = self.shift_net(timestep_normalized).unsqueeze(1)  # (B, 1, D)
        
        # Apply modulation: x * (1 + scale) + shift
        return x * (1 + scale) + shift


class SD3TransformerWrapper(nn.Module):
    """
    Wrapper around SD3's DiT transformer to extract features during denoising.
    Implements TIDE-inspired improvements for better feature quality.
    """
    
    def __init__(self, transformer, vae, scheduler, text_encoder, tokenizer, 
                 use_half=False, device='cpu', sampling_ratio=1/16):
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.use_half = use_half
        self.device = device
        self.sampling_ratio = sampling_ratio  # Sample 1/16 of tokens
        
        # Denoising parameters
        self.num_inference_steps = 5
        self.timestep_to_extract = 2
        
        # Get hidden dimension from transformer config
        if hasattr(transformer.config, 'hidden_size'):
            hidden_dim = transformer.config.hidden_size
        elif hasattr(transformer.config, 'in_channels'):
            hidden_dim = transformer.config.in_channels
        else:
            hidden_dim = 1024  # Default for SD3
        
        # Timestep modulation module
        self.timestep_modulation = TimestepModulation(hidden_dim).to(device)
        if use_half:
            self.timestep_modulation = self.timestep_modulation.half()
        
    def sample_tokens(self, features, ratio=None):
        """
        Sample a subset of spatial tokens.
        
        Following TIDE's approach of sampling 1/16 tokens for better generalization.
        """
        if ratio is None:
            ratio = self.sampling_ratio
            
        B, C, H, W = features.shape
        total_tokens = H * W
        num_samples = max(1, int(total_tokens * ratio))
        
        # Flatten spatial dimensions
        features_flat = features.reshape(B, C, total_tokens).transpose(1, 2)  # (B, HW, C)
        
        # Random sampling (same indices for all samples in batch for consistency)
        indices = torch.randperm(total_tokens, device=features.device)[:num_samples]
        sampled_features = features_flat[:, indices, :]  # (B, num_samples, C)
        
        return sampled_features, indices.tolist()
        
    @torch.no_grad()
    def forward_features(self, x):
        """
        Extract transformer features during denoising process with TIDE improvements.
        
        Process:
        1. Encode images to latent space
        2. Add noise (forward diffusion)
        3. Denoise through transformer
        4. Extract intermediate activations with timestep modulation
        5. Sample 1/16 of tokens
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
        activations = {}
        current_timestep = None
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        # Register hook on the last transformer block
        hook_handle = None
        if hasattr(self.transformer, 'transformer_blocks'):
            last_block = self.transformer.transformer_blocks[-1]
            hook_handle = last_block.register_forward_hook(hook_fn('last_block'))
        
        # Run denoising steps
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        for i, t in enumerate(self.scheduler.timesteps[:self.timestep_to_extract + 1]):
            # Prepare timestep
            timestep_tensor = torch.tensor([t], device=self.device).expand(batch_size)
            
            if i == self.timestep_to_extract:
                current_timestep = timestep_tensor
            
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
        
        # Ensure features are 4D (B, C, H, W)
        if features.dim() == 3:
            # If features are (B, N, C), reshape to approximate square
            B, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.transpose(1, 2).reshape(B, C, H, W)
        
        # Step 5: Sample 1/16 of tokens (TIDE improvement)
        sampled_features, sampled_indices = self.sample_tokens(features)
        
        # Step 6: Apply timestep-dependent modulation (TIDE improvement)
        if current_timestep is not None:
            sampled_features = self.timestep_modulation(sampled_features, current_timestep)
        
        # Create CLS token as spatial average of sampled tokens
        cls_token = sampled_features.mean(dim=1)  # (B, C)
        patch_tokens = sampled_features  # (B, N_sampled, C)
        
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens
        }
    
    def eval(self):
        """Set to evaluation mode."""
        self.transformer.eval()
        self.vae.eval()
        self.text_encoder.eval()
        self.timestep_modulation.eval()
        return self
    
    def to(self, device):
        """Move to device."""
        self.transformer.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)
        self.timestep_modulation.to(device)
        self.device = device
        return self
    
    def half(self):
        """Convert to half precision."""
        self.transformer.half()
        self.vae.half()
        self.text_encoder.half()
        self.timestep_modulation.half()
        return self


class StableDiffusion(BaseModel):
    """
    Concrete class for Stable Diffusion 3 DiT transformer feature extraction.

    Extracts features from the diffusion transformer during the denoising process
    with TIDE-inspired improvements:
    - Samples 1/16 of spatial tokens for better generalization
    - Applies timestep-dependent modulation to capture temporal dynamics
    """

    def __init__(self, use_half=False, device='cpu', sampling_ratio=1/16):
        super().__init__(use_half, device)
        
        print("Loading Stable Diffusion 3 pipeline...")
        print(f"Token sampling ratio: {sampling_ratio} (sampling {int(sampling_ratio * 100)}% of tokens)")
        
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
        
        # Create wrapper with TIDE improvements
        self.model = SD3TransformerWrapper(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            use_half=use_half,
            device=device,
            sampling_ratio=sampling_ratio
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
        Runs through the diffusion process and extracts transformer features
        with TIDE improvements (token sampling + timestep modulation).
        """
        with torch.no_grad():
            if self.use_half:
                x = x.half()
            return self.model.forward_features(x)['x_norm_patchtokens']
