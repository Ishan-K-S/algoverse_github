"""
Stable Diffusion 3 MMDiT (Multimodal Diffusion Transformer) for feature extraction.

SD3 uses rectified flow (straight-line paths between data and noise) instead of 
traditional diffusion. This module extracts raw transformer activations for 
training Sparse Autoencoders (SAEs).
"""

import torch
from torch import nn
from torchvision import transforms
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler

from overcomplete.models import BaseModel


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """Calculate dynamic shift (mu) for the scheduler based on image sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class TimestepModulation(nn.Module):
    """
    Timestep-dependent modulation operation.
    Uses adaptive normalization conditioned on timestep.
    """
    
    def __init__(self, hidden_dim, max_timestep=1.0):
        super().__init__()
        # Learnable scale and shift parameters conditioned on timestep
        self.scale_net = nn.Linear(1, hidden_dim)
        self.shift_net = nn.Linear(1, hidden_dim)
        self.max_timestep = max_timestep
        
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
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(-1)  # (B, 1)
        timestep_normalized = timestep.float() / self.max_timestep
        
        # Compute scale and shift
        scale = self.scale_net(timestep_normalized).unsqueeze(1)  # (B, 1, D)
        shift = self.shift_net(timestep_normalized).unsqueeze(1)  # (B, 1, D)
        
        # Apply modulation: x * (1 + scale) + shift
        return x * (1 + scale) + shift


class SD3TransformerWrapper(nn.Module):
    """
    Wrapper around SD3's MMDiT transformer to extract features during rectified flow.
    """
    
    def __init__(self, transformer, vae, scheduler, text_encoder, text_encoder_2, 
                 text_encoder_3, tokenizer, tokenizer_2, tokenizer_3,
                 use_half=False, device='cpu', sampling_ratio=1/16):
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        
        # SD3 uses 3 text encoders: CLIP-L, CLIP-G, T5
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        
        self.use_half = use_half
        self.device = device
        self.sampling_ratio = sampling_ratio
        
        # Flow matching parameters
        self.num_inference_steps = 5
        self.timestep_to_extract = 2
        
        # Get hidden dimension from transformer
        # SD3 uses joint_attention_dim for the hidden dimension
        if hasattr(transformer.config, 'joint_attention_dim'):
            hidden_dim = transformer.config.joint_attention_dim
        else:
            raise ValueError("transformer.config must have 'joint_attention_dim' attribute")
        
        # Store patch size for mu calculation
        self.patch_size = transformer.config.patch_size if hasattr(transformer.config, 'patch_size') else 2
        
        # Determine max timestep from scheduler config (FlowMatch uses [0, 1])
        max_timestep = 1.0
        
        # Timestep modulation module
        self.timestep_modulation = TimestepModulation(hidden_dim, max_timestep).to(device)
        if use_half:
            self.timestep_modulation = self.timestep_modulation.half()
    
    def encode_prompt(self, batch_size):
        """
        Encode null prompts using all three text encoders.
        """
        null_prompt = [""] * batch_size
        
        # Tokenize for all encoders
        text_inputs_1 = self.tokenizer(
            null_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        text_inputs_2 = self.tokenizer_2(
            null_prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        text_inputs_3 = self.tokenizer_3(
            null_prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Encode with all three encoders
        prompt_embeds_1 = self.text_encoder(text_inputs_1, output_hidden_states=True)
        pooled_prompt_embeds_1 = prompt_embeds_1[0]
        prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]
        
        prompt_embeds_2 = self.text_encoder_2(text_inputs_2, output_hidden_states=True)
        pooled_prompt_embeds_2 = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
        
        prompt_embeds_3 = self.text_encoder_3(text_inputs_3)[0]
        
        # Concatenate CLIP embeddings
        clip_prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        # Pad CLIP embeddings to match T5 dimension
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, prompt_embeds_3.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        
        # Concatenate all embeddings
        prompt_embeds = torch.cat([clip_prompt_embeds, prompt_embeds_3], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)
        
        return prompt_embeds, pooled_prompt_embeds
        
    @torch.no_grad()
    def forward_features(self, x):
        """
        Extract transformer features during rectified flow process.
        
        Process:
        1. Encode images to latent space
        2. Get text embeddings
        3. Set up flow matching with dynamic shifting
        4. Add noise using scheduler
        5. Run through transformer and extract features
        6. Sample spatial positions (1/16) from features
        7. Apply timestep modulation
        """
        batch_size = x.shape[0]
        
        # Step 1: Encode images to latent space
        if self.use_half:
            x = x.half()
        latents = self.vae.encode(x).latent_dist.sample()
        # Correct VAE scaling formula (inverse of decode formula)
        # Decode: output = (latent / scaling_factor) + shift_factor
        # Encode: latent = (input - shift_factor) * scaling_factor
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        
        # Step 2: Get text embeddings
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(batch_size)
        
        # Step 3: Set up flow matching with dynamic shifting (mu calculation)
        _, _, height, width = latents.shape
        image_seq_len = (height // self.patch_size) * (width // self.patch_size)
        
        # Calculate mu for dynamic shifting
        # Note: Source uses 1.16 in pipeline __call__ but 1.15 in calculate_shift definition
        # Using 1.16 to match actual pipeline behavior
        mu = calculate_shift(
            image_seq_len,
            base_seq_len=self.scheduler.config.get("base_image_seq_len", 256),
            max_seq_len=self.scheduler.config.get("max_image_seq_len", 4096),
            base_shift=self.scheduler.config.get("base_shift", 0.5),
            max_shift=self.scheduler.config.get("max_shift", 1.16),
        )
        
        # Set timesteps with dynamic shifting
        scheduler_kwargs = {"mu": mu} if self.scheduler.config.get("use_dynamic_shifting", False) else {}
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device, **scheduler_kwargs)
        timesteps = self.scheduler.timesteps
        
        # Step 4: Add noise using scheduler's noise schedule
        # Fixed: Use timestep_to_extract for consistency
        noise = torch.randn_like(latents)
        t = timesteps[self.timestep_to_extract]
        
        # Get sigma from scheduler for proper noise addition
        # In flow matching: x(t) = (1-sigma)*x_0 + sigma*noise
        # The scheduler handles this through its timesteps
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        
        # Step 5: Run through transformer and extract features
        activations = {}
        current_timestep = None
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        # Register hook on last transformer block with proper error handling
        hook_handle = None
        if hasattr(self.transformer, 'transformer_blocks') and len(self.transformer.transformer_blocks) > 0:
            last_block = self.transformer.transformer_blocks[-1]
            hook_handle = last_block.register_forward_hook(hook_fn('last_block'))
        elif hasattr(self.transformer, 'blocks') and len(self.transformer.blocks) > 0:
            last_block = self.transformer.blocks[-1]
            hook_handle = last_block.register_forward_hook(hook_fn('last_block'))
        else:
            raise ValueError(
                "Transformer must have 'transformer_blocks' or 'blocks' attribute with at least one block. "
                f"Found attributes: {[attr for attr in dir(self.transformer) if not attr.startswith('_')]}"
            )
        
        # Run denoising steps
        latents_flow = noisy_latents
        
        for i, t in enumerate(timesteps[:self.timestep_to_extract + 1]):
            if i == self.timestep_to_extract:
                current_timestep = t
            
            # Expand timestep properly
            timestep_expanded = t.expand(batch_size)
            
            # Call transformer with proper parameters
            noise_pred = self.transformer(
                hidden_states=latents_flow,
                timestep=timestep_expanded,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,  # Can be extended for IP-Adapter support
                return_dict=False
            )[0]
            
            # Step along rectified flow
            latents_flow = self.scheduler.step(
                noise_pred,
                t,
                latents_flow,
                return_dict=False
            )[0]
        
        # Remove hook
        if hook_handle:
            hook_handle.remove()
        
        # Extract features with proper error handling
        if 'last_block' not in activations:
            raise RuntimeError(
                "Failed to capture transformer activations. "
                "The hook may not have been triggered or the transformer architecture is different than expected."
            )
        
        features = activations['last_block']
        
        # Convert to (B, N, D) format if needed
        if features.dim() == 4:
            B, C, H, W = features.shape
            features = features.reshape(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        elif features.dim() == 3:
            # Already in (B, N, D) format
            pass
        else:
            raise ValueError(
                f"Unexpected feature dimensions: {features.shape}. "
                f"Expected 3D (B, N, D) or 4D (B, C, H, W) tensor."
            )
        
        # Step 6: Sample spatial positions (1/16) from features
        B, N, D = features.shape
        num_positions = max(1, int(N * self.sampling_ratio))
        
        # Randomly sample positions
        indices = torch.randperm(N, device=features.device)[:num_positions]
        sampled_features = features[:, indices, :]  # (B, num_positions, D)
        
        # Step 7: Apply timestep modulation
        if current_timestep is not None:
            # Handle scalar timestep properly
            if current_timestep.dim() == 0:
                timestep_batch = current_timestep.unsqueeze(0).repeat(batch_size)
            else:
                timestep_batch = current_timestep.repeat(batch_size) if current_timestep.shape[0] == 1 else current_timestep
            sampled_features = self.timestep_modulation(sampled_features, timestep_batch)
        
        return {
            'sampled_features': sampled_features,
            'current_timestep': current_timestep,
            'num_sampled_positions': num_positions
        }
    
    def eval(self):
        """Set to evaluation mode."""
        self.transformer.eval()
        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.text_encoder_3.eval()
        self.timestep_modulation.eval()
        return self
    
    def to(self, device):
        """Move to device."""
        self.transformer.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)
        self.text_encoder_2.to(device)
        self.text_encoder_3.to(device)
        self.timestep_modulation.to(device)
        self.device = device
        return self
    
    def half(self):
        """Convert to half precision."""
        self.transformer.half()
        self.vae.half()
        self.text_encoder.half()
        self.text_encoder_2.half()
        self.text_encoder_3.half()
        self.timestep_modulation.half()
        return self


class StableDiffusion(BaseModel):
    """
    Concrete class for Stable Diffusion 3 MMDiT transformer feature extraction.

    Extracts raw transformer activations during the rectified flow matching 
    process for training Sparse Autoencoders (SAEs). Returns timestep-modulated
    features sampled at 1/16 spatial positions for efficiency.
    """

    def __init__(self, use_half=False, device='cpu', sampling_ratio=1/16):
        super().__init__(use_half, device)
        
        print("Loading Stable Diffusion 3 pipeline...")
        print(f"Using rectified flow with FlowMatchEulerDiscreteScheduler")
        print(f"Token sampling ratio: {sampling_ratio}")
        
        # Load SD3 pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16 if use_half else torch.float32
        )
        
        # Extract components
        transformer = pipe.transformer
        vae = pipe.vae
        text_encoder = pipe.text_encoder
        text_encoder_2 = pipe.text_encoder_2
        text_encoder_3 = pipe.text_encoder_3
        tokenizer = pipe.tokenizer
        tokenizer_2 = pipe.tokenizer_2
        tokenizer_3 = pipe.tokenizer_3
        
        # Use FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        
        # Create wrapper
        self.model = SD3TransformerWrapper(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            use_half=use_half,
            device=device,
            sampling_ratio=sampling_ratio
        ).eval().to(self.device)
        
        if self.use_half:
            self.model = self.model.half()

        # SD3 preprocessing
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
        Extract raw transformer activations for SAE training.
        """
        with torch.no_grad():
            if self.use_half:
                x = x.half()
            return self.model.forward_features(x)
