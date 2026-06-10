"""
Activation extraction from Stable Diffusion 3 and FLUX.1 transformers during denoising.

Both models use rectified flow (straight-line paths between data and noise).
This module extracts transformer activations at each denoising timestep.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from torchvision import transforms


@dataclass
class ActivationOutput:
    """Container for extracted activations across timesteps."""
    
    # List of activations, one per timestep. Each tensor is (B, N, D)
    activations: List[torch.Tensor]
    
    # Corresponding timesteps for each activation
    timesteps: List[torch.Tensor]
    
    # Corresponding sigma values for each timestep
    sigmas: List[float]
    
    # Original latents before noise addition
    clean_latents: torch.Tensor
    
    # Final denoised latents
    denoised_latents: torch.Tensor


class BaseActivationExtractor(ABC, nn.Module):
    """Base class for extracting transformer activations during denoising."""
    
    def __init__(
        self,
        model_id: str,
        num_inference_steps: int = 4,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.dtype = dtype
        
        # To be set by subclasses
        self.pipe = None
        self.transformer = None
        self.vae = None
        self.scheduler = None
        
    @abstractmethod
    def _load_pipeline(self):
        """Load the diffusion pipeline and extract components."""
        pass
    
    @abstractmethod
    def _encode_null_prompt(self, batch_size: int) -> dict:
        """Encode null/empty prompt for unconditional generation."""
        pass
    
    @abstractmethod
    def _get_transformer_input(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: dict,
    ) -> dict:
        """Prepare inputs for the transformer forward pass."""
        pass
    
    @abstractmethod
    def _get_last_block(self) -> nn.Module:
        """Return the last transformer block to hook."""
        pass
    
    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space using VAE."""
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            
            shift = getattr(self.vae.config, "shift_factor", None)
            scale = self.vae.config.scaling_factor

            if shift is not None:
                latents = (latents - shift) * scale
            else:
                latents = latents * scale
            """ # Apply VAE scaling
            if hasattr(self.vae.config, 'shift_factor'):
                # SD3/FLUX style: latent = (sample - shift) * scale
                latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            else:
                # Standard VAE scaling
                latents = latents * self.vae.config.scaling_factor"""
                
        return latents
    
    def _add_noise_flow_matching(
        self, 
        sample: torch.Tensor, 
        noise: torch.Tensor, 
        sigma: float
    ) -> torch.Tensor:
        """
        Add noise using flow matching formula.
        
        In flow matching / rectified flow:
            x_t = (1 - sigma) * x_0 + sigma * noise
        
        Where sigma=0 is clean and sigma=1 is pure noise.
        """
        return (1.0 - sigma) * sample + sigma * noise
    
    @torch.no_grad()
    def extract_activations(self, image: torch.Tensor) -> ActivationOutput:
        """
        Extract transformer activations during the full denoising process.
        
        Args:
            image: Input image tensor of shape (B, C, H, W), normalized to [-1, 1]
            
        Returns:
            ActivationOutput containing activations at each timestep
        """
        batch_size = image.shape[0]
        image = image.to(device=self.device, dtype=self.dtype)
        
        # Step 1: Encode image to latent space
        clean_latents = self._encode_image(image)
        
        # Step 2: Get null prompt embeddings
        prompt_embeds = self._encode_null_prompt(batch_size)
        
        # Step 3: Set up scheduler timesteps
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas
        
        # Step 4: Add noise at the highest noise level (first sigma)
        # In flow matching, sigmas[0] is the highest noise level
        noise = torch.randn_like(clean_latents)
        initial_sigma = float(sigmas[0])
        noisy_latents = self._add_noise_flow_matching(clean_latents, noise, initial_sigma)
        
        # Step 5: Set up hook to capture activations
        activations_list = []
        timesteps_list = []
        sigmas_list = []
        
        def hook_fn(module, input, output):
            # Handle different output formats:
            # - SD3 JointTransformerBlock returns (encoder_hidden_states, hidden_states)
            #   where encoder_hidden_states can be None
            # - FLUX single blocks return just hidden_states
            # - Some blocks return tuples with hidden_states at different positions
            
            if output is None:
                return
            
            if isinstance(output, tuple):
                # SD3 blocks return (encoder_hidden_states, hidden_states)
                # We want hidden_states, which is typically the last non-None element
                # or specifically at index 1 for SD3
                activation = None
                for item in reversed(output):
                    if item is not None and isinstance(item, torch.Tensor):
                        activation = item
                        break
                if activation is None:
                    return
            else:
                activation = output
            
            activations_list.append(activation.clone())
        
        # Register hook on the last transformer block
        last_block = self._get_last_block().attn2
        hook_handle = last_block.register_forward_hook(hook_fn)
        
        # Step 6: Denoise step by step, capturing activations at each step
        latents = noisy_latents
        
        try:
            for i, t in enumerate(timesteps):
                # Prepare transformer inputs
                transformer_inputs = self._get_transformer_input(latents, t, prompt_embeds)
                
                # Forward pass through transformer (hook captures activation)
                #noise_pred = self.transformer(**transformer_inputs, return_dict=False)[0]
                model_output = self.transformer(**transformer_inputs, return_dict=False)[0]

                noise_pred = self._process_model_output(model_output, latents)

                # Record the timestep and sigma
                timesteps_list.append(t.clone())
                sigmas_list.append(float(sigmas[i]))
                
                # Step along the flow
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
        finally:
            # Always remove hook
            hook_handle.remove()
        
        return ActivationOutput(
            activations=activations_list,
            timesteps=timesteps_list,
            sigmas=sigmas_list,
            clean_latents=clean_latents,
            denoised_latents=latents,
        )
    def _process_model_output(self, model_output, latents):
        
        return model_output


class SD3ActivationExtractor(BaseActivationExtractor):
    """
    Extract activations from Stable Diffusion 3 MMDiT transformer.
    
    SD3 uses a Multimodal Diffusion Transformer (MMDiT) with joint attention
    between image and text tokens. It uses three text encoders: CLIP-L, CLIP-G, and T5.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        num_inference_steps: int = 28,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(model_id, num_inference_steps, device, dtype)
        self._load_pipeline()
        
        # SD3 preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def _load_pipeline(self):
        """Load SD3 pipeline components."""
        from diffusers import StableDiffusion3Pipeline
        
        print(f"Loading SD3 pipeline from {self.model_id}...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        
        # Text encoders and tokenizers
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2
        self.text_encoder_3 = self.pipe.text_encoder_3
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer_2 = self.pipe.tokenizer_2
        self.tokenizer_3 = self.pipe.tokenizer_3
        
        # Set to eval mode
        self.transformer.eval()
        self.vae.eval()
        
        print(f"SD3 loaded. Transformer blocks: {len(self.transformer.transformer_blocks)}")
    
    def _encode_null_prompt(self, batch_size: int) -> dict:
        """Encode empty prompt using all three text encoders."""
        null_prompt = [""] * batch_size
        
        # Tokenize for CLIP-L
        text_inputs_1 = self.tokenizer(
            null_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        
        # Tokenize for CLIP-G
        text_inputs_2 = self.tokenizer_2(
            null_prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        
        # Tokenize for T5
        text_inputs_3 = self.tokenizer_3(
            null_prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        
        # Encode with CLIP-L
        prompt_embeds_1 = self.text_encoder(text_inputs_1, output_hidden_states=True)
        pooled_prompt_embeds_1 = prompt_embeds_1[0]
        prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]
        
        # Encode with CLIP-G
        prompt_embeds_2 = self.text_encoder_2(text_inputs_2, output_hidden_states=True)
        pooled_prompt_embeds_2 = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
        
        # Encode with T5
        prompt_embeds_3 = self.text_encoder_3(text_inputs_3)[0]
        
        # Concatenate CLIP embeddings along feature dimension
        clip_prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        # Pad CLIP embeddings to match T5 dimension
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, prompt_embeds_3.shape[-1] - clip_prompt_embeds.shape[-1]),
        )
        
        # Concatenate all embeddings along sequence dimension
        prompt_embeds = torch.cat([clip_prompt_embeds, prompt_embeds_3], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)
        
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
    
    def _get_transformer_input(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: dict,
    ) -> dict:
        """Prepare inputs for SD3 transformer."""
        batch_size = latents.shape[0]
        
        return {
            "hidden_states": latents,
            "timestep": timestep.expand(batch_size),
            "encoder_hidden_states": prompt_embeds["prompt_embeds"],
            "pooled_projections": prompt_embeds["pooled_prompt_embeds"],
        }
    
    def _get_last_block(self) -> nn.Module:
        """Return the last MMDiT block."""
        return self.transformer.transformer_blocks[-1]


class FLUXActivationExtractor(BaseActivationExtractor):
    """
    Extract activations from FLUX.1 transformer.
    
    FLUX uses a hybrid architecture with dual-stream MMDiT blocks followed by
    single-stream DiT blocks. It uses CLIP and T5 text encoders.
    
    FLUX.1-schnell is distilled for 1-4 steps with guidance_scale=0.
    """
    
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        num_inference_steps: int = 4,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(model_id, num_inference_steps, device, dtype)
        self._load_pipeline()
        
        # FLUX preprocessing (typically 1024x1024 but can vary)
        self.preprocess = transforms.Compose([
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def _load_pipeline(self):
        """Load FLUX pipeline components."""
        from diffusers import FluxPipeline
        
        print(f"Loading FLUX pipeline from {self.model_id}...")
        self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        
        # Text encoders and tokenizers (CLIP + T5)
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer_2 = self.pipe.tokenizer_2
        
        # Set to eval mode
        self.transformer.eval()
        self.vae.eval()
        
        num_dual = len(self.transformer.transformer_blocks)
        num_single = len(self.transformer.single_transformer_blocks)
        print(f"FLUX loaded. Dual-stream blocks: {num_dual}, Single-stream blocks: {num_single}")
    
    def _encode_null_prompt(self, batch_size: int) -> dict:
        """Encode empty prompt using CLIP and T5."""
        null_prompt = [""] * batch_size
        
        # Tokenize for CLIP
        text_inputs = self.tokenizer(
            null_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Tokenize for T5
        text_inputs_2 = self.tokenizer_2(
            null_prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        
        # Encode with CLIP (pooled output)
        pooled_prompt_embeds = self.text_encoder(
            text_inputs.input_ids, output_hidden_states=False
        ).pooler_output
        
        # Encode with T5 (sequence output)
        prompt_embeds = self.text_encoder_2(text_inputs_2)[0]
        
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
    
    def _prepare_latent_image_ids(self, latents: torch.Tensor) -> torch.Tensor:
        """Prepare image position IDs for FLUX."""
        batch_size, channels, height, width = latents.shape
        
        # FLUX packs latents into 2x2 patches
        latent_height = height // 2
        latent_width = width // 2
        
        # Create position grid
        latent_image_ids = torch.zeros(latent_height, latent_width, 3, device=self.device)
        latent_image_ids[..., 1] = torch.arange(latent_height, device=self.device)[:, None]
        latent_image_ids[..., 2] = torch.arange(latent_width, device=self.device)[None, :]
        
        # Flatten and expand for batch
        latent_image_ids = latent_image_ids.reshape(
            latent_height * latent_width, 3
        ).unsqueeze(0).expand(batch_size, -1, -1)
        
        return latent_image_ids
    
    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack latents into 2x2 patches for FLUX."""
        batch_size, channels, height, width = latents.shape
        
        # Reshape: (B, C, H, W) -> (B, C, H//2, 2, W//2, 2)
        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        
        # Permute and flatten patches: -> (B, H//2 * W//2, C * 4)
        latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
        latents = latents.view(batch_size, (height // 2) * (width // 2), channels * 4)
        
        return latents
    
    def _unpack_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Unpack latents from 2x2 patches."""
        batch_size = latents.shape[0]
        channels = latents.shape[-1] // 4
        
        latents = latents.view(batch_size, height // 2, width // 2, channels, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5).contiguous()
        latents = latents.view(batch_size, channels, height, width)
        
        return latents
    
    def _get_transformer_input(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: dict,
    ) -> dict:
        """Prepare inputs for FLUX transformer."""
        batch_size = latents.shape[0]
        height, width = latents.shape[2], latents.shape[3]
        
        # Pack latents into patches
        packed_latents = self._pack_latents(latents)
        
        # Prepare position IDs
        img_ids = self._prepare_latent_image_ids(latents)
        txt_ids = torch.zeros(
            batch_size,
            prompt_embeds["prompt_embeds"].shape[1],
            3,
            device=self.device,
        )
        
        return {
            "hidden_states": packed_latents,
            "timestep": timestep.expand(batch_size) / 1000,  # FLUX expects normalized timesteps
            "encoder_hidden_states": prompt_embeds["prompt_embeds"],
            "pooled_projections": prompt_embeds["pooled_prompt_embeds"],
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": None,  # Schnell doesn't use guidance
        }
    
    def _get_last_block(self) -> nn.Module:
        """Return the last single-stream transformer block."""
        return self.transformer.single_transformer_blocks[-1]
    
    @torch.no_grad()
    def extract_activations(self, image: torch.Tensor) -> ActivationOutput:
        """
        Extract transformer activations during the full denoising process.
        
        Overridden to handle FLUX's latent packing/unpacking.
        """
        batch_size = image.shape[0]
        image = image.to(device=self.device, dtype=self.dtype)
        
        # Step 1: Encode image to latent space
        clean_latents = self._encode_image(image)
        height, width = clean_latents.shape[2], clean_latents.shape[3]
        
        # Step 2: Get null prompt embeddings
        prompt_embeds = self._encode_null_prompt(batch_size)
        
        # Step 3: Set up scheduler timesteps
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas
        
        # Step 4: Add noise at the highest noise level using flow matching formula
        noise = torch.randn_like(clean_latents)
        initial_sigma = float(sigmas[0])
        noisy_latents = self._add_noise_flow_matching(clean_latents, noise, initial_sigma)
        
        # Step 5: Set up hook to capture activations
        activations_list = []
        timesteps_list = []
        sigmas_list = []
        
        def hook_fn(module, input, output):
            # Handle different output formats
            if output is None:
                return
            
            if isinstance(output, tuple):
                # Find the last non-None tensor in the tuple
                activation = None
                for item in reversed(output):
                    if item is not None and isinstance(item, torch.Tensor):
                        activation = item
                        break
                if activation is None:
                    return
            else:
                activation = output
            
            activations_list.append(activation.clone())
        
        last_block = self._get_last_block()
        hook_handle = last_block.register_forward_hook(hook_fn)
        
        # Step 6: Denoise step by step
        latents = noisy_latents
        
        try:
            for i, t in enumerate(timesteps):
                # Prepare transformer inputs
                transformer_inputs = self._get_transformer_input(latents, t, prompt_embeds)
                
                # Forward pass
                noise_pred = self.transformer(**transformer_inputs, return_dict=False)[0]
                
                # Unpack noise prediction
                noise_pred = self._unpack_latents(noise_pred, height, width)
                
                timesteps_list.append(t.clone())
                sigmas_list.append(float(sigmas[i]))
                
                # Step along the flow
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
        finally:
            hook_handle.remove()
        
        return ActivationOutput(
            activations=activations_list,
            timesteps=timesteps_list,
            sigmas=sigmas_list,
            clean_latents=clean_latents,
            denoised_latents=latents,
        )

class PixArtActivationExtractor(BaseActivationExtractor):
    """
    Extract activations from Pixart diffuser model.
    

    """
    
    def __init__(
        self,
        model_id: str = "PixArt-alpha/PixArt-XL-2-1024-MS",
        num_inference_steps: int = 28,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(model_id, num_inference_steps, device, dtype)
        self._load_pipeline()
        
        # PixArt preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def _load_pipeline(self):
        """Load PixArt pipeline components."""
        from diffusers import PixArtAlphaPipeline, DDIMScheduler

        print(f"Loading PixArt pipeline from {self.model_id}...")
        self.pipe = PixArtAlphaPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True
        ).to(self.device)

        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae

        # Replace DPMSolverMultistepScheduler with DDIMScheduler.
        # DPMSolver's convert_model_output tries to broadcast its internal sigma/alpha
        # tensors against the latent spatial dims after the learned-variance split,
        # producing a shape mismatch (e.g. "tensor a (4) must match tensor b (8)").
        # DDIMScheduler handles epsilon-only predictions cleanly without that broadcast.
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # PixArtAlpha uses a single T5 text encoder
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

        # Set to eval mode
        self.transformer.eval()
        self.vae.eval()

        print(f"PixArt loaded. Transformer blocks: {len(self.transformer.transformer_blocks)}")
    
    def _encode_null_prompt(self, batch_size: int) -> dict:
        """Encode empty prompt using the one T5 text encoder."""
        null_prompt = [""] * batch_size
        
        # Tokenize for T5
        text_inputs_1 = self.tokenizer(
            null_prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        


        """# Tokenize for CLIP-G
        text_inputs_2 = self.tokenizer_2(
            null_prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        
        # Tokenize for T5
        text_inputs_3 = self.tokenizer_3(
            null_prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)"""
        
        """# Encode with CLIP-L
        prompt_embeds_1 = self.text_encoder(text_inputs_1, output_hidden_states=True)
        pooled_prompt_embeds_1 = prompt_embeds_1[0]
        prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]
        
        
        # Encode with CLIP-G
        prompt_embeds_2 = self.text_encoder_2(text_inputs_2, output_hidden_states=True)
        pooled_prompt_embeds_2 = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]"""
        
        # Encode with T5
        prompt_embeds_3 = self.text_encoder(text_inputs_1)[0]
        
        """# Concatenate CLIP embeddings along feature dimension
        clip_prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        # Pad CLIP embeddings to match T5 dimension
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, prompt_embeds_3.shape[-1] - clip_prompt_embeds.shape[-1]),
        )
        
        # Concatenate all embeddings along sequence dimension
        prompt_embeds = torch.cat([clip_prompt_embeds, prompt_embeds_3], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)"""
        
        return {
            "prompt_embeds": prompt_embeds_3,
            #"pooled_prompt_embeds": pooled_prompt_embeds,
        }
    


    def _get_transformer_input(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: dict,
    ) -> dict:
        """Prepare inputs for PixArt transformer."""
        batch_size = latents.shape[0]
        B, C, H, W = latents.shape
        
        aspect_ratio = torch.tensor([H / W], device=self.device).unsqueeze(0).repeat(B, 1)

        added_cond_kwargs = {
        "resolution": torch.tensor([H, W], device=self.device).unsqueeze(0).repeat(B, 1),
        "aspect_ratio": aspect_ratio,
        }

        prompt_embeds["prompt_embeds"] = prompt_embeds["prompt_embeds"].to(self.dtype)

        """added_cond_kwargs=self.pipe.prepare_added_cond_kwargs(
            prompt_embeds = prompt_embeds["prompt_embeds"],

        )"""
        
        return {
            "hidden_states": latents,
            "timestep": timestep.expand(batch_size).to(self.dtype),
            "encoder_hidden_states": prompt_embeds["prompt_embeds"],
            "added_cond_kwargs": added_cond_kwargs,
        }
    def _process_model_output(self, model_output, latents):
        # PixArt outputs (B, 2*C, H, W): [epsilon | learned_variance_logits].
        # DDIMScheduler only wants the epsilon half (B, C, H, W).
        if model_output.shape[1] == 2 * latents.shape[1]:
            model_output, _ = model_output.chunk(2, dim=1)
        return model_output

    # Which transformer block to hook, as a fraction of total depth.
    #   1.0 -> last block (original behaviour, noise-prediction-dominated)
    #   0.5 -> middle block (Option A: richer, higher-rank semantic residual stream)
    HOOK_DEPTH_FRAC = 8 / 27

    def _get_last_block(self) -> nn.Module:
        blocks = self.transformer.transformer_blocks
        n_blocks = len(blocks)
        idx = int(round(self.HOOK_DEPTH_FRAC * (n_blocks - 1)))
        idx = max(0, min(idx, n_blocks - 1))
        print(f"[PixArt] Hooking transformer block {idx}/{n_blocks - 1} "
              f"(depth_frac={self.HOOK_DEPTH_FRAC})")
        return blocks[idx]

    def _get_ddim_sigmas(self) -> list:
        """
        DDIMScheduler has no .sigmas attribute. Derive equivalent sigma values
        from its alphas_cumprod: sigma = sqrt((1 - alpha) / alpha), which is the
        standard DDPM/DDIM noise level parameterisation.
        Returns a list of floats, one per scheduled timestep, plus a trailing 0.0
        to mirror the len(sigmas) == len(timesteps) + 1 convention used by flow
        schedulers (so sigmas[i] lines up with timestep i throughout the loop).
        """
        alphas_cumprod = self.scheduler.alphas_cumprod  # (num_train_timesteps,)
        timesteps = self.scheduler.timesteps             # scheduled inference steps
        sigmas = []
        for t in timesteps:
            alpha = alphas_cumprod[int(t)].item()
            sigmas.append(float(((1 - alpha) / alpha) ** 0.5))
        sigmas.append(0.0)   # terminal sigma (clean image)
        return sigmas

    @torch.no_grad()
    def extract_activations(self, image: torch.Tensor) -> ActivationOutput:
        """
        Overridden for PixArt / DDIMScheduler, which does not expose .sigmas.
        The flow-matching noise-addition formula is also replaced by the standard
        DDPM forward-process: x_t = sqrt(alpha_t)*x_0 + sqrt(1-alpha_t)*eps.
        """
        batch_size = image.shape[0]
        image = image.to(device=self.device, dtype=self.dtype)

        # Step 1: Encode image to latent space
        clean_latents = self._encode_image(image)

        # Step 2: Get null prompt embeddings
        prompt_embeds = self._encode_null_prompt(batch_size)

        # Step 3: Set up scheduler timesteps
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = self._get_ddim_sigmas()   # <-- derived, not scheduler.sigmas

        # Step 4: Add noise at the highest noise level using DDPM forward process
        noise = torch.randn_like(clean_latents)
        initial_t = int(timesteps[0])
        alpha = self.scheduler.alphas_cumprod[initial_t].item()
        noisy_latents = (alpha ** 0.5) * clean_latents + ((1 - alpha) ** 0.5) * noise

        # Step 5: Set up hook to capture activations
        activations_list = []
        timesteps_list = []
        sigmas_list = []

        def hook_fn(module, input, output):
            if output is None:
                return
            if isinstance(output, tuple):
                activation = None
                for item in reversed(output):
                    if item is not None and isinstance(item, torch.Tensor):
                        activation = item
                        break
                if activation is None:
                    return
            else:
                activation = output
            activations_list.append(activation.clone())

        last_block = self._get_last_block()
        hook_handle = last_block.register_forward_hook(hook_fn)

        # Step 6: Denoise step by step, capturing activations at each step
        latents = noisy_latents

        try:
            for i, t in enumerate(timesteps):
                transformer_inputs = self._get_transformer_input(latents, t, prompt_embeds)
                model_output = self.transformer(**transformer_inputs, return_dict=False)[0]
                noise_pred = self._process_model_output(model_output, latents)

                timesteps_list.append(t.clone())
                sigmas_list.append(sigmas[i])   # per-step sigma from _get_ddim_sigmas

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        finally:
            hook_handle.remove()

        if not activations_list:
            raise RuntimeError(
                f"[PixArt] No activations captured after {self.num_inference_steps} steps. "
                "The forward hook may not have fired — check that the hooked transformer block exists and produces an output tensor."
            )
        print(f"[PixArt] Captured {len(activations_list)} activations, "
              f"each shape {tuple(activations_list[0].shape)}")

        return ActivationOutput(
            activations=activations_list,
            timesteps=timesteps_list,
            sigmas=sigmas_list,
            clean_latents=clean_latents,
            denoised_latents=latents,
        )


# Convenience functions
def create_sd3_extractor(
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    num_steps: int = 28,
) -> SD3ActivationExtractor:
    """Create an SD3 activation extractor."""
    return SD3ActivationExtractor(
        model_id="stabilityai/stable-diffusion-3-medium-diffusers",
        num_inference_steps=num_steps,
        device=device,
        dtype=dtype,
    )


def create_flux_extractor(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_steps: int = 4,
) -> FLUXActivationExtractor:
    """Create a FLUX.1-schnell activation extractor."""
    return FLUXActivationExtractor(
        model_id="black-forest-labs/FLUX.1-schnell",
        num_inference_steps=num_steps,
        device=device,
        dtype=dtype,
    )

def create_PixArt_extractor(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_steps: int = 15,    
) -> PixArtActivationExtractor:
    """creates a PixArt Activation extractor"""
    return PixArtActivationExtractor(
        model_id="PixArt-alpha/PixArt-XL-2-1024-MS",
        num_inference_steps=num_steps,
        device=device,
        dtype=dtype,
    )

if __name__ == "__main__":
    # Example usage
    from PIL import Image
    import requests
    from io import BytesIO
    
    # Load a sample image
    img = Image.open("./curvature/320px-Camponotus_flavomarginatus_ant.jpg").convert("RGB")
    
    print("=" * 60)
    print("Testing SD3 Activation Extractor")
    print("=" * 60)
    
    # Create SD3 extractor
    sd3_extractor = create_sd3_extractor(device="cuda", num_steps=5)
    
    # Preprocess image
    img_tensor = sd3_extractor.preprocess(img).unsqueeze(0)
    
    # Extract activations
    output = sd3_extractor.extract_activations(img_tensor)
    
    print(f"\nExtracted {len(output.activations)} activations")
    for i, (act, t, s) in enumerate(zip(output.activations, output.timesteps, output.sigmas)):
        print(f"  Step {i}: timestep={t.item():.3f}, sigma={s:.4f}, activation shape={act.shape}")
    
    print(f"\nClean latents shape: {output.clean_latents.shape}")
    print(f"Denoised latents shape: {output.denoised_latents.shape}")
    
    print("\n" + "=" * 60)
    print("Testing FLUX Activation Extractor")
    print("=" * 60)
    
    # Create FLUX extractor
    flux_extractor = create_flux_extractor(device="cuda", num_steps=4)
    
    # Preprocess image
    img_tensor = flux_extractor.preprocess(img).unsqueeze(0)
    
    # Extract activations
    output = flux_extractor.extract_activations(img_tensor)
    
    print(f"\nExtracted {len(output.activations)} activations")
    for i, (act, t, s) in enumerate(zip(output.activations, output.timesteps, output.sigmas)):
        print(f"  Step {i}: timestep={t.item():.3f}, sigma={s:.4f}, activation shape={act.shape}")
    
    print(f"\nClean latents shape: {output.clean_latents.shape}")
    print(f"Denoised latents shape: {output.denoised_latents.shape}")
