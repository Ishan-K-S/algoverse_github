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
    def __init__(self, hidden_dim, max_timestep=1.0):
        super().__init__()
        self.scale_net = nn.Linear(1, hidden_dim)
        self.shift_net = nn.Linear(1, hidden_dim)
        self.max_timestep = max_timestep

        nn.init.ones_(self.scale_net.weight)
        nn.init.zeros_(self.scale_net.bias)
        nn.init.zeros_(self.shift_net.weight)
        nn.init.zeros_(self.shift_net.bias)

    def forward(self, x, timestep):
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(-1)

        timestep = timestep.float() / self.max_timestep
        scale = self.scale_net(timestep).unsqueeze(1)
        shift = self.shift_net(timestep).unsqueeze(1)
        return x * (1 + scale) + shift


class SD3TransformerWrapper(nn.Module):
    def __init__(
        self,
        transformer,
        vae,
        scheduler,
        text_encoder,
        text_encoder_2,
        text_encoder_3,
        tokenizer,
        tokenizer_2,
        tokenizer_3,
        use_half=False,
        device="cpu",
        sampling_ratio=1 / 16,
    ):
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3

        self.use_half = use_half
        self.device = device
        self.sampling_ratio = sampling_ratio

        self.num_inference_steps = 5

        hidden_dim = transformer.config.joint_attention_dim
        self.patch_size = getattr(transformer.config, "patch_size", 2)

        self.timestep_modulation = TimestepModulation(hidden_dim, 1.0).to(device)
        if use_half:
            self.timestep_modulation = self.timestep_modulation.half()

    def encode_prompt(self, batch_size):
        null_prompt = [""] * batch_size

        text_inputs_1 = self.tokenizer(
            null_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)

        text_inputs_2 = self.tokenizer_2(
            null_prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)

        text_inputs_3 = self.tokenizer_3(
            null_prompt,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        ).input_ids.to(self.device)

        pe1 = self.text_encoder(text_inputs_1, output_hidden_states=True)
        pe2 = self.text_encoder_2(text_inputs_2, output_hidden_states=True)

        clip = torch.cat(
            [pe1.hidden_states[-2], pe2.hidden_states[-2]], dim=-1
        )
        t5 = self.text_encoder_3(text_inputs_3)[0]

        clip = torch.nn.functional.pad(
            clip, (0, t5.shape[-1] - clip.shape[-1])
        )

        prompt_embeds = torch.cat([clip, t5], dim=-2)
        pooled = torch.cat([pe1[0], pe2[0]], dim=-1)

        return prompt_embeds, pooled

    @torch.no_grad()
    def forward_features(self, x):
        B = x.shape[0]

        if self.use_half:
            x = x.half()

        latents = self.vae.encode(x).latent_dist.sample()
        latents = (
            latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(B)

        _, _, H, W = latents.shape
        image_seq_len = (H // self.patch_size) * (W // self.patch_size)

        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.16),
        )

        scheduler_kwargs = (
            {"mu": mu}
            if self.scheduler.config.get("use_dynamic_shifting", False)
            else {}
        )

        self.scheduler.set_timesteps(
            self.num_inference_steps, device=self.device, **scheduler_kwargs
        )
        timesteps = self.scheduler.timesteps

        # ✅ FIX 1: initialize from noise ONCE
        latents_flow = torch.randn_like(latents)

        activations = []

        def hook_fn(module, input, output):
            activations.append(output)

        if hasattr(self.transformer, "transformer_blocks"):
            last_block = self.transformer.transformer_blocks[-1]
        else:
            last_block = self.transformer.blocks[-1]

        hook_handle = last_block.register_forward_hook(hook_fn)

        # ✅ FIX 2: full denoising, extract every timestep
        for t in timesteps:
            timestep_expanded = t.expand(B)

            noise_pred = self.transformer(
                hidden_states=latents_flow,
                timestep=timestep_expanded,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            latents_flow = self.scheduler.step(
                noise_pred, t, latents_flow, return_dict=False
            )[0]

        hook_handle.remove()

        if len(activations) == 0:
            raise RuntimeError("No activations captured.")

        # (T, B, N, D)
        features = []
        for f in activations:
            if f.dim() == 4:
                B, C, H, W = f.shape
                f = f.reshape(B, C, H * W).transpose(1, 2)
            features.append(f)

        # spatial subsample
        T = len(features)
        B, N, D = features[0].shape
        k = max(1, int(N * self.sampling_ratio))
        idx = torch.randperm(N, device=features[0].device)[:k]

        features = torch.stack(
            [f[:, idx] for f in features], dim=0
        )  # (T, B, k, D)

        return {
            "features": features,
            "timesteps": timesteps,
            "num_sampled_positions": k,
        }


class StableDiffusion(BaseModel):
    def __init__(self, use_half=False, device="cpu", sampling_ratio=1 / 16):
        super().__init__(use_half, device)

        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16 if use_half else torch.float32,
        )

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

        self.model = (
            SD3TransformerWrapper(
                transformer=pipe.transformer,
                vae=pipe.vae,
                scheduler=scheduler,
                text_encoder=pipe.text_encoder,
                text_encoder_2=pipe.text_encoder_2,
                text_encoder_3=pipe.text_encoder_3,
                tokenizer=pipe.tokenizer,
                tokenizer_2=pipe.tokenizer_2,
                tokenizer_3=pipe.tokenizer_3,
                use_half=use_half,
                device=device,
                sampling_ratio=sampling_ratio,
            )
            .eval()
            .to(device)
        )

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def forward_features(self, x):
        return self.model.forward_features(x)
