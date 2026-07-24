from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn



def _ensure_bt(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Ensure timestep/noise conditioning is shaped (B, T).
    """
    values = values.squeeze()
    if values.dim() == 1:
        values = values.unsqueeze(0).expand(batch_size, -1)
    if values.dim() != 2 or values.shape[0] != batch_size:
        raise ValueError(
            f"Expected timestep values shaped (B, T), got {tuple(values.shape)} for B={batch_size}"
        )
    return values


def _select_inference_activations(
    x: torch.Tensor,
    source: str,
    is_diffusion: bool,
    timestep_values: Optional[torch.Tensor] = None,
    timestep_idx: Optional[int] = None,
    layer_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Normalize source activations into one SAE-ready slice.

    Supported layouts:
      vision:
        - (B, N, D)
        - (B, L, N, D)
      diffusion:
        - (B, T, N, D)
        - (B, L, T, N, D)

    Returns:
        x_slice: (B, N, D)
        sigma: (B,) timestep/noise values for diffusion models, else None
    """
    if is_diffusion:
        if x.dim() == 5:
            total_layers = x.shape[1]
            selected_layer = 0 if layer_idx is None else layer_idx
            if not 0 <= selected_layer < total_layers:
                raise ValueError(
                    f"layer_idx={selected_layer} is out of range for source '{source}' "
                    f"with {total_layers} layers"
                )
            x = x[:, selected_layer]
        elif x.dim() != 4:
            raise ValueError(
                f"Expected diffusion activations for source '{source}' to be shaped "
                f"(B, T, N, D) or (B, L, T, N, D), got {tuple(x.shape)}"
            )

        if timestep_values is None:
            raise ValueError(
                f"Source '{source}' is a diffusion model and requires timestep_values "
                "shaped (B, T) or (T,) for inference."
            )

        timesteps_bt = _ensure_bt(timestep_values, x.shape[0]).to(x.device)
        total_steps = x.shape[1]
        selected_timestep = total_steps - 1 if timestep_idx is None else timestep_idx
        if not 0 <= selected_timestep < total_steps:
            raise ValueError(
                f"timestep_idx={selected_timestep} is out of range for source '{source}' "
                f"with {total_steps} timesteps"
            )
        return x[:, selected_timestep], timesteps_bt[:, selected_timestep]

    if x.dim() == 4:
        total_layers = x.shape[1]
        selected_layer = 0 if layer_idx is None else layer_idx
        if not 0 <= selected_layer < total_layers:
            raise ValueError(
                f"layer_idx={selected_layer} is out of range for source '{source}' "
                f"with {total_layers} layers"
            )
        x = x[:, selected_layer]
    elif x.dim() != 3:
        raise ValueError(
            f"Expected vision activations for source '{source}' to be shaped "
            f"(B, N, D) or (B, L, N, D), got {tuple(x.shape)}"
        )

    return x, None


def _score_latents(z: torch.Tensor) -> torch.Tensor:
    """
    Convert latent activations into one score per feature.

    Supported latent shapes:
      - (B, K)
      - (B, N, K)

    Returns:
        scores: (K,)
    """
    if z.dim() == 2:
        # Average over batch
        return z.abs().mean(dim=0)

    if z.dim() == 3:
        # Average over batch and token dimensions
        return z.abs().amax(dim=(0, 1))

    raise ValueError(
        f"Unsupported latent shape {tuple(z.shape)}. Expected (B, K) or (B, N, K)."
    )


@torch.no_grad()
def top_features_from_activations(
    model: nn.Module,
    image_activations: torch.Tensor,
    source: str,
    top_k: int = 5,
    device: str = "cuda",
    timestep_values: Optional[torch.Tensor] = None,
    timestep_idx: Optional[int] = None,
    layer_idx: Optional[int] = None,
    spatial_aligner=None,
) -> Tuple[List[Dict[str, float]], torch.Tensor]:
    """
    Run Universal SAE inference on precomputed image activations and return the
    top-k latent features.

    Args:
        model: UniversalSAE
        image_activations: Activations shaped:
            - vision: (B, N, D) or (B, L, N, D)
            - diffusion: (B, T, N, D) or (B, L, T, N, D)
        source: Source vision model name
        top_k: Number of top features to return
        device: Inference device
        timestep_values: Diffusion timestep/noise values shaped (B, T) or (T,)
        timestep_idx: Optional timestep index to score for diffusion sources.
            Defaults to the final timestep in image_activations.
        layer_idx: Optional layer index for multi-layer activation tensors.
            Defaults to layer 0.
        spatial_aligner: Optional SpatialAligner used during training. Must be
            passed at eval time if it was used at training time, or features
            won't match the trained dictionary.

    Returns:
        features: List of {"feature_idx": int, "score": float}
        z: Raw latent activations returned by the encoder
    """
    model.eval()

    activations = image_activations.to(device)
    is_diffusion = source in getattr(model, "diffusion_models", set())
    x, sigma = _select_inference_activations(
        activations,
        source=source,
        is_diffusion=is_diffusion,
        timestep_values=timestep_values,
        timestep_idx=timestep_idx,
        layer_idx=layer_idx,
    )
    if spatial_aligner is not None:
        x = spatial_aligner.align(x, source=source)
    _z_pre, z = model.encode(x, source=source, sigma=sigma)

    scores = _score_latents(z)
    k = min(top_k, scores.numel())
    top_scores, top_indices = torch.topk(scores, k=k)

    features = [
        {"feature_idx": int(idx.item()), "score": float(score.item())}
        for idx, score in zip(top_indices, top_scores)
    ]
    return features, z


@torch.no_grad()
def top_features_from_image(
    model: nn.Module,
    image: torch.Tensor,
    source: str,
    activation_extractor: Callable[
        [torch.Tensor, str],
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ],
    top_k: int = 5,
    device: str = "cuda",
    timestep_idx: Optional[int] = None,
    layer_idx: Optional[int] = None,
    spatial_aligner=None,
) -> Tuple[List[Dict[str, float]], torch.Tensor]:
    """
    End-to-end inference for a raw image tensor.

    The SAE does not operate directly on pixels, so you must supply an
    activation_extractor that converts the image into source-model activations.
    It may return either:
      - activations only
      - (activations, timestep_values) for diffusion sources

    Args:
        model: UniversalSAE
        image: Raw image tensor shaped (C, H, W) or (B, C, H, W)
        source: Source vision model name
        activation_extractor: Callable returning activations or
            (activations, timestep_values)
        top_k: Number of top features to return
        device: Inference device
        timestep_idx: Optional timestep index to score for diffusion sources.
        layer_idx: Optional layer index for multi-layer activation tensors.

    Returns:
        features: List of {"feature_idx": int, "score": float}
        z: Raw latent activations returned by the encoder
    """
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)
    elif image.dim() != 4:
        raise ValueError(
            f"Expected image tensor shaped (C, H, W) or (B, C, H, W), got {tuple(image.shape)}"
        )

    image = image.to(device)
    extracted = activation_extractor(image, source)
    if isinstance(extracted, tuple):
        image_activations, timestep_values = extracted
    else:
        image_activations, timestep_values = extracted, None

    return top_features_from_activations(
        model=model,
        image_activations=image_activations,
        source=source,
        top_k=top_k,
        device=device,
        timestep_values=timestep_values,
        timestep_idx=timestep_idx,
        layer_idx=layer_idx,
        spatial_aligner=spatial_aligner,
    )


def print_top_features(features: List[Dict[str, float]]) -> None:
    # Print the ranked features.
    for rank, feature in enumerate(features, start=1):
        print(
            f"{rank}. feature {feature['feature_idx']} "
            f"(score={feature['score']:.6f})"
        )

