from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _validate_image_activations(x: torch.Tensor, source: str) -> torch.Tensor:
    """
    Validate image activations for a vision source model.

    Expected shape: (B, N, D)
    """
    if x.dim() != 3:
        raise ValueError(
            f"Expected image activations for source '{source}' to be shaped (B, N, D), "
            f"got {tuple(x.shape)}"
        )
    return x


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
        return z.abs().mean(dim=(0, 1))

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
) -> Tuple[List[Dict[str, float]], torch.Tensor]:
    """
    Run Universal SAE inference on precomputed image activations and return the
    top-k latent features.

    Args:
        model: UniversalSAE
        image_activations: Vision activations shaped (B, N, D)
        source: Source vision model name
        top_k: Number of top features to return
        device: Inference device

    Returns:
        features: List of {"feature_idx": int, "score": float}
        z: Raw latent activations returned by the encoder
    """
    model.eval()

    x = _validate_image_activations(image_activations.to(device), source)
    _z_pre, z = model.encode(x, source=source, sigma=None)

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
    activation_extractor: Callable[[torch.Tensor, str], torch.Tensor],
    top_k: int = 5,
    device: str = "cuda",
) -> Tuple[List[Dict[str, float]], torch.Tensor]:
    """
    End-to-end inference for a raw image tensor.

    The SAE does not operate directly on pixels, so you must supply an
    activation_extractor that converts the image into source-model activations
    shaped (B, N, D).

    Args:
        model: UniversalSAE
        image: Raw image tensor shaped (C, H, W) or (B, C, H, W)
        source: Source vision model name
        activation_extractor: Callable returning activations shaped (B, N, D)
        top_k: Number of top features to return
        device: Inference device

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
    image_activations = activation_extractor(image, source)
    return top_features_from_activations(
        model=model,
        image_activations=image_activations,
        source=source,
        top_k=top_k,
        device=device,
    )


def print_top_features(features: List[Dict[str, float]]) -> None:
    # Print the ranked features.
    for rank, feature in enumerate(features, start=1):
        print(
            f"{rank}. feature {feature['feature_idx']} "
            f"(score={feature['score']:.6f})"
        )
