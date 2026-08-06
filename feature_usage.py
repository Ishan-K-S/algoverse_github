"""
One place for "does this model use dictionary feature k" and "do two models
agree on which features fire where".

Before this module existed, train.py, dictionary_diagnostic.py, and
cross_model_overlap.py each answered "is feature k used" with a different
aggregation, so a "used" count from one script was never comparable to
another's:
  - train.py:                 usage-EMA rate > threshold (continuous, whole run)
  - dictionary_diagnostic.py:  fired on >=1 token of >=1 image in a fixed sample
  - cross_model_overlap.py:    in the per-image top-K by max|activation|, for >=1 image
None of these is "more correct" -- they answer different questions (a
decaying rate vs. an ever/never indicator vs. a top-K membership indicator).
Making the criterion an explicit argument, with one shared implementation
per criterion, means two scripts using the same criterion are now
guaranteed to agree, instead of drifting because each reimplemented the same
english phrase slightly differently.

Also: the project's stated goal is "the same dictionary features fire on the
same CONTENT across models," but every existing metric (partition/usage_cosine,
dictionary_diagnostic's per-image Jaccard, cross_model_overlap's top-K
Jaccard) aggregates away the token/spatial axis before comparing models --
exactly the axis the goal lives on. per_token_cofire_jaccard and
feature_heatmap_iou below measure per-token/per-position agreement instead
of per-image aggregate co-usage.
"""

from __future__ import annotations

from typing import Optional

import torch


def compute_feature_usage(
    activations: torch.Tensor,
    criterion: str,
    threshold: float = 1e-3,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Returns a (K,) bool tensor: True where dictionary feature k counts as
    "used" by this model, under `criterion`.

    criterion="rate_above_threshold": `activations` is already a (K,) usage
        rate (e.g. a usage EMA) in [0, 1]; used = rate > threshold. Matches
        train.py's resample_dead_features / partition-diagnostic definition.
    criterion="ever_fired": `activations` is (..., K) raw latent codes over
        any number of leading sample/token dims; used = nonzero at least
        once anywhere in those dims. Matches dictionary_diagnostic.py.
    criterion="top_k_per_sample": `activations` is (num_samples, K) (one row
        per image, e.g. amax over that image's tokens); used = in the top
        `top_k` by |value| for at least one row. Matches
        cross_model_overlap.py / dictionary_diagnostic.py's Jaccard scoring.
    """
    if criterion == "rate_above_threshold":
        if activations.dim() != 1:
            raise ValueError(
                f"rate_above_threshold expects a (K,) rate tensor, got {tuple(activations.shape)}"
            )
        return activations > threshold

    if criterion == "ever_fired":
        if activations.dim() < 1:
            raise ValueError("ever_fired expects at least 1 dimension (..., K)")
        flat = activations.reshape(-1, activations.shape[-1])
        return (flat != 0).any(dim=0)

    if criterion == "top_k_per_sample":
        if top_k is None:
            raise ValueError("top_k_per_sample requires top_k")
        if activations.dim() != 2:
            raise ValueError(
                f"top_k_per_sample expects (num_samples, K), got {tuple(activations.shape)}"
            )
        k = min(top_k, activations.shape[-1])
        idx = activations.abs().topk(k, dim=-1).indices  # (num_samples, k)
        used = torch.zeros(activations.shape[-1], dtype=torch.bool, device=activations.device)
        used[idx.reshape(-1).unique()] = True
        return used

    raise ValueError(f"Unknown criterion {criterion!r}")


def per_token_cofire_jaccard(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """
    Per-token co-fire: compares WHICH feature
    indices are nonzero AT EACH (image, token) POSITION independently,
    instead of aggregating over tokens first the way every prior metric did.

    z_a, z_b: (B, N, K) latent codes for two models, ALREADY spatially
    aligned to the same N (e.g. via SpatialAligner) so position n means the
    same image location in both -- meaningless otherwise.
    This function does not and cannot verify that its inputs were aligned;
    that is the caller's responsibility.

    Returns a 0-d tensor: the mean Jaccard(active features at position) over
    all B*N token positions. 1.0 = the two models fire the exact same
    feature set at every position; 0.0 = no overlap anywhere.
    """
    if z_a.shape != z_b.shape:
        raise ValueError(
            f"per_token_cofire_jaccard requires matching shapes, got "
            f"{tuple(z_a.shape)} vs {tuple(z_b.shape)}"
        )
    fired_a = z_a != 0
    fired_b = z_b != 0
    intersection = (fired_a & fired_b).sum(dim=-1).float()
    union = (fired_a | fired_b).sum(dim=-1).float()
    jaccard = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    return jaccard.mean()


def per_token_cofire_jaccard_chance(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """
    The value per_token_cofire_jaccard would take if the two models picked their
    active features INDEPENDENTLY at each position, given the number each
    actually activates there.

    Without this, the co-fire number is uninterpretable. Hard TopK puts exactly
    top_k nonzeros in every token for both models, so with top_k=128 and
    latent_dim=12288 the chance level is 128*128/12288 / (256 - 128*128/12288)
    ~= 0.0052 -- a raw reading of 0.02 is 4x chance, not "2% agreement, basically
    nothing." Computed from the OBSERVED per-token active counts rather than
    assumed to be top_k, so it stays correct if a criterion other than hard TopK
    is ever used.

    z_a, z_b: (B, N, K), same alignment requirement as per_token_cofire_jaccard.
    Returns a 0-d tensor. This is a ratio of expectations (E|A n B| / E|A u B|),
    not the expectation of the ratio -- close enough to read a chart against, not
    an exact null distribution.
    """
    if z_a.shape != z_b.shape:
        raise ValueError(
            f"per_token_cofire_jaccard_chance requires matching shapes, got "
            f"{tuple(z_a.shape)} vs {tuple(z_b.shape)}"
        )
    n_features = z_a.shape[-1]
    n_a = (z_a != 0).sum(dim=-1).float()
    n_b = (z_b != 0).sum(dim=-1).float()
    exp_intersection = n_a * n_b / n_features
    exp_union = n_a + n_b - exp_intersection
    chance = torch.where(
        exp_union > 0, exp_intersection / exp_union, torch.zeros_like(exp_union)
    )
    return chance.mean()


def feature_heatmap_iou(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    feature_idx: int,
    threshold: float = 0.0,
) -> float:
    """
    IoU between two models' per-token activation
    heatmaps for ONE shared dictionary feature, across a batch of spatially
    aligned images: "does feature k light up the same image REGIONS in both
    models," not just "do both models ever use feature k somewhere."

    z_a, z_b: (B, N, K) aligned latent codes (see per_token_cofire_jaccard --
    same alignment requirement and caveat apply here).
    feature_idx: which of the K dictionary indices to compare.
    threshold: a token counts as "active" for this feature if its magnitude
        exceeds this value.

    Returns a single float IoU in [0, 1]; 0.0 if the union is empty (feature
    never fires for either model anywhere in this batch).
    """
    if z_a.shape != z_b.shape:
        raise ValueError(
            f"feature_heatmap_iou requires matching shapes, got "
            f"{tuple(z_a.shape)} vs {tuple(z_b.shape)}"
        )
    active_a = z_a[..., feature_idx].abs() > threshold  # (B, N)
    active_b = z_b[..., feature_idx].abs() > threshold
    intersection = (active_a & active_b).sum().item()
    union = (active_a | active_b).sum().item()
    return intersection / union if union > 0 else 0.0


def labelmap_to_token_labels(label_map, grid_size: int):
    """
    Downsample a dense (H, W) integer label map onto the SAE's token grid,
    giving one concept id per token. Nearest-neighbour, so ids are never
    blended into ids that don't exist.
    """
    import numpy as np
    from PIL import Image
    small = np.asarray(
        Image.fromarray(label_map).resize((grid_size, grid_size), resample=Image.NEAREST)
    )
    return small.reshape(-1)


def concept_selectivity_margin(
    z: torch.Tensor,
    token_labels,
    min_percentile: float = 50.0,
    id_to_name: Optional[dict] = None,
):
    """
    Score each feature by TOP-CONCEPT MEAN minus SECOND-CONCEPT MEAN.

    This is the "is this feature high for ONE concept" question. It is
    deliberately not mean-or-max over the whole image: a feature that fires
    strongly everywhere scores a large max and a large mean but a margin near
    zero, and is correctly ranked last.

    z            : (1, N, K) or (N, K) latent code for ONE image.
    token_labels : (N,) concept id per token, from labelmap_to_token_labels.
    min_percentile : features whose top-concept mean falls below this
        percentile of the image's pooled nonzero activations are gated out
        (margin = -inf), so a large margin between two tiny numbers can't win.
    id_to_name   : optional {concept_id: name} for the returned report.

    Returns (margins (K,) float array, per-feature info list). Gated features
    keep an entry with margin -inf and a gate_reason, so the record of WHY a
    feature dropped out survives.
    """
    import numpy as np

    if z.dim() == 3:
        z = z[0]
    vals = z.abs().detach().cpu().numpy()          # (N, K)
    labels = np.asarray(token_labels).reshape(-1)
    if labels.shape[0] != vals.shape[0]:
        raise ValueError(
            f"token_labels has {labels.shape[0]} entries but z has {vals.shape[0]} tokens"
        )
    K = vals.shape[1]
    id_to_name = id_to_name or {}

    unique_ids = np.unique(labels)
    if len(unique_ids) < 2:
        return np.full(K, -np.inf), [
            {"gated": True, "gate_reason": "fewer than 2 concepts present in this image"}
            for _ in range(K)
        ]

    concept_means = np.stack([vals[labels == cid].mean(axis=0) for cid in unique_ids])
    order = np.argsort(-concept_means, axis=0)
    feat = np.arange(K)
    top1_row, top2_row = order[0], order[1]
    top1, top2 = concept_means[top1_row, feat], concept_means[top2_row, feat]
    raw = top1 - top2

    # Threshold pooled across all features, not per-feature: a per-feature
    # percentile is a no-op, since a feature's top-concept mean is always one
    # of its own higher values.
    pool = vals[vals > 0]
    thresh = float(np.percentile(pool, min_percentile)) if pool.size >= 2 else float("inf")

    has_signal = (vals > 0).any(axis=0)
    eligible = has_signal & (top1 >= thresh)
    margins = np.where(eligible, raw, -np.inf)

    info = []
    for k in range(K):
        e = {
            "top_concept": id_to_name.get(int(unique_ids[top1_row[k]]),
                                          f"class_{unique_ids[top1_row[k]]}"),
            "top_concept_mean": float(top1[k]),
            "second_concept": id_to_name.get(int(unique_ids[top2_row[k]]),
                                             f"class_{unique_ids[top2_row[k]]}"),
            "second_concept_mean": float(top2[k]),
            "margin": float(raw[k]),
            "magnitude_threshold": thresh,
            "gated": not bool(eligible[k]),
        }
        if not has_signal[k]:
            e["gate_reason"] = "feature never fires on this image"
        elif not eligible[k]:
            e["gate_reason"] = (f"top-concept mean {top1[k]:.4g} below image-wide "
                                f"p{min_percentile:.0f} threshold {thresh:.4g}")
        info.append(e)
    return margins, info
