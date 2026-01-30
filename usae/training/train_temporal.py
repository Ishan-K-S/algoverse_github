"""
Training entry for Temporal Universal SAE (vision + diffusion).

This uses UniversalSparseAutoencoder and TemporalUniversalSAETrainer from usae.sae.temporal_usae.

Dataset yields per-image activations; we collate to:
    activations_dict[source] -> (B, T, N, D)

For vision sources (N,D), we repeat across T.
For diffusion sources (T,N,D), we keep as-is.

Sigma schedule used for timestep conditioning is taken from a designated diffusion source.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple

from usae.datasets.imagenet_acts import ImageNetCombinedActivationDataset
from usae.sae.temporal_usae import UniversalSparseAutoencoder, TemporalUniversalSAETrainer


def make_collate_fn(sources: List[str], sigma_source: str):
    def collate(batch: List[Tuple[Dict[str, torch.Tensor], int, Dict[str, Any]]]):
        # batch is list of (acts, label, metadata)
        # determine T from sigma_source in metadata, else from any 3D activation
        T = None
        for acts, _, meta in batch:
            if sigma_source in meta.get('sigmas', {}):
                T = int(meta['sigmas'][sigma_source].shape[0])
                break
            for s, x in acts.items():
                if x.dim() == 3:
                    T = int(x.shape[0])
                    break
            if T is not None:
                break
        if T is None:
            T = 1

        out_acts: Dict[str, torch.Tensor] = {}
        for s in sources:
            xs = []
            for acts, _, _ in batch:
                if s not in acts:
                    raise KeyError(f"Missing source {s} in batch sample")
                x = acts[s]
                if x.dim() == 2:
                    x = x.unsqueeze(0).repeat(T, 1, 1)  # (T,N,D)
                elif x.dim() == 3:
                    # if mismatch, pad/truncate
                    if x.shape[0] != T:
                        if x.shape[0] > T:
                            x = x[:T]
                        else:
                            pad = x[-1:].repeat(T - x.shape[0], 1, 1)
                            x = torch.cat([x, pad], dim=0)
                else:
                    raise ValueError(f"Unexpected activation dim {x.dim()} for {s}")
                xs.append(x)
            out_acts[s] = torch.stack(xs, dim=0)  # (B,T,N,D)

        # sigmas: (B,T) from sigma_source
        sigmas_list = []
        for _, _, meta in batch:
            if sigma_source not in meta.get('sigmas', {}):
                raise KeyError(f"sigma_source={sigma_source} not found in metadata['sigmas']")
            sig = meta['sigmas'][sigma_source]
            if sig.shape[0] != T:
                sig = sig[:T] if sig.shape[0] > T else torch.cat([sig, sig[-1:].repeat(T - sig.shape[0])], dim=0)
            sigmas_list.append(sig)
        sigmas = torch.stack(sigmas_list, dim=0)  # (B,T)

        labels = torch.tensor([y for _, y, _ in batch], dtype=torch.long)
        metadata = {"sigmas": sigmas}
        return out_acts, labels, metadata
    return collate


def train(
    imagenet_root: str,
    activation_root: str,
    sources: List[str],
    sigma_source: str,
    latent_dim: int = 16384,
    batch_size: int = 8,
    num_workers: int = 8,
    lr: float = 1e-4,
    nb_epochs: int = 1,
    device: str = "cuda",
    use_class_tokens: bool = False,
    standardize: bool = True,
    divide_norm: bool = False,
    top_k: int = 32,
):
    dataset = ImageNetCombinedActivationDataset(
        root=imagenet_root,
        activation_root=activation_root,
        sources=sources,
        split="train",
        target_class="ALL",
        use_class_tokens=use_class_tokens,
        standardize=standardize,
        divide_norm=divide_norm,
    )

    # model_dims: feature dimension for each source (last dim)
    model_dims = {}
    # peek first item
    acts0, _, _ = dataset[0]
    for s in sources:
        x = acts0[s]
        model_dims[s] = int(x.shape[-1])

    model = UniversalSparseAutoencoder(
        model_dims=model_dims,
        latent_dim=latent_dim,
        use_timestep_conditioning=True,
        top_k=top_k,
        use_soft_topk=True,
    )

    trainer = TemporalUniversalSAETrainer(
        model=model,
        learning_rate=lr,
        device=device,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=make_collate_fn(sources, sigma_source),
    )

    for epoch in range(nb_epochs):
        for step, (acts, _labels, meta) in enumerate(loader):
            metrics = trainer.train_step(acts, meta)
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}:", {k: round(v, 4) if isinstance(v, float) else v for k,v in metrics.items() if k in ['total','reconstruction','sparsity','temporal','l0_norm','dead_features','lr','source_model']})
    return trainer.model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_root", type=str, required=True)
    parser.add_argument("--activation_root", type=str, required=True)
    parser.add_argument("--sources", type=str, nargs='+', required=True)
    parser.add_argument("--sigma_source", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k", type=int, default=32)
    args = parser.parse_args()

    train(
        imagenet_root=args.imagenet_root,
        activation_root=args.activation_root,
        sources=args.sources,
        sigma_source=args.sigma_source,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        nb_epochs=args.epochs,
        device=args.device,
        top_k=args.top_k,
    )
