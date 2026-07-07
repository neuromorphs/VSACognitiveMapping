"""Collapse diagnostics, computed live during training on the batch latents.

The three known collapse modes and their signatures (notes/topjepa-lessons.md §6):
- magnitude collapse:    per_dim_std -> 0
- dimensional collapse:  effective_rank -> 1
- directional collapse:  mean_pairwise_cosine -> 1
Healthy BatchNorm'd latents: per_dim_std ≈ 1, cosine ≈ 0, rank well above 1.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def collapse_metrics(z: torch.Tensor) -> dict[str, float]:
    """z: (N, D) latents -> the three collapse statistics."""
    per_dim_std = z.std(dim=0).mean()

    cov = torch.cov(z.T)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    p = eigvals / eigvals.sum().clamp(min=1e-12)
    effective_rank = torch.exp(-(p * (p + 1e-12).log()).sum())

    z_unit = F.normalize(z, dim=-1)
    cos = z_unit @ z_unit.T
    n = z.size(0)
    mean_pairwise_cosine = (cos.sum() - n) / (n * (n - 1))

    return {"per_dim_std": per_dim_std.item(),
            "effective_rank": effective_rank.item(),
            "mean_pairwise_cosine": mean_pairwise_cosine.item()}
