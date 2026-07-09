"""Frozen pretrained ViT backbone + the trained projection head.

The backbone (DINOv2, loaded via torch.hub) is the project's only pretrained
component and is never trained. The head (Linear + BatchNorm1d) produces the
shared latent z used everywhere — prediction MSE, SIGReg, and the predictor.
The per-dim BatchNorm is the primary collapse defense (see
notes/topjepa-lessons.md §1); do not replace it with LayerNorm or
L2-normalization.
"""

from pathlib import Path

import torch
from torch import nn

from .data import TransitionDataset, load_image


def load_backbone(name: str = "dinov2_vits14") -> nn.Module:
    """Load a frozen DINOv2 backbone; forward returns the CLS embedding."""
    backbone = torch.hub.load("facebookresearch/dinov2", name)
    assert isinstance(backbone, nn.Module)
    backbone.eval().requires_grad_(False)
    return backbone


class Head(nn.Module):
    """Backbone embedding -> shared latent z (the only trained encoder part)."""

    def __init__(self, backbone_dim: int = 384, latent_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.bn(self.proj(emb))


def cache_path(root: str | Path, backbone_name: str) -> Path:
    return Path(root) / f"embeddings_{backbone_name}.pt"


@torch.no_grad()
def build_embedding_cache(root: str | Path, backbone: nn.Module, backbone_name: str,
                          img_size: int = 224, batch_size: int = 32,
                          device: str = "cpu") -> dict[str, torch.Tensor]:
    """Run the frozen backbone once over every image referenced by the CSV.

    Saves {relative image path: (backbone_dim,) embedding} to
    root/embeddings_<backbone_name>.pt and returns it.
    """
    root = Path(root)
    full = TransitionDataset(root, split="train", val_fraction=0.0, img_size=img_size)
    paths = sorted(set(full.df["image_t"]) | set(full.df["image_tp1"]))

    backbone = backbone.to(device)
    cache: dict[str, torch.Tensor] = {}
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        imgs = torch.stack([load_image(root / p, img_size) for p in batch]).to(device)
        # .clone(): the CLS embedding is a view into the full 257-token tensor,
        # and torch.save would otherwise persist every view's whole storage
        # (~80x cache bloat on disk and in RAM when loaded back).
        embs = backbone(imgs).cpu().clone()
        cache.update(zip(batch, embs))

    torch.save(cache, cache_path(root, backbone_name))
    return cache


def load_or_build_cache(root: str | Path, backbone_name: str = "dinov2_vits14",
                        img_size: int = 224, device: str = "cpu") -> dict[str, torch.Tensor]:
    path = cache_path(root, backbone_name)
    if path.exists():
        return torch.load(path)
    print(f"embedding cache not found, building {path} (downloads {backbone_name} on first use)")
    return build_embedding_cache(root, load_backbone(backbone_name), backbone_name,
                                 img_size=img_size, device=device)
