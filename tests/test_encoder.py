import pytest
import torch
from torch import nn

from vsa_cognitive_mapping.encoder import (Head, build_embedding_cache, cache_path,
                                           load_backbone)


class FakeBackbone(nn.Module):
    """Stands in for DINOv2: (B, 3, H, W) -> (B, 384), no network needed."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 384)

    def forward(self, x):
        return self.proj(x.mean(dim=(2, 3)))


def test_head_shape_and_batchnorm_modes():
    torch.manual_seed(0)
    head = Head(backbone_dim=384, latent_dim=32)
    emb = torch.randn(16, 384) * 3 + 1
    head.train()
    z = head(emb)
    assert z.shape == (16, 32)
    # train mode normalizes by batch statistics
    assert torch.allclose(z.mean(dim=0), torch.zeros(32), atol=1e-5)
    assert torch.allclose(z.std(dim=0, unbiased=False), torch.ones(32), atol=1e-4)
    # eval mode uses running stats -> generally different output
    head.eval()
    assert not torch.allclose(head(emb), z)
    # and works on a single observation (batch stats would need N > 1)
    assert head(emb[:1]).shape == (1, 32)


def test_cache_roundtrip(tiny_dataset):
    backbone = FakeBackbone()
    cache = build_embedding_cache(tiny_dataset, backbone, "fake", img_size=16)
    reloaded = torch.load(cache_path(tiny_dataset, "fake"))
    assert set(cache) == set(reloaded)
    from vsa_cognitive_mapping.data import load_image
    path = next(iter(cache))
    direct = backbone(load_image(tiny_dataset / path, 16).unsqueeze(0)).squeeze(0)
    assert torch.allclose(reloaded[path], direct, atol=1e-6)


@pytest.mark.slow
def test_real_backbone_is_frozen():
    backbone = load_backbone("dinov2_vits14")  # downloads on first use
    assert all(not p.requires_grad for p in backbone.parameters())
    assert not backbone.training
    out = backbone(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 384)
