"""The JEPA world model: head + predictor + SIGReg, LeWorldModel loss.

    L = MSE(ẑ_{t+1}, z_{t+1}) + λ · SIGReg(z pooled over both timesteps)

Plain MSE in the BatchNorm'd latent space, no L2 normalization, no
stop-gradient on the target (intentional — the LeJEPA recipe; gradient flows
to the head through both prediction and target).
"""

import torch
import torch.nn.functional as F
from torch import nn

from .diagnostics import collapse_metrics
from .encoder import Head
from .predictor import make_predictor
from .sigreg import SIGReg


class JEPAWorldModel(nn.Module):
    def __init__(self, head: Head, predictor: nn.Module, sigreg: SIGReg, lam: float = 0.1):
        super().__init__()
        self.head = head
        self.predictor = predictor
        self.sigreg = sigreg
        self.lam = lam

    @classmethod
    def from_config(cls, cfg: dict) -> "JEPAWorldModel":
        """cfg is the `model:` section of a config file."""
        latent_dim = cfg.get("latent_dim", 128)
        return cls(head=Head(cfg.get("backbone_dim", 384), latent_dim),
                   predictor=make_predictor(cfg.get("predictor", {}), latent_dim),
                   sigreg=SIGReg(**cfg.get("sigreg", {})),
                   lam=cfg.get("lam", 0.1))

    def encode(self, emb: torch.Tensor) -> torch.Tensor:
        return self.head(emb)

    def loss(self, emb_t: torch.Tensor, emb_tp1: torch.Tensor,
             action: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Backbone embeddings (B, backbone_dim) x2 + one-hot actions (B, 4)."""
        # One concatenated batch through the head so z_t and z_tp1 share the
        # same BatchNorm batch statistics.
        z = self.head(torch.cat([emb_t, emb_tp1], dim=0))
        z_t, z_tp1 = z.chunk(2, dim=0)

        z_pred = self.predictor(z_t, action)
        mse = F.mse_loss(z_pred, z_tp1)
        reg = self.sigreg(z)
        loss = mse + self.lam * reg

        metrics = {"loss": loss.item(), "mse": mse.item(), "sigreg": reg.item(),
                   **collapse_metrics(z.detach())}
        return loss, metrics
