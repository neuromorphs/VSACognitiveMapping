"""Action-conditioned predictor: latent z_t + action a_t -> predicted z_{t+1}.

Single-step MLP (Markovian dynamics) — the minimal choice matching the
dataset's single-step transitions. The `predictor(z, a) -> ẑ` interface and
the `make_predictor` config switch keep an AdaLN transformer over latent
history swappable later (TASKS.md T05b) if partial observability shows up as
an error floor.
"""

import torch
from torch import nn


class MLPPredictor(nn.Module):
    def __init__(self, latent_dim: int = 128, action_dim: int = 4,
                 hidden_dim: int = 256, action_embed_dim: int = 32):
        super().__init__()
        self.action_embed = nn.Linear(action_dim, action_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self.action_embed(action)], dim=-1))


def make_predictor(cfg: dict, latent_dim: int) -> nn.Module:
    kind = cfg.get("kind", "mlp")
    if kind == "mlp":
        return MLPPredictor(latent_dim=latent_dim,
                            action_dim=cfg.get("action_dim", 4),
                            hidden_dim=cfg.get("hidden_dim", 256),
                            action_embed_dim=cfg.get("action_embed_dim", 32))
    raise ValueError(f"unknown predictor kind {kind!r} (an 'adaln' variant is TASKS.md T05b)")
