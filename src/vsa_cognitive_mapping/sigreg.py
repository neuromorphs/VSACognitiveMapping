"""SIGReg — Sketched Isotropic Gaussian Regularization (the LeJEPA regularizer).

Pushes a batch of embeddings toward N(0, I) and is the ONLY regularizer in
this project (no EMA, no stop-gradient — the LeJEPA point). Via the
Cramér-Wold theorem, the batch is projected onto random unit directions and
each 1-D projection is tested for standard normality with the Epps-Pulley
statistic:

    T = n * ∫ w(t) |φ_N(t) − exp(−t²/2)|² dt,   w(t) = exp(−t²/2),

where φ_N is the empirical characteristic function of the n projected samples
and exp(−t²/2) is the characteristic function of N(0, 1). The integral runs
over t ∈ [0, t_max] with trapezoid weights doubled for the (even) t < 0 half.

Implemented from the paper's specification. Reference: Balestriero & LeCun,
"LeJEPA" (arXiv:2511.08544); official code (CC BY-NC 4.0):
https://github.com/rbalestr-lab/lejepa

Two lessons from a previous re-implementation (notes/topjepa-lessons.md §2-3):
- The `* n` scaling is essential — without it SIGReg is ~batch-size× too weak
  and cannot prevent collapse at any λ.
- The quadrature constants are buffers, NEVER nn.Parameters — a trainable
  quadrature lets the optimizer drive the statistic negative, turning the
  regularizer into a collapse *reward*.
"""

import torch
from torch import nn


class SIGReg(nn.Module):
    t: torch.Tensor
    target: torch.Tensor
    weights: torch.Tensor

    def __init__(self, num_projections: int = 256, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        self.num_projections = num_projections
        t = torch.linspace(0.0, t_max, knots)
        dt = t_max / (knots - 1)
        trapezoid = torch.full((knots,), 2.0 * dt)  # x2: integrand is even in t
        trapezoid[0] = trapezoid[-1] = dt
        target = torch.exp(-t.square() / 2.0)  # φ of N(0,1); also the window w(t)
        self.register_buffer("t", t)
        self.register_buffer("target", target)
        self.register_buffer("weights", trapezoid * target)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, D) pooled embeddings -> scalar statistic (mean over projections)."""
        directions = torch.randn(z.size(-1), self.num_projections, device=z.device)
        directions = directions / directions.norm(dim=0)
        proj = z @ directions                    # (N, M) 1-D samples per direction
        x_t = proj.unsqueeze(-1) * self.t        # (N, M, K)
        err = (x_t.cos().mean(0) - self.target).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights) * z.size(0)  # (M,) Epps-Pulley with ×n
        return statistic.mean()
