import torch

from vsa_cognitive_mapping.sigreg import SIGReg


def test_no_trainable_parameters():
    # A trainable quadrature lets the optimizer drive the statistic negative
    # and reward collapse (topjepa BUG04) — SIGReg must expose zero parameters.
    assert len(list(SIGReg().parameters())) == 0


def test_gaussian_scores_lowest():
    # Note: a high-dim isotropic *uniform* projected onto random directions is
    # ~Gaussian by CLT, so that is not a discriminable case (nor one we care
    # about). What SIGReg must flag: wrong scale, collapse, and low-dim
    # non-normality. N(0,I) at batch 512 scores ~1.2 (the reference value).
    torch.manual_seed(0)
    sigreg = SIGReg()

    def avg(z, k=8):
        return torch.stack([sigreg(z) for _ in range(k)]).mean()

    gaussian = torch.randn(512, 64)
    collapsed = torch.randn(1, 64).expand(512, 64) + 0.01 * torch.randn(512, 64)
    assert avg(gaussian) < 3.0
    assert avg(gaussian) < avg(3 * gaussian)   # wrong scale
    assert avg(gaussian) < avg(collapsed)      # directional collapse
    gaussian_2d = torch.randn(512, 2)
    uniform_2d = (torch.rand(512, 2) - 0.5) * (12.0 ** 0.5)  # zero mean, unit var
    assert avg(gaussian_2d) < avg(uniform_2d)  # low-dim non-normality


def test_gradient_reaches_input():
    torch.manual_seed(0)
    z = torch.randn(64, 16, requires_grad=True)
    SIGReg()(z).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_stays_nonnegative_under_optimization():
    # The statistic is a weighted sum of squares — optimizing against it must
    # never drive it below zero (it would if the quadrature were trainable).
    torch.manual_seed(0)
    z = torch.nn.Parameter(torch.rand(64, 16) * 5)
    sigreg = SIGReg()
    opt = torch.optim.Adam([z] + list(sigreg.parameters()), lr=0.1)
    for _ in range(50):
        loss = sigreg(z)
        assert loss.item() >= 0.0
        opt.zero_grad()
        loss.backward()
        opt.step()
