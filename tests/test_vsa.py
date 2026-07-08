import numpy as np
import torch

from vsa_cognitive_mapping.vsa import (
    Phasor,
    phasor_correlation_matrix,
    random_project_to_phasor,
)


def test_bind_unbind_are_inverses():
    p = Phasor(dim=256, seed=0)
    q = Phasor(dim=256, seed=1)
    recovered = p.bind(q).unbind(q)
    assert np.allclose(recovered.values, p.values, atol=1e-10)


def test_similarity_self_is_one():
    p = Phasor(dim=256, seed=0)
    assert abs(p.similarity(p) - 1.0) < 1e-10


def test_fpe_group_homomorphism():
    # phi(a) * phi(b) == phi(a + b), the defining property of FPE.
    Bx = Phasor(dim=512, seed=1)
    By = Phasor(dim=512, seed=2)

    def encode(v):
        return (Bx ** float(v[0])) * (By ** float(v[1]))

    a = np.array([3.2, -1.7])
    b = np.array([-0.5, 4.1])
    sim = encode(a).bind(encode(b)).similarity(encode(a + b))
    assert sim > 0.999


def test_bundle_is_a_mean_not_a_sum():
    p = Phasor(dim=64, seed=0)
    bundled = p.bundle(p)
    assert np.allclose(bundled.values, p.values, atol=1e-10)


def test_random_project_to_phasor_unit_modulus_and_shape():
    torch.manual_seed(0)
    x = torch.randn(10, 128)
    z, W = random_project_to_phasor(x, d=32, seed=0)
    assert z.shape == (10, 32)
    assert torch.allclose(z.abs(), torch.ones_like(z.abs()), atol=1e-5)
    assert W.shape == (128, 64)


def test_phasor_correlation_matrix_symmetric_unit_diagonal():
    torch.manual_seed(0)
    x = torch.randn(6, 32)
    z, _ = random_project_to_phasor(x, d=16, seed=0)
    corr = phasor_correlation_matrix(z.numpy())
    assert corr.shape == (6, 6)
    assert np.allclose(corr, corr.T, atol=1e-10)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-5)
