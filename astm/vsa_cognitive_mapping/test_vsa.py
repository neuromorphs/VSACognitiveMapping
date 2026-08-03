"""Correctness tests for the ported phasor/VSA core.

Adapted from ``neuromorphs/VSACognitiveMapping`` (jepa branch,
``tests/test_vsa.py``). The assertions are identical to upstream; the only
changes are (1) dropping the hard ``pytest`` dependency — this repo's env has
no pytest installed — via a tiny ``approx`` shim and a plain ``__main__``
runner, and (2) a ``sys.path`` insert so the file runs directly with
``python vsa_cognitive_mapping/test_vsa.py`` (project convention).

It is still discoverable by pytest if you install it (``pip install pytest``
then ``pytest vsa_cognitive_mapping/test_vsa.py``): every ``test_*`` function
is a standard no-fixture test.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.vsa import (  # noqa: E402
    Phasor,
    cosine_self_correlation,
    fidelity_score,
    fpe_bundle_encode,
    make_axis_bases,
    orthogonality_score,
    pca_components,
    phasor_correlation_matrix,
    random_project_to_phasor,
)


class _Approx:
    """Minimal stand-in for ``pytest.approx`` covering the scalar `==` use
    in these tests (``value == approx(target, abs=tol)``)."""

    def __init__(self, expected, abs=1e-8):
        self.expected = expected
        self.abs = abs

    def __eq__(self, other):
        return abs(float(other) - float(self.expected)) <= self.abs


def approx(expected, abs=1e-8):
    return _Approx(expected, abs=abs)


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


def test_cosine_self_correlation_diagonal_is_one_and_symmetric():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 10))
    corr = cosine_self_correlation(x)
    assert corr.shape == (6, 6)
    assert np.allclose(corr, corr.T, atol=1e-10)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-6)


def test_fidelity_score_identical_matrices_is_one():
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    z, _ = random_project_to_phasor(x, d=8, seed=0)
    corr = phasor_correlation_matrix(z.numpy())
    assert fidelity_score(corr, corr) == approx(1.0, abs=1e-8)


def test_fidelity_score_negated_matrix_is_minus_one():
    # Pearson correlation of a vector with its own negation is exactly -1 --
    # a deterministic check that fidelity_score really is a correlation, not
    # e.g. a distance that would be insensitive to sign.
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    z, _ = random_project_to_phasor(x, d=8, seed=0)
    corr = phasor_correlation_matrix(z.numpy())
    assert fidelity_score(corr, -corr) == approx(-1.0, abs=1e-8)


def test_orthogonality_score_identity_matrix_is_one():
    # No off-diagonal similarity at all -> perfectly orthogonal.
    assert orthogonality_score(np.eye(5)) == approx(1.0, abs=1e-10)


def test_orthogonality_score_all_ones_matrix_is_zero():
    # Every pair identical -> zero orthogonality.
    assert orthogonality_score(np.ones((5, 5))) == approx(0.0, abs=1e-10)


def test_pca_components_unit_variance_and_variance_ratio_sums_to_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 12))
    scores, explained_variance_ratio = pca_components(x, n_components=4)
    assert scores.shape == (30, 4)
    assert np.allclose(scores.std(axis=0), 1.0, atol=1e-6)
    assert explained_variance_ratio.sum() == approx(1.0, abs=1e-6)
    assert np.all(np.diff(explained_variance_ratio) <= 1e-10)  # sorted descending


def test_pca_components_recovers_dominant_axis():
    rng = np.random.default_rng(0)
    dominant = rng.normal(scale=100.0, size=(200, 1)) @ rng.normal(size=(1, 5))
    noise = rng.normal(scale=1e-3, size=(200, 5))
    scores, explained_variance_ratio = pca_components(dominant + noise, n_components=1)
    assert explained_variance_ratio[0] > 0.999
    assert scores.shape == (200, 1)


def test_make_axis_bases_is_seed_prefix_stable():
    # A sweep over "how many axes to use" should never reshuffle bases
    # already in play -- the first k bases must be identical regardless of
    # how many total axes were requested.
    few = make_axis_bases(2, d=32, seed=0)
    many = make_axis_bases(5, d=32, seed=0)
    for a, b in zip(few, many):
        assert np.array_equal(a.values, b.values)


def test_fpe_bundle_encode_single_axis_matches_plain_fpe():
    scores = np.array([[0.5], [-1.2], [3.0]])
    bases = make_axis_bases(1, d=16, seed=0)
    content = fpe_bundle_encode(scores, bases, length_scale=1.0)
    expected = np.stack([(bases[0] ** float(v)).values for v in scores[:, 0]])
    assert np.allclose(content, expected, atol=1e-10)


def test_fpe_bundle_encode_two_axes_matches_manual_bundle():
    scores = np.array([[0.3, -0.7], [1.1, 2.4]])
    bases = make_axis_bases(2, d=16, seed=0)
    length_scale = 2.0
    content = fpe_bundle_encode(scores, bases, length_scale)
    expected = np.stack([
        ((bases[0] ** float(scores[i, 0] / length_scale)).bundle(
            bases[1] ** float(scores[i, 1] / length_scale))).values
        for i in range(scores.shape[0])
    ])
    assert np.allclose(content, expected, atol=1e-10)


def test_fpe_bundle_encode_shorter_length_scale_gives_narrower_kernel():
    # Two frames with different score values should look *more* similar
    # under a longer length scale (slower rotation per unit of score) --
    # this is the whole point of the length-scale sweep in
    # scripts/encoder_sweep.py, so pin the direction down explicitly.
    scores = np.array([[0.0], [2.0]])
    bases = make_axis_bases(1, d=64, seed=0)
    narrow = fpe_bundle_encode(scores, bases, length_scale=0.25)
    wide = fpe_bundle_encode(scores, bases, length_scale=4.0)
    sim_narrow = phasor_correlation_matrix(narrow)[0, 1]
    sim_wide = phasor_correlation_matrix(wide)[0, 1]
    assert sim_wide > sim_narrow


def test_pca_fpe_pipeline_more_components_improves_fidelity():
    # Integration check that pca_components + fpe_bundle_encode +
    # phasor_correlation_matrix + fidelity_score compose the way
    # scripts/encoder_sweep.py assumes: on generic (no dominant-axis) data,
    # keeping more components should preserve more of the raw similarity
    # structure than keeping just one.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 30))
    raw_corr = cosine_self_correlation(x)

    def fidelity_with_k(k):
        scores, _ = pca_components(x, k)
        bases = make_axis_bases(k, d=64, seed=0)
        content = fpe_bundle_encode(scores, bases, length_scale=1.0)
        return fidelity_score(raw_corr, phasor_correlation_matrix(content))

    assert fidelity_with_k(10) > fidelity_with_k(1)


def _main() -> int:
    """Run every ``test_*`` function in this module without pytest."""
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((name, exc))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
