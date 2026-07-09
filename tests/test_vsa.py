import numpy as np
import pytest
import torch

from vsa_cognitive_mapping.vsa import (
    Phasor,
    best_matches,
    circular_similarity,
    circular_wraparound_sweep,
    cosine_self_correlation,
    fidelity_score,
    fpe_bundle_encode,
    make_axis_bases,
    neg_abs_diff,
    orthogonality_score,
    pca_components,
    phasor_correlation_matrix,
    phasor_cross_correlation,
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
    assert fidelity_score(corr, corr) == pytest.approx(1.0, abs=1e-8)


def test_fidelity_score_negated_matrix_is_minus_one():
    # Pearson correlation of a vector with its own negation is exactly -1 --
    # a deterministic check that fidelity_score really is a correlation, not
    # e.g. a distance that would be insensitive to sign.
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    z, _ = random_project_to_phasor(x, d=8, seed=0)
    corr = phasor_correlation_matrix(z.numpy())
    assert fidelity_score(corr, -corr) == pytest.approx(-1.0, abs=1e-8)


def test_orthogonality_score_identity_matrix_is_one():
    # No off-diagonal similarity at all -> perfectly orthogonal.
    assert orthogonality_score(np.eye(5)) == pytest.approx(1.0, abs=1e-10)


def test_orthogonality_score_all_ones_matrix_is_zero():
    # Every pair identical -> zero orthogonality.
    assert orthogonality_score(np.ones((5, 5))) == pytest.approx(0.0, abs=1e-10)


def test_pca_components_unit_variance_and_variance_ratio_sums_to_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 12))
    scores, explained_variance_ratio = pca_components(x, n_components=4)
    assert scores.shape == (30, 4)
    assert np.allclose(scores.std(axis=0), 1.0, atol=1e-6)
    assert explained_variance_ratio.sum() == pytest.approx(1.0, abs=1e-6)
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


def test_circular_base_is_unit_modulus_and_seed_stable():
    a = Phasor(dim=64, seed=0, circular=True)
    b = Phasor(dim=64, seed=0, circular=True)
    assert np.allclose(np.abs(a.values), 1.0, atol=1e-10)
    assert np.array_equal(a.values, b.values)


def test_circular_base_wraps_exactly_after_one_turn():
    # The defining property: base**(angle + 2*pi) == base**angle, because
    # every dimension's phase is an integer number of radians.
    base = Phasor(dim=64, seed=0, circular=True)
    assert (base ** 0.7).similarity(base ** (0.7 + 2 * np.pi)) == pytest.approx(1.0, abs=1e-8)


def test_continuous_base_does_not_wrap_after_one_turn():
    # Contrast case: an ordinary continuous-random-phase base generically
    # does *not* return to the same phasor after a full 2*pi turn -- this is
    # exactly the defect circular=True fixes for a periodic scalar like yaw.
    base = Phasor(dim=64, seed=0)
    sim = (base ** 0.7).similarity(base ** (0.7 + 2 * np.pi))
    assert sim < 0.999


def test_circular_wraparound_sweep_matches_ground_truth_for_max_freq_one():
    # max_freq=1 -> every dimension is +-1, and cos(+-angle) == cos(angle),
    # so the encoded curve collapses to exactly the ground-truth cos(angle).
    base = Phasor(dim=128, seed=0, circular=True, max_freq=1)
    _, ground_truth, encoded = circular_wraparound_sweep(base, n_points=25)
    assert np.allclose(encoded, ground_truth, atol=1e-6)


def test_circular_wraparound_sweep_is_periodic_and_symmetric_for_mixed_frequencies():
    # With max_freq > 1, mixing harmonics per dimension no longer traces
    # exactly cos(angle) (each dim contributes cos(k*angle) for its own k),
    # but two properties must still hold for *any* integer frequency mix:
    # exact 2*pi periodicity, and symmetry (cos is even in angle).
    base = Phasor(dim=128, seed=0, circular=True, max_freq=3)
    angles, _, encoded = circular_wraparound_sweep(base, n_points=25)
    mid = len(angles) // 2
    assert angles[mid] == pytest.approx(0.0)
    assert encoded[mid] == pytest.approx(1.0, abs=1e-8)          # base**0 vs itself
    assert encoded[0] == pytest.approx(encoded[-1], abs=1e-8)    # -2*pi == +2*pi
    assert np.allclose(encoded, encoded[::-1], atol=1e-8)        # even in angle


def test_circular_wraparound_sweep_diverges_for_continuous_base_past_one_period():
    base = Phasor(dim=128, seed=0)
    angles, ground_truth, encoded = circular_wraparound_sweep(base, n_points=25)
    past_one_period = np.abs(angles) > np.pi
    assert not np.allclose(encoded[past_one_period], ground_truth[past_one_period], atol=1e-2)


def test_circular_similarity_matches_manual_cosine():
    angle = np.array([0.0, np.pi, np.pi / 2])
    corr = circular_similarity(angle)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-10)
    assert corr[0, 1] == pytest.approx(-1.0, abs=1e-10)  # 0 vs pi: opposite headings


def test_neg_abs_diff_diagonal_is_zero_and_symmetric():
    x = np.array([0.0, 5.0, 2.0])
    d = neg_abs_diff(x)
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T)
    assert d[0, 1] == pytest.approx(-5.0)


def test_phasor_cross_correlation_self_case_matches_phasor_correlation_matrix():
    torch.manual_seed(0)
    x = torch.randn(6, 32)
    z, _ = random_project_to_phasor(x, d=16, seed=0)
    z_np = z.numpy()
    assert np.allclose(phasor_cross_correlation(z_np, z_np), phasor_correlation_matrix(z_np), atol=1e-10)


def test_best_matches_recovers_bundled_item_after_unbind():
    # The full associative-memory recall round trip: bind each item's
    # content to its own context key, bundle every (content * context) trace
    # into one memory, then for each item unbind the memory by that item's
    # context -- best_matches should identify that item's own content as the
    # closest match in the codebook, not some other item's.
    dim = 512
    n_items = 5
    content = [Phasor(dim=dim, seed=10 + i) for i in range(n_items)]
    context = [Phasor(dim=dim, seed=20 + i) for i in range(n_items)]

    traces = [c.bind(k) for c, k in zip(content, context)]
    memory = traces[0].bundle(traces[1:])
    codebook = np.stack([c.values for c in content])

    for i in range(n_items):
        residual = memory.unbind(context[i])
        indices, similarities = best_matches(residual.values, codebook, k=2)
        # Correct item ranks first, with a clear margin over the runner-up --
        # not just barely ahead of bundling noise from the other 4 items.
        assert indices[0] == i
        assert similarities[0] - similarities[1] > 0.1


def test_best_matches_orders_by_descending_similarity():
    dim = 256
    codebook = np.stack([Phasor(dim=dim, seed=i).values for i in range(4)])
    query = Phasor(dim=dim, seed=2).values  # exact match to codebook row 2

    indices, similarities = best_matches(query, codebook, k=4)
    assert indices[0] == 2
    assert similarities[0] > 0.999
    assert np.all(np.diff(similarities) <= 1e-10)


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
