"""Hyperdimensional (phasor / holographic reduced representation) primitives.

A `Phasor` is a vector of unit-modulus complex numbers, C^dim. Binding two
phasors (elementwise complex multiply) is the frequency-domain form of
circular convolution; unbinding (elementwise divide) is its approximate
inverse; bundling (elementwise mean) superposes several phasors into one
trace that still correlates with each input. Fractional power encoding
(FPE) raises a fixed random base phasor to a continuous scalar exponent,
`phi(v) = B**v`, which is a group homomorphism (`phi(a) * phi(b) == phi(a+b)`)
and gives a bell/sinc-shaped similarity kernel over the encoded scalar — the
standard VSA scheme for embedding continuous quantities like position and
time into a bind/bundle-able vector.

Provenance
----------
Ported verbatim (2026-07-14) from ``neuromorphs/VSACognitiveMapping``
(``jepa`` branch, ``src/vsa_cognitive_mapping/vsa.py``), the Telluride
Neuromorphic Workshop 2026 collaboration. Kept byte-compatible with that
repo's ``tests/test_vsa.py`` so the two implementations can be diffed.

The only local change is the ``from __future__ import annotations`` line
below: this repo runs on Python 3.9, where the ``int | None`` style hints in
the original would raise ``TypeError`` at definition time. The future import
makes all annotations lazy (strings), so the same signatures parse cleanly
on 3.9 while remaining identical to upstream on 3.10+.

Upstream provenance note: originally ported from that repo's exploratory
notebook (``notebooks/loading_sample_blender_data.ipynb``, "Associative
Memory" sections), which has worked examples of building a bundled memory
trace and querying it back out by unbinding + a similarity sweep.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class Phasor:
    """Unit-modulus complex vector in C^dim, with VSA bind/bundle/FPE ops."""

    def __init__(self, dim: int | None = None, seed: int | None = None,
                 data: np.ndarray | None = None):
        if data is not None:
            self.values = data
            self.dim = data.shape[0]
        else:
            phases = np.random.RandomState(seed).uniform(0, 2 * np.pi, dim)
            self.values = np.exp(1j * phases)
            self.dim = dim

    def bind(self, other: "Phasor") -> "Phasor":
        return Phasor(data=np.multiply(self.values, other.values))

    def unbind(self, other: "Phasor") -> "Phasor":
        return Phasor(data=np.divide(self.values, other.values))

    def fpe(self, exponent: float) -> "Phasor":
        return Phasor(data=self.values ** exponent)

    def bundle(self, others: "Phasor | list[Phasor]") -> "Phasor":
        if not isinstance(others, list):
            others = [others]
        raw = np.add(self.values, sum(o.values for o in others))
        return Phasor(data=raw / (len(others) + 1))

    def similarity(self, other: "Phasor") -> float:
        return float(np.real(np.sum(np.conj(self.values) * other.values)) / self.dim)

    __mul__ = bind
    __rmul__ = bind
    __truediv__ = unbind
    __pow__ = fpe
    __add__ = bundle
    __mod__ = similarity


def random_project_to_phasor(
    x: torch.Tensor,
    d: int,
    W: torch.Tensor | None = None,
    seed: int | None = None,
    eps: float = 1e-8,
    return_iq: bool = False,
):
    """Randomly project real embeddings into I/Q phasor space.

    Args:
        x: Input embeddings with shape (batch, input_dim), e.g. (100, 256)
        d: Number of phasor dimensions
        W: Optional fixed random projection matrix with shape (input_dim, 2*d)
        seed: Optional seed used only if W is not provided
        eps: Small value to avoid divide-by-zero
        return_iq: If True, also return normalized real I/Q representation

    Returns:
        z: Complex phasor tensor with shape (batch, d)
        iq_normed: Optional real tensor with shape (batch, 2*d)
        W: Random projection matrix, useful for reusing the same encoder
    """
    batch_size, input_dim = x.shape

    if W is None:
        if seed is not None:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(seed)
        else:
            generator = None

        W = torch.randn(
            input_dim,
            2 * d,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        ) / math.sqrt(input_dim)

    # Project to concatenated I/Q values
    iq = x @ W  # shape: (batch, 2*d)

    # Split into in-phase and quadrature components
    i, q = iq.chunk(2, dim=-1)  # each shape: (batch, d)

    # Normalize each (i, q) pair onto the unit circle
    mag = torch.sqrt(i**2 + q**2 + eps)

    i_norm = i / mag
    q_norm = q / mag

    # Complex phasor representation
    z = torch.complex(i_norm, q_norm)  # shape: (batch, d)

    if return_iq:
        iq_normed = torch.cat([i_norm, q_norm], dim=-1)  # shape: (batch, 2*d)
        return z, iq_normed, W

    return z, W


def phasor_correlation_matrix(phasors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Pairwise normalized correlation/similarity between complex phasors.

    Args:
        phasors: complex array with shape (batch, d)

    Returns:
        corr: real-valued correlation matrix with shape (batch, batch)
    """
    phasors_unit = phasors / (np.linalg.norm(phasors, axis=1, keepdims=True) + eps)
    return np.real(phasors_unit @ phasors_unit.conj().T)


def cosine_self_correlation(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(N, D) real vectors -> (N, N) pairwise cosine similarity against themselves."""
    x_unit = x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    return x_unit @ x_unit.T


def _upper_triangle(mat: np.ndarray) -> np.ndarray:
    return mat[np.triu_indices_from(mat, k=1)]


def fidelity_score(reference_corr: np.ndarray, encoded_corr: np.ndarray) -> float:
    """Pearson correlation between two pairwise similarity matrices' upper
    triangles — how well `encoded_corr` preserves the neighbor structure
    (which items look alike) of `reference_corr`."""
    return float(np.corrcoef(_upper_triangle(reference_corr), _upper_triangle(encoded_corr))[0, 1])


def orthogonality_score(corr: np.ndarray) -> float:
    """1 - mean |off-diagonal similarity| — how close a set of encoded
    vectors is to mutually orthogonal (quasi-orthogonality): 1.0 means every
    pair is orthogonal (no interference when bundled together), 0.0 means
    every pair is identical (total collapse)."""
    return float(1.0 - np.abs(_upper_triangle(corr)).mean())


def pca_components(x: np.ndarray, n_components: int, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Top-`n_components` PCA scores of `x` (N, d_in), each standardized to
    unit variance, via numpy SVD — no sklearn dependency, same approach as
    the exploratory notebook's "Associative Memory 2" section.

    Returns:
        scores: (N, n_components) real, each column unit-variance
        explained_variance_ratio: (min(N, d_in),) fraction of total variance
            per component, over the full SVD rank (not just n_components) so
            a caller can plot a scree/cumulative-variance curve independently
            of how many components it chose to keep.
    """
    centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    explained_variance = s ** 2 / max(x.shape[0] - 1, 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    scores = centered @ vt[:n_components].T
    scores = scores / (scores.std(axis=0, keepdims=True) + eps)
    return scores, explained_variance_ratio


def make_axis_bases(n_axes: int, d: int, seed: int) -> list[Phasor]:
    """One independent random base phasor (dim d) per axis, seeded off
    `seed + axis index` so a smaller axis count is always a prefix of a
    larger one — sweeping over how many axes to use doesn't reshuffle the
    bases already in play."""
    return [Phasor(dim=d, seed=seed + i) for i in range(n_axes)]


def fpe_bundle_encode(scores: np.ndarray, bases: list[Phasor], length_scale: float) -> np.ndarray:
    """FPE-encode each column of `scores` (N, k) against its matching base
    phasor from `bases`, scaling the exponent by 1/length_scale (smaller
    length_scale -> faster rotation per unit of score -> narrower similarity
    kernel), then bundle (elementwise mean) the k per-axis phasors into one
    (N, d) complex content array per row.

    Returns:
        content: (N, d) complex ndarray
    """
    content = []
    for i in range(scores.shape[0]):
        axis_phasors = [base ** float(scores[i, c] / length_scale) for c, base in enumerate(bases)]
        content.append(axis_phasors[0].bundle(axis_phasors[1:]).values)
    return np.stack(content)
