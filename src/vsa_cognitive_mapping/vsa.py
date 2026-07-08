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

Ported from the exploratory notebook (`notebooks/loading_sample_blender_data.ipynb`,
"Associative Memory" sections), which also has worked examples of building a
bundled memory trace and querying it back out by unbinding + a similarity
sweep.
"""

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
