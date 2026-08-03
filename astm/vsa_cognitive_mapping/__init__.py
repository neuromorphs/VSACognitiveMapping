"""Standalone port of the VSACognitiveMapping phasor/VSA core.

Ported from ``neuromorphs/VSACognitiveMapping`` (jepa branch), the Telluride
Neuromorphic Workshop 2026 project. This package deliberately mirrors the
upstream layout (``vsa_cognitive_mapping.vsa``) so the two can be compared
directly. See ``README.md`` for provenance and the runnable demo.
"""

from .vsa import (
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

__all__ = [
    "Phasor",
    "cosine_self_correlation",
    "fidelity_score",
    "fpe_bundle_encode",
    "make_axis_bases",
    "orthogonality_score",
    "pca_components",
    "phasor_correlation_matrix",
    "random_project_to_phasor",
]
