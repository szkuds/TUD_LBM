"""Differential operators — pure functions for LBM stencil gradient and Laplacian."""

# Import modules to trigger registration with the global registry
from operators.differential import gradient as _grad_mod  # noqa: F401
from operators.differential import laplacian as _lap_mod  # noqa: F401

from operators.differential.gradient import compute_gradient, compute_wetting_gradient
from operators.differential.laplacian import compute_laplacian

__all__ = [
    "compute_gradient",
    "compute_wetting_gradient",
    "compute_laplacian",
]
