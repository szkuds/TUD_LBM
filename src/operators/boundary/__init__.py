"""Boundary condition operators — pure functions + composite builder."""

# Import modules to trigger registration with the global registry
from operators.boundary import bounce_back as _bb  # noqa: F401
from operators.boundary import symmetry as _sym  # noqa: F401
from operators.boundary import periodic as _per  # noqa: F401

from operators.boundary.bounce_back import apply_bounce_back
from operators.boundary.symmetry import apply_symmetry
from operators.boundary.periodic import apply_periodic
from operators.boundary.composite import build_composite_bc

__all__ = [
    "apply_bounce_back",
    "apply_symmetry",
    "apply_periodic",
    "build_composite_bc",
]
