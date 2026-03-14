"""Macroscopic field operators — pure functions."""

from operators.macroscopic.single_phase import compute_macroscopic
from operators.macroscopic.multiphase import compute_macroscopic_multiphase

__all__ = [
    "compute_macroscopic",
    "compute_macroscopic_multiphase",
]
