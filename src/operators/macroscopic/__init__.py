"""Macroscopic field operators — pure functions."""

from operators.macroscopic.multiphase import compute_macroscopic_multiphase
from operators.macroscopic.single_phase import compute_macroscopic

__all__ = [
    "compute_macroscopic",
    "compute_macroscopic_multiphase",
]
