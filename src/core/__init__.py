"""Simulation runner for LBM simulations.

Provides a high-level interface for configuring and executing
lattice Boltzmann simulations with automatic setup and I/O.

Classes:
    Run: Main entry point for running LBM simulations.
"""

from .run import Run
__all__ = ["Run"]
