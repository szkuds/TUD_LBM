"""Simulation classes for LBM.

Provides simulation containers that orchestrate the complete LBM workflow
including initialisation, time-stepping, and field management.

Classes:
    BaseSimulation: Abstract base class defining the simulation interface.
    SinglePhaseSimulation: Single-phase flow simulation.
    MultiphaseSimulation: Multiphase (two-phase) flow simulation.

Configuration:
    from app_setup import SimulationSetup
"""

from .base import BaseSimulation
from .multiphase import MultiphaseSimulation
from .single_phase import SinglePhaseSimulation

__all__ = [
    "BaseSimulation",
    "SinglePhaseSimulation",
    "MultiphaseSimulation",
]

