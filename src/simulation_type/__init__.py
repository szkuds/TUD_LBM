"""Simulation classes for LBM.

Provides simulation_type containers that orchestrate the complete LBM workflow
including initialisation, time-stepping, and field management.

Classes:
    BaseSimulation: Abstract base class defining the simulation_type interface.
    SinglePhaseSimulation: Single-phase flow simulation_type.
    MultiphaseSimulation: Multiphase (two-phase) flow simulation_type.

Configuration dataclasses (import from app_setup module):
    from app_setup import SinglePhaseConfig, MultiphaseConfig
"""

from .base import BaseSimulation
from .multiphase import MultiphaseSimulation
from .single_phase import SinglePhaseSimulation

__all__ = [
    "BaseSimulation",
    "SinglePhaseSimulation",
    "MultiphaseSimulation",
]

