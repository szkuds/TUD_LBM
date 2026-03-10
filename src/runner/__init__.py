"""Simulation runner for LBM simulations.

Provides a high-level interface for configuring and executing
lattice Boltzmann simulations with automatic app_setup and I/O.

Classes:
    Run: Main entry point for running LBM simulations.
    SimulationFactory: Creates simulation_type instances from app_setup.
    SimulationRunner: Owns time loop, saving, and NaN checking.
    StepResult: Standardized result dataclass from simulation_type timesteps.
"""

from .run import Run
from .simulation_factory import SimulationFactory
from .simulation_runner import SimulationRunner
from .step_result import StepResult

__all__ = ["Run", "SimulationFactory", "SimulationRunner", "StepResult"]
