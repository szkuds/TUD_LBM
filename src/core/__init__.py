"""Simulation runner for LBM simulations.

Provides a high-level interface for configuring and executing
lattice Boltzmann simulations with automatic setup and I/O.

Classes:
    Run: Main entry point for running LBM simulations (thin façade).
    ConfigLoader: Handles config loading and normalisation.
    SimulationFactory: Creates simulation instances from config.
    SimulationRunner: Owns time loop, saving, and NaN checking.
    StepResult: Standardized result dataclass from simulation timesteps.
"""

from .run import Run
from .config_loader import ConfigLoader
from .simulation_factory import SimulationFactory
from .simulation_runner import SimulationRunner
from .step_result import StepResult

__all__ = ["Run", "ConfigLoader", "SimulationFactory", "SimulationRunner", "StepResult"]

