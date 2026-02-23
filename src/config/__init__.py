"""Configuration package for TUD-LBM simulations.

This package provides the configuration interface for TUD-LBM simulations.
The primary interface is SimulationBundle::

    from config import SimulationBundle, SinglePhaseConfig, RunnerConfig
    from core import Run

    bundle = SimulationBundle(
        simulation=SinglePhaseConfig(grid_shape=(100, 100), tau=0.6, nt=10000),
        runner=RunnerConfig(save_interval=1000),
    )
    sim = Run(bundle)
    sim.run()

Main exports:

- Simulation configuration (primary interface)
  - :class:`SimulationBundle`: Top-level composite containing all simulation
    parameters. Pass directly to Run().
  - :class:`SinglePhaseConfig`: Physics config for single-phase simulations.
  - :class:`MultiphaseConfig`: Physics config for multiphase simulations.
  - :class:`RunnerConfig`: Runner/IO configuration.

- Directory configuration
  - :data:`BASE_RESULTS_DIR`: Base directory used for saving results.

- JAX runtime configuration
  - :func:`configure_jax`: Applies common JAX flags from a central place.
  - :data:`ENABLE_X64`: Default for 64-bit precision in JAX.
  - :data:`DISABLE_JIT`: Default toggle to disable JIT for debugging.

- Saving / output configuration
  - :data:`DEFAULT_SAVE_FIELDS`: Default fields written to disk.
  - :data:`AVAILABLE_FIELDS`: All supported output fields.
  - :data:`FORCE_REGISTRY`: Mapping from force names to force implementations.

- File adapters
  - :class:`ConfigAdapter`: Abstract base for config file readers.
  - :class:`TomlAdapter`: TOML file adapter.
  - :func:`get_adapter`: Factory that picks the right adapter by file extension.
"""

# Directory configuration
from config.dir_config import BASE_RESULTS_DIR

# Saving configuration
from config.saving_config import (
    DEFAULT_SAVE_FIELDS,
    AVAILABLE_FIELDS,
    FORCE_REGISTRY,
)

# JAX config
from config.jax_config import configure_jax, ENABLE_X64, DISABLE_JIT

# Simulation config dataclasses (primary interface)
from config.simulation_config import (
    BaseSimulationConfig,
    SinglePhaseConfig,
    MultiphaseConfig,
    RunnerConfig,
    SimulationConfig,
    SimulationBundle,
)

# File adapters
from config.adapter_base import ConfigAdapter, get_adapter
from config.adapter_toml import TomlAdapter

__all__ = [
    # Primary interface
    "SimulationBundle",
    # Simulation config dataclasses
    "BaseSimulationConfig",
    "SinglePhaseConfig",
    "MultiphaseConfig",
    "RunnerConfig",
    "SimulationConfig",
    # Directory config
    "BASE_RESULTS_DIR",
    # JAX config
    "configure_jax",
    "ENABLE_X64",
    "DISABLE_JIT",
    # Saving config
    "DEFAULT_SAVE_FIELDS",
    "AVAILABLE_FIELDS",
    "FORCE_REGISTRY",
    # File adapters
    "ConfigAdapter",
    "TomlAdapter",
    "get_adapter",
]
