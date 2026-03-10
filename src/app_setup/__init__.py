"""Configuration package for TUD-LBM simulations.

This package provides the configuration interface for TUD-LBM simulations.
The primary interface is SimulationBundle::

    from app_setup import SimulationBundle, SinglePhaseConfig, RunnerConfig
    from runner import Run

    bundle = SimulationBundle(
        simulation_type=SinglePhaseConfig(grid_shape=(100, 100), tau=0.6, nt=10000),
        runner=RunnerConfig(save_interval=1000),
    )
    sim = Run(bundle)
    sim.run()

Registry-aware validation
~~~~~~~~~~~~~~~~~~~~~~~~~
``SinglePhaseConfig`` and ``MultiphaseConfig`` validate ``collision_scheme``
and ``eos`` against the global operator registry (``src/registry.py``) at
construction time.  When new simulation_operators are registered via
``@register_operator``, they are automatically accepted by app_setup validation
— no hardcoded sets need updating.

See ``dev_notes/OperatorRegistry.md`` for the full developer guide.

Main exports:

- Simulation configuration (primary interface)
  - :class:`SimulationBundle`: Top-level composite containing all simulation_type
    parameters. Pass directly to Run().
  - :class:`SinglePhaseConfig`: Physics app_setup for single-phase simulations.
  - :class:`MultiphaseConfig`: Physics app_setup for multiphase simulations.
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
  - :data:`FORCE_REGISTRY`: Legacy force name mapping (prefer the global
    operator registry via ``get_operators("force")``).

- File adapters
  - :class:`ConfigAdapter`: Abstract base for app_setup file readers.
  - :class:`TomlAdapter`: TOML file adapter.
  - :func:`get_adapter`: Factory that picks the right adapter by file extension.
"""

# Directory configuration
from app_setup.dir_config import BASE_RESULTS_DIR

# Saving configuration
from app_setup.saving_config import (
    DEFAULT_SAVE_FIELDS,
    AVAILABLE_FIELDS,
    FORCE_REGISTRY,
)

# JAX app_setup
from app_setup.jax_config import configure_jax, ENABLE_X64, DISABLE_JIT

# Simulation app_setup dataclasses (primary interface)
from app_setup.simulation_config import (
    BaseSimulationConfig,
    SinglePhaseConfig,
    MultiphaseConfig,
    RunnerConfig,
    SimulationConfig,
    SimulationBundle,
)

# File adapters
from app_setup.adapter_base import ConfigAdapter, get_adapter
from app_setup.adapter_toml import TomlAdapter

__all__ = [
    # Primary interface
    "SimulationBundle",
    # Simulation app_setup dataclasses
    "BaseSimulationConfig",
    "SinglePhaseConfig",
    "MultiphaseConfig",
    "RunnerConfig",
    "SimulationConfig",
    # Directory app_setup
    "BASE_RESULTS_DIR",
    # JAX app_setup
    "configure_jax",
    "ENABLE_X64",
    "DISABLE_JIT",
    # Saving app_setup
    "DEFAULT_SAVE_FIELDS",
    "AVAILABLE_FIELDS",
    "FORCE_REGISTRY",
    # File adapters
    "ConfigAdapter",
    "TomlAdapter",
    "get_adapter",
]
