"""Configuration package for TUD-LBM simulations.

This package groups all user-facing configuration utilities and constants used across
TUD-LBM. The public surface is re-exported here so callers can do::

    from config import load, BASE_RESULTS_DIR

The main configuration areas are:

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

- Key mappings for structured configs (e.g., TOML)
  - :data:`SIMULATION_DIRECT_KEYS`, :data:`SIMULATION_SPECIAL_KEYS`
  - :data:`MULTIPHASE_KEYS`, :data:`COLLISION_MRT_KEYS`
  - :data:`BOUNDARY_SIDES`, :data:`BOUNDARY_NESTED_PARAMS`
  - :data:`OUTPUT_KEY_MAPPING`
  - :data:`TOML_SIMULATION_KEYS`, :data:`TOML_MULTIPHASE_KEYS`,
    :data:`TOML_OUTPUT_KEYS`

- Config loading
  - :func:`load`: Load and normalise a structured config into ``Run(...)`` kwargs.
  - :func:`load_raw`: Load a config file without normalisation.

- Simulation configuration dataclasses
  - :class:`BaseSimulationConfig`: Base config shared by all simulation types.
  - :class:`SinglePhaseConfig`: Config for single-phase simulations.
  - :class:`MultiphaseConfig`: Config for multiphase simulations.
  - :class:`RunnerConfig`: Config for the simulation runner (I/O, saving).

"""

# Directory configuration
from config.dir_config import BASE_RESULTS_DIR

# Saving configuration
from config.saving_config import (
    DEFAULT_SAVE_FIELDS,
    AVAILABLE_FIELDS,
    FORCE_REGISTRY,
)

# Jax config loader
from config.jax_config import configure_jax, ENABLE_X64, DISABLE_JIT

# Key mappings
from config.keys import (
    SIMULATION_DIRECT_KEYS,
    SIMULATION_SPECIAL_KEYS,
    MULTIPHASE_KEYS,
    COLLISION_MRT_KEYS,
    BOUNDARY_SIDES,
    BOUNDARY_NESTED_PARAMS,
    OUTPUT_KEY_MAPPING,
    TOML_SIMULATION_KEYS,
    TOML_MULTIPHASE_KEYS,
    TOML_OUTPUT_KEYS,
)

# Config loader
from config.config_loader import load, load_raw

# Simulation configuration dataclasses
from config.simulation_config import (
    BaseSimulationConfig,
    SinglePhaseConfig,
    MultiphaseConfig,
    RunnerConfig,
    RUN_DEFAULTS,
)

__all__ = [
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
    # Key mappings
    "SIMULATION_DIRECT_KEYS",
    "SIMULATION_SPECIAL_KEYS",
    "MULTIPHASE_KEYS",
    "COLLISION_MRT_KEYS",
    "BOUNDARY_SIDES",
    "BOUNDARY_NESTED_PARAMS",
    "OUTPUT_KEY_MAPPING",
    "TOML_SIMULATION_KEYS",
    "TOML_MULTIPHASE_KEYS",
    "TOML_OUTPUT_KEYS",
    # Config loader
    "load",
    "load_raw",
    # Simulation config dataclasses
    "BaseSimulationConfig",
    "SinglePhaseConfig",
    "MultiphaseConfig",
    "RunnerConfig",
    "RUN_DEFAULTS",
]
