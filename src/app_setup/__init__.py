"""Configuration package for TUD-LBM simulations.

The primary interface is :class:`SimulationSetup`::

    from app_setup import SimulationSetup

    setup = SimulationSetup(
        grid_shape=(100, 100),
        tau=0.6,
        nt=10000,
        save_interval=1000
    )

Registry-aware validation
~~~~~~~~~~~~~~~~~~~~~~~~~
``SimulationSetup`` validates ``collision_scheme`` and ``eos`` against
the global operator registry at construction time.

See ``dev_notes/OperatorRegistry.md`` for the full developer guide.
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

# Primary interface — single flat configuration dataclass
from app_setup.simulation_setup import SimulationSetup

# File adapters
from app_setup.adapter_base import ConfigAdapter, get_adapter
from app_setup.adapter_toml import TomlAdapter

__all__ = [
    # Primary interface
    "SimulationSetup",
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
