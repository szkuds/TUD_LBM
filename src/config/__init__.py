"""Configuration package for TUD-LBM simulations.

This package provides configuration loading and constants for the TUD-LBM package.
"""

# Directory configuration
from config.dir_config import BASE_RESULTS_DIR

# Saving configuration
from config.saving_config import (
    DEFAULT_SAVE_FIELDS,
    AVAILABLE_FIELDS,
    FORCE_REGISTRY,
)

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

__all__ = [
    # Directory config
    "BASE_RESULTS_DIR",
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
]
