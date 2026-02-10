"""Configuration key mappings for TUD-LBM simulations.

This module contains all the key sets and mappings used when parsing
configuration files. Having them in one place provides a clear overview
of all available configuration options.
"""

# =============================================================================
# Simulation Section Keys
# =============================================================================

#: Keys that map directly from [simulation] section to Run() kwargs
SIMULATION_DIRECT_KEYS: list[str] = [
    "tau",
    "nt",
    "save_interval",
    "skip_interval",
    "init_type",
    "init_dir",
    "simulation_name",
]

#: Keys that require special handling in [simulation] section
SIMULATION_SPECIAL_KEYS: dict[str, str] = {
    "type": "simulation_type",  # Renamed in kwargs
    "grid_shape": "grid_shape",  # Converted to tuple
    "lattice_type": "lattice_type",
}


# =============================================================================
# Multiphase Section Keys
# =============================================================================

#: Keys from [multiphase] section
MULTIPHASE_KEYS: list[str] = [
    "kappa",
    "rho_l",
    "rho_v",
    "interface_width",
    "eos",
]


# =============================================================================
# Collision Section Keys
# =============================================================================

#: MRT-specific keys from [collision] section
COLLISION_MRT_KEYS: list[str] = [
    "k_diag",
    "k0",
    "kb",
    "k1",
    "k2",
    "k3",
    "k4",
]


# =============================================================================
# Boundary Condition Keys
# =============================================================================

#: Valid boundary sides
BOUNDARY_SIDES: list[str] = [
    "left",
    "right",
    "top",
    "bottom",
]

#: Nested parameter sections in boundary_conditions
BOUNDARY_NESTED_PARAMS: list[str] = [
    "wetting_params",
    "hysteresis_params",
]


# =============================================================================
# Output Section Keys
# =============================================================================

#: Keys from [output] section with their kwargs mapping
OUTPUT_KEY_MAPPING: dict[str, str] = {
    "results_dir": "results_dir",
    "simulation_name": "simulation_name",
    "fields": "save_fields",
}


# =============================================================================
# TOML Restructuring Keys (for saving config back to TOML)
# =============================================================================

#: Keys to include in [simulation] section when writing TOML
TOML_SIMULATION_KEYS: list[str] = [
    "simulation_type",
    "grid_shape",
    "lattice_type",
    "tau",
    "nt",
    "save_interval",
    "skip_interval",
    "init_type",
    "init_dir",
    "simulation_name",
]

#: Keys to include in [multiphase] section when writing TOML
TOML_MULTIPHASE_KEYS: list[str] = [
    "kappa",
    "rho_l",
    "rho_v",
    "interface_width",
    "eos",
]

#: Keys to include in [output] section when writing TOML
TOML_OUTPUT_KEYS: list[str] = [
    "results_dir",
    "save_fields",
]
