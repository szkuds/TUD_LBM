"""Configuration loader for TUD-LBM simulations.

This module bridges the gap between structured TOML/JSON config files
and the flat **kwargs that Run() expects.
"""

import json
import os
import sys
from typing import Any

# For Python < 3.11, use tomli; otherwise use tomllib from stdlib
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# Registry mapping force type names to their import paths and class names
FORCE_REGISTRY: dict[str, tuple[str, str]] = {
    "gravity_multiphase": ("tud_lbm.operators", "GravityForceMultiphase"),
    "electric": ("tud_lbm.operators", "ElectricForce"),
}


def _import_class(module_path: str, class_name: str) -> type:
    """Dynamically import a class from a module path.

    Args:
        module_path: The module path (e.g., 'tud_lbm.operators')
        class_name: The class name to import

    Returns:
        The imported class
    """
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _instantiate_forces(force_configs: list[dict], grid_shape: tuple) -> list:
    """Instantiate force objects from configuration.

    Args:
        force_configs: List of force configuration dictionaries
        grid_shape: The simulation grid shape (required by most force constructors)

    Returns:
        List of instantiated force objects
    """
    forces = []
    for force_cfg in force_configs:
        force_type = force_cfg.get("type")
        if force_type not in FORCE_REGISTRY:
            raise ValueError(
                f"Unknown force type: '{force_type}'. "
                f"Available types: {list(FORCE_REGISTRY.keys())}"
            )

        module_path, class_name = FORCE_REGISTRY[force_type]
        force_class = _import_class(module_path, class_name)

        # Build constructor kwargs (exclude 'type' key)
        kwargs = {k: v for k, v in force_cfg.items() if k != "type"}

        # Add grid_shape if not already present (most forces need it)
        if "grid_shape" not in kwargs:
            kwargs["grid_shape"] = grid_shape

        # Handle specific force constructor signatures
        if force_type == "gravity_multiphase":
            # GravityForceMultiphase(force_g, inclination_angle_deg, grid_shape)
            force_g = kwargs.pop("force_g")
            inclination_angle = kwargs.pop("inclination_angle", 0.0)
            forces.append(force_class(force_g, inclination_angle, grid_shape))
        elif force_type == "electric":
            # ElectricForce has more complex constructor
            forces.append(force_class(**kwargs))
        else:
            forces.append(force_class(**kwargs))

    return forces


def _build_bc_config(boundary_config: dict) -> dict:
    """Build the bc_config dictionary from TOML boundary_conditions section.

    Args:
        boundary_config: The boundary_conditions section from TOML

    Returns:
        The bc_config dict expected by MultiphaseSimulation
    """
    bc_config = {}

    # Copy direct boundary condition mappings
    for side in ["left", "right", "top", "bottom"]:
        if side in boundary_config:
            bc_config[side] = boundary_config[side]

    # Copy nested params
    if "wetting_params" in boundary_config:
        bc_config["wetting_params"] = dict(boundary_config["wetting_params"])

    if "hysteresis_params" in boundary_config:
        bc_config["hysteresis_params"] = dict(boundary_config["hysteresis_params"])

    return bc_config


def _parse_toml(config_data: dict) -> dict:
    """Parse TOML configuration data into Run() kwargs.

    Args:
        config_data: The parsed TOML data

    Returns:
        Dictionary of kwargs suitable for Run()
    """
    kwargs: dict[str, Any] = {}

    # --- Simulation section ---
    sim = config_data.get("simulation", {})

    if "type" in sim:
        kwargs["simulation_type"] = sim["type"]
    if "grid_shape" in sim:
        kwargs["grid_shape"] = tuple(sim["grid_shape"])
    if "lattice_type" in sim:
        kwargs["lattice_type"] = sim["lattice_type"]

    # Direct mappings
    for key in ["tau", "nt", "save_interval", "skip_interval", "init_type", "init_dir"]:
        if key in sim:
            kwargs[key] = sim[key]

    # --- Multiphase section ---
    mp = config_data.get("multiphase", {})
    for key in ["kappa", "rho_l", "rho_v", "interface_width", "eos"]:
        if key in mp:
            kwargs[key] = mp[key]

    # --- Collision section ---
    collision = config_data.get("collision", {})
    if collision:
        scheme = collision.get("scheme", "BGK")
        if scheme.upper() == "MRT":
            # Build MRT config dict
            collision_cfg = {"collision_scheme": "MRT"}
            for key in ["k_diag", "k0", "kb", "k1", "k2", "k3", "k4"]:
                if key in collision:
                    collision_cfg[key] = collision[key]
            kwargs["collision"] = collision_cfg
        else:
            kwargs["collision"] = scheme

    # --- Force section (array of tables) ---
    force_configs = config_data.get("force", [])
    if force_configs:
        grid_shape = kwargs.get("grid_shape", (100, 100))
        kwargs["force_enabled"] = True
        kwargs["force_obj"] = _instantiate_forces(force_configs, grid_shape)
    else:
        kwargs["force_enabled"] = False

    # --- Boundary conditions section ---
    bc = config_data.get("boundary_conditions", {})
    if bc:
        bc_config = _build_bc_config(bc)
        kwargs["bc_config"] = bc_config

        # Check for wetting
        if any(bc.get(side) == "wetting" for side in ["left", "right", "top", "bottom"]):
            kwargs["wetting_enabled"] = True

            # Copy wetting_params values to top-level kwargs if needed
            wetting_params = bc.get("wetting_params", {})
            if "phi_left" in wetting_params:
                kwargs["phi_value"] = wetting_params.get("phi_left", 1.0)
            if "d_rho_left" in wetting_params:
                kwargs["d_rho_value"] = wetting_params.get("d_rho_left", 0.0)

            # Copy wetting_params rho values to top-level if not set from multiphase
            if "rho_l" in wetting_params and "rho_l" not in kwargs:
                kwargs["rho_l"] = wetting_params["rho_l"]
            if "rho_v" in wetting_params and "rho_v" not in kwargs:
                kwargs["rho_v"] = wetting_params["rho_v"]

        # Hysteresis params - passed both in bc_config and as separate kwarg
        if "hysteresis_params" in bc:
            kwargs["hysteresis_params"] = dict(bc["hysteresis_params"])

    # --- Output section ---
    output = config_data.get("output", {})
    if "results_dir" in output:
        kwargs["results_dir"] = os.path.expanduser(output["results_dir"])
    if "simulation_name" in output:
        kwargs["simulation_name"] = output["simulation_name"]
    if "fields" in output:
        kwargs["save_fields"] = output["fields"]

    return kwargs


def load(path: str) -> dict:
    """Load TOML or JSON config and return flat kwargs dict for Run().

    Args:
        path: Path to the configuration file (.toml or .json)

    Returns:
        Dictionary of kwargs suitable for passing to Run()

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the file format is not supported
    """
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".toml":
        with open(path, "rb") as f:
            config_data = tomllib.load(f)
        return _parse_toml(config_data)

    elif ext == ".json":
        with open(path, "r") as f:
            config_data = json.load(f)
        # JSON files might already be in flat kwargs format
        # or they might be structured like TOML
        if "simulation" in config_data:
            return _parse_toml(config_data)
        else:
            # Assume it's already flat kwargs
            return config_data

    else:
        raise ValueError(
            f"Unsupported configuration file format: '{ext}'. "
            "Supported formats: .toml, .json"
        )


def load_raw(path: str) -> dict:
    """Load raw TOML/JSON config without transformation.

    This is useful for saving the original structured config.

    Args:
        path: Path to the configuration file

    Returns:
        Raw parsed configuration data
    """
    path = os.path.expanduser(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".toml":
        with open(path, "rb") as f:
            return tomllib.load(f)
    elif ext == ".json":
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported format: {ext}")
