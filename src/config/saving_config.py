"""Saving configuration constants for TUD-LBM simulations.

This module contains all constants related to data saving, fields,
and force registries.
"""

# =============================================================================
# Field Constants
# =============================================================================

#: Default fields to save during simulation
DEFAULT_SAVE_FIELDS: list[str] = ["rho", "u"]

#: All available fields that can be saved
AVAILABLE_FIELDS: list[str] = ["rho", "u", "force", "force_ext", "f", "h"]


# =============================================================================
# Force Registry
# =============================================================================

#: Registry mapping force type names to their import paths and class names
FORCE_REGISTRY: dict[str, tuple[str, str]] = {
    "gravity_multiphase": ("tud_lbm.operators", "GravityForceMultiphase"),
    "electric": ("tud_lbm.operators", "ElectricForce"),
}
