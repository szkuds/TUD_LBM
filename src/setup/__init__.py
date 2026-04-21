"""JAX-friendly simulation setup objects for TUD-LBM.

Public API::

    from setup import SimulationSetup, build_setup
    from setup import Lattice, build_lattice
    from setup import BCMasks, MultiphaseParams
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from setup.lattice import Lattice
from setup.lattice import build_lattice

# Re-export from their canonical locations; avoid circular imports
# by using TYPE_CHECKING for type hints and lazy imports for runtime
if TYPE_CHECKING:
    from operators.boundary import BCMasks
    from operators.macroscopic import MultiphaseParams
    from setup.simulation_setup import SimulationSetup


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name == "SimulationSetup":
        from setup.simulation_setup import SimulationSetup

        return SimulationSetup
    if name == "build_setup":
        from setup.simulation_setup import build_setup

        return build_setup
    if name == "BCMasks":
        from operators.boundary import BCMasks

        return BCMasks
    if name == "build_bc_masks":
        from operators.boundary import build_bc_masks

        return build_bc_masks
    if name == "MultiphaseParams":
        from operators.macroscopic import MultiphaseParams

        return MultiphaseParams
    if name == "build_multiphase_params":
        from operators.macroscopic import build_multiphase_params

        return build_multiphase_params
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BCMasks",
    "Lattice",
    "MultiphaseParams",
    "SimulationSetup",
    "build_bc_masks",
    "build_lattice",
    "build_multiphase_params",
    "build_setup",
]
