"""JAX-friendly simulation setup objects for TUD-LBM.

Public API::

    from setup import SimulationSetup, build_setup
    from setup import Lattice, build_lattice
    from setup import BCMasks, MultiphaseParams
"""

from setup.lattice import Lattice, build_lattice
from setup.simulation_setup import (
    BCMasks,
    MultiphaseParams,
    SimulationSetup,
    build_bc_masks,
    build_multiphase_params,
    build_setup,
)

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
