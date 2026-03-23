"""JAX-friendly simulation setup objects for TUD-LBM.

Public API::

    from setup import SimulationSetup, build_setup
    from setup import Lattice, build_lattice
    from setup import BCMasks, MultiphaseParams
"""

from setup.lattice import Lattice
from setup.lattice import build_lattice
from setup.simulation_setup import BCMasks
from setup.simulation_setup import MultiphaseParams
from setup.simulation_setup import SimulationSetup
from setup.simulation_setup import build_bc_masks
from setup.simulation_setup import build_multiphase_params
from setup.simulation_setup import build_setup

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
