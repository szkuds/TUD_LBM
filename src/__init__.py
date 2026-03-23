"""TUD-LBM: Lattice Boltzmann Method package from Delft University of Technology.

Public API::

    from config import SimulationConfig, from_toml, from_dict
    from setup import SimulationSetup, build_setup
    from state.state import State, WettingState
    from runner import run, init_state, step_single_phase, step_multiphase
"""

from config import SimulationConfig, from_toml, from_dict
from setup import SimulationSetup, build_setup
from state.state import State, WettingState
from runner import run, init_state, step_single_phase, step_multiphase

__all__ = [
    "SimulationConfig",
    "from_toml",
    "from_dict",
    "SimulationSetup",
    "build_setup",
    "State",
    "WettingState",
    "run",
    "init_state",
    "step_single_phase",
    "step_multiphase",
]
