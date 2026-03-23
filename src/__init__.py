"""TUD-LBM: Lattice Boltzmann Method package from Delft University of Technology.

Public API::

    from config import SimulationConfig, from_toml, from_dict
    from setup import SimulationSetup, build_setup
    from state.state import State, WettingState
    from runner import run, init_state, step_single_phase, step_multiphase
"""

from config import SimulationConfig
from config import from_dict
from config import from_toml
from runner import init_state
from runner import run
from runner import step_multiphase
from runner import step_single_phase
from setup import SimulationSetup
from setup import build_setup
from state.state import State
from state.state import WettingState

__all__ = [
    "SimulationConfig",
    "SimulationSetup",
    "State",
    "WettingState",
    "build_setup",
    "from_dict",
    "from_toml",
    "init_state",
    "run",
    "step_multiphase",
    "step_single_phase",
]
