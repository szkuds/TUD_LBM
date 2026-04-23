"""Application layer: composition and execution.

Modules:
- setup.py     : build_setup() — wires operators, composition root
- runner.py    : run(), init_state() — timestep loop, functional API
- step.py      : Step executor
- state.py     : State, WettingState (JAX pytrees)
"""

from tud_lbm.pipeline.runner import init_state, run
from tud_lbm.pipeline.setup import build_setup
from tud_lbm.pipeline.state.state import State, WettingState

__all__ = ["build_setup", "run", "init_state", "State", "WettingState"]
