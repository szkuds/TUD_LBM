"""Application layer: composition and execution.

Modules:
- setup.py     : build_setup() — wires operators, composition root
- runner.py    : run(), init_state() — timestep loop, functional API
- step.py      : kernel(), step(), observe() — the one time step and its derived fields
- state/       : State (JAX pytree)
"""

from src.pipeline.runner import init_state
from src.pipeline.runner import run
from src.pipeline.setup import build_setup
from src.pipeline.state import WettingState
from src.pipeline.state.state import State

__all__ = ["State", "WettingState", "build_setup", "init_state", "run"]
