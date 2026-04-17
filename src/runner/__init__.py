"""Simulation runner for TUD-LBM.

Functional API — all orchestration uses pure functions and
``jax.lax.scan``.  No mutable classes are involved.

Public API::

    from runner import run, init_state
    from runner import step_single_phase, step_multiphase
"""

from .run import init_state
from .run import run

__all__ = [
    "init_state",
    "run",
]
