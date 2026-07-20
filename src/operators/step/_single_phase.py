"""Single-phase LBM step operator.

Registered as ``update_timestep:single_phase`` via the operator registry.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.force import compute_total_force_ext
from src.operators.step._common import _apply_common_step
from src.pipeline.state import update_extra_state
from src.registry import update_timestep_operator

if TYPE_CHECKING:
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State


@update_timestep_operator(name="single_phase")
def step_single_phase(setup: SimulationSetup, state: State) -> State:
    """Single-phase LBM step using prebuilt operator closures from setup.

    Args:
        setup: Closed-over :class:`~src.pipeline.setup.SimulationSetup`.
        state: Current :class:`~src.pipeline.state.state.State`.

    Returns:
        Updated :class:`~src.pipeline.state.state.State` after one time step.
    """
    if setup.macroscopic_fn is None:
        msg = "macroscopic_fn is required for single phase step"
        raise TypeError(msg)

    # 1. External forces
    force_ext, state = compute_total_force_ext(setup, state, setup.forces)

    # 2. Macroscopic fields
    rho, u, force_tot = setup.macroscopic_fn(state.f, setup.lattice, force=force_ext)

    # 3-6. Equilibrium → collision → streaming → BCs (shared)
    new_state = _apply_common_step(setup, state, rho, u, force_tot)
    new_state = new_state._replace(force=force_tot, force_ext=force_ext)

    # Zero velocity inside the obstacle for diagnostic/plotting purposes only —
    # f and next-step dynamics are unaffected, since only state.u (not state.f) is touched here.
    if setup.obstacle_mask is not None:
        new_state = new_state._replace(u=jnp.where(setup.obstacle_mask, 0.0, new_state.u))

    return update_extra_state(setup, state, new_state, force_ext=force_ext)
