"""Single-phase LBM step operator.

Registered as ``update_timestep:single_phase`` via the operator registry.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from operators.force import compute_total_force_ext
from operators.step._common import _apply_common_step
from registry import update_timestep_operator
from state import update_extra_state

if TYPE_CHECKING:
    from setup import SimulationSetup
    from state.state import State


@update_timestep_operator(name="single_phase")
def step_single_phase(setup: SimulationSetup, state: State) -> State:
    """Single-phase LBM step using prebuilt operator closures from setup.

    Args:
        setup: Closed-over :class:`~setup.simulation_setup.SimulationSetup`.
        state: Current :class:`~state.state.State`.

    Returns:
        Updated :class:`~state.state.State` after one time step.
    """
    # 1. External forces
    force_ext, state = compute_total_force_ext(setup, state, setup.forces)

    # 2. Macroscopic fields
    if force_ext is not None:
        rho, u, force_tot = setup.macroscopic_fn(state.f, setup.lattice, force=force_ext)
    else:
        rho, u = setup.macroscopic_fn(state.f, setup.lattice)
        force_tot = None

    # 3-6. Equilibrium → collision → streaming → BCs (shared)
    new_state = _apply_common_step(setup, state, rho, u, force_tot)
    return update_extra_state(setup, state, new_state, force_ext=force_ext)
