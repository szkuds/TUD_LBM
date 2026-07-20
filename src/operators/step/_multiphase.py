"""Multiphase LBM step operator.

Registered as ``update_timestep:multiphase`` via the operator registry.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators.force import compute_total_force_ext
from src.operators.step._common import _multiphase_pipeline
from src.pipeline.state import update_extra_state
from src.registry import update_timestep_operator

if TYPE_CHECKING:
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State


@update_timestep_operator(name="multiphase")
def step_multiphase(setup: SimulationSetup, state: State) -> State:
    """Multiphase LBM step — the canonical multiphase time-stepping operator.

    This is the main time-stepping function for multiphase simulations.
    It runs the complete multiphase LBM pipeline once and updates the state.

    The implementation is minimal:
    1. Compute external forces
    2. Run the physics kernel (_multiphase_pipeline)
    3. Update extra state (wetting, etc.) if active
    4. Return updated state

    Args:
        setup: Closed-over :class:`~src.pipeline.setup.SimulationSetup`.
        state: Current :class:`~src.pipeline.state.state.State`.

    Returns:
        Updated :class:`~src.pipeline.state.state.State` after one time step.
    """
    if setup.gradient_density is None:
        msg = "gradient_density is required for multiphase step"
        raise TypeError(msg)
    if setup.laplacian_density is None:
        msg = "laplacian_density is required for multiphase step"
        raise TypeError(msg)

    # 1. Compute external forces
    force_ext, state = compute_total_force_ext(setup, state, setup.forces)

    # 2. Run multiphase physics kernel
    f_out, rho, u, force_tot = _multiphase_pipeline(
        setup,
        state.f,
        force_ext,
        setup.gradient_density,
        setup.laplacian_density,
    )

    # 3. Create new state with updated fields
    new_state = state._replace(
        f=f_out,
        rho=rho,
        u=u,
        force=force_tot,
        force_ext=force_ext,
        t=state.t + 1,
    )

    # 4. Update extra state (plugins: wetting, electric potential, etc.)
    return update_extra_state(setup, state, new_state, force_ext=force_ext)
