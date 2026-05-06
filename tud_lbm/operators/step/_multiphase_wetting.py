"""Fixed wetting multiphase LBM step operator.

Registered as ``update_timestep:multiphase_wetting`` via the operator registry.

This step function is used when wetting is configured but hysteresis is NOT active.
The wetting parameters (phi_left, phi_right, d_rho_left, d_rho_right) are constant
across all timesteps, baked into setup.gradient_density and setup.laplacian_density
at setup time (Phase 3, Case 2).

The function body is structurally identical to step_multiphase; the distinction is
purely in how setup.gradient_density and setup.laplacian_density are constructed.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from tud_lbm.operators.force import compute_total_force_ext
from tud_lbm.operators.step._common import _multiphase_pipeline
from tud_lbm.pipeline.state import update_extra_state
from tud_lbm.registry import update_timestep_operator

if TYPE_CHECKING:
    from tud_lbm.pipeline.setup import SimulationSetup
    from tud_lbm.pipeline.state.state import State


@update_timestep_operator(name="multiphase_wetting")
def step_multiphase_wetting(setup: SimulationSetup, state: State) -> State:
    """Fixed wetting multiphase LBM step.

    Used when wetting is configured but hysteresis is NOT active.
    The wetting parameters are constant and baked into the density operators
    at setup time.

    The implementation is identical to step_multiphase: wetting correction is
    already applied in setup.gradient_density and setup.laplacian_density.

    Args:
        setup: Closed-over :class:`~tud_lbm.pipeline.setup.SimulationSetup`.
               setup.gradient_density and setup.laplacian_density are
               wetting-corrected with fixed parameters.
        state: Current :class:`~tud_lbm.pipeline.state.state.State`.

    Returns:
        Updated :class:`~tud_lbm.pipeline.state.state.State` after one time step.
    """
    # 1. Compute external forces
    force_ext, state = compute_total_force_ext(setup, state, setup.forces)

    # 2. Run multiphase physics kernel
    # (gradient_density and laplacian_density already include wetting correction)
    f_out, rho, u, force_tot = _multiphase_pipeline(
        setup,
        state.f,
        force_ext,
        setup.gradient_density,
        setup.laplacian_density,
    )

    # 3. Create new state with updated fields
    _new_state = state._replace(
        f=f_out,
        rho=rho,
        u=u,
        force=force_tot,
        t=state.t + 1,
    )

    # 4. Update extra state (plugins: electric potential, etc.)
    # Note: Wetting plugin is NOT active for this case (no state.wetting)
    return update_extra_state(setup, state, _new_state, force_ext=force_ext)
