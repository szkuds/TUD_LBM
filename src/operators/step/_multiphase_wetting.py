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
from typing import cast
import jax.numpy as jnp
from src.operators.force import compute_total_force_ext
from src.operators.step._common import _multiphase_pipeline
from src.operators.wetting._contact_angle import compute_contact_angle
from src.operators.wetting._contact_line import compute_contact_line_location
from src.pipeline.state import update_extra_state
from src.registry import update_timestep_operator

if TYPE_CHECKING:
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State
    from src.pipeline.state.state import WettingState


@update_timestep_operator(name="multiphase_wetting")
def step_multiphase_wetting(setup: SimulationSetup, state: State) -> State:
    """Fixed wetting multiphase LBM step.

    Used when wetting is configured but hysteresis is NOT active.
    The wetting parameters are constant and baked into the density operators
    at setup time.

    The implementation is identical to step_multiphase: wetting correction is
    already applied in setup.gradient_density and setup.laplacian_density.

    Args:
        setup: Closed-over :class:`~src.pipeline.setup.SimulationSetup`.
               setup.gradient_density and setup.laplacian_density are
               wetting-corrected with fixed parameters.
        state: Current :class:`~src.pipeline.state.state.State`.

    Returns:
        Updated :class:`~src.pipeline.state.state.State` after one time step.
    """
    if setup.gradient_density is None:
        msg = "gradient_density is required for multiphase wetting step"
        raise TypeError(msg)
    if setup.laplacian_density is None:
        msg = "laplacian_density is required for multiphase wetting step"
        raise TypeError(msg)

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

    updated_wetting = state.wetting
    if state.wetting is not None:
        if setup.multiphase_params is None:
            msg = "multiphase_params is required for contact angle computation"
            raise TypeError(msg)
        mp = setup.multiphase_params
        rho_mean = 0.5 * (mp.rho_l + mp.rho_v)
        if setup.wetting_edge is None:
            msg = "wetting_edge is required for contact angle computation"
            raise TypeError(msg)
        edge = setup.wetting_edge
        ca_left, ca_right = compute_contact_angle(rho, jnp.array(rho_mean), edge=edge)
        cll_left, cll_right = compute_contact_line_location(rho, ca_left, ca_right, jnp.array(rho_mean), edge=edge)
        wetting = cast("WettingState", state.wetting)
        updated_wetting = wetting._replace(
            ca_left=ca_left,
            ca_right=ca_right,
            cll_left=cll_left,
            cll_right=cll_right,
        )

    # 3. Create new state with updated fields
    _new_state = state._replace(
        f=f_out,
        rho=rho,
        u=u,
        force=force_tot,
        force_ext=force_ext,
        t=state.t + 1,
        wetting=updated_wetting,
    )

    # 4. Update extra state (plugins: electric potential, etc.)
    return update_extra_state(setup, state, _new_state, force_ext=force_ext)
