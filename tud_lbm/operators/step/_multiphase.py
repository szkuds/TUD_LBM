"""Multiphase LBM step operator.

Registered as ``update_timestep:multiphase`` via the operator registry.
Supports both wetting and non-wetting multiphase simulations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from tud_lbm.operators.force import compute_total_force_ext
from tud_lbm.operators.step._common import _apply_common_step
from tud_lbm.operators.step._wetting_differential_operators import _make_wetting_differential_ops
from tud_lbm.pipeline.state import update_extra_state
from tud_lbm.pipeline.state.state import State
from tud_lbm.pipeline.state.state import WettingState
from tud_lbm.registry import update_timestep_operator

if TYPE_CHECKING:
    from tud_lbm.operators.differential import DifferentialOperator
    from tud_lbm.pipeline.setup import SimulationSetup


def _run_multiphase_pipeline(
    setup: SimulationSetup,
    f_t: jnp.ndarray,
    *,
    force_ext: jnp.ndarray = None,
    wetting: WettingState | None = None,
    gradient_density: DifferentialOperator = None,
    laplacian_density: DifferentialOperator = None,
) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run one multiphase pipeline pass and return ``(f_out, rho, u, force_tot)``."""
    if gradient_density is None or laplacian_density is None:
        if wetting is not None and setup.config.wetting_config is not None:
            wet_grad, wet_lap = _make_wetting_differential_ops(setup, wetting)
        else:
            wet_grad = setup.gradient_density
            wet_lap = setup.laplacian_density
        if gradient_density is None:
            gradient_density = wet_grad
        if laplacian_density is None:
            laplacian_density = wet_lap

    rho, u, force_tot = setup.macroscopic_fn(
        f_t,
        setup.lattice,
        setup.multiphase_params,
        force_ext,
        gradient_standard=setup.gradient_standard,
        laplacian_density=laplacian_density,
    )

    # Reuse the shared equilibrium→collision→streaming→BC path for consistency.
    temp_state = State(
        f=f_t,
        rho=rho,
        u=u,
        t=jnp.array(0),
        wetting=wetting,
    )
    next_state = _apply_common_step(setup, temp_state, rho, u, force_tot, gradient_density=gradient_density)
    return next_state.f, rho, u, force_tot


def multiphase_step(
    setup: SimulationSetup,
    f_t: jnp.ndarray,
    *,
    force_ext: jnp.ndarray = None,
    wetting: WettingState | None = None,
    gradient_density: DifferentialOperator = None,
    laplacian_density: DifferentialOperator = None,
) -> jnp.ndarray:
    """Run one multiphase trial step and return post-BC populations."""
    f_out, _rho, _u, _force_tot = _run_multiphase_pipeline(
        setup,
        f_t,
        force_ext=force_ext,
        wetting=wetting,
        gradient_density=gradient_density,
        laplacian_density=laplacian_density,
    )
    return f_out


@update_timestep_operator(name="multiphase")
def step_multiphase(setup: SimulationSetup, state: State) -> State:
    """Multiphase LBM step for both wetting and non-wetting simulations.

    When wetting is not active (state.wetting is None), uses the prebuilt
    gradient_density and laplacian_density closures from setup.

    When wetting is active (state.wetting is not None), builds wetting shims
    that inject live wetting parameters into the density operators at each step.

    Args:
        setup: Closed-over :class:`~tud_lbm.pipeline.setup.SimulationSetup`.
        state: Current :class:`~tud_lbm.pipeline.state.state.State`.

    Returns:
        Updated :class:`~tud_lbm.pipeline.state.state.State` after one time step.
    """
    # 1. External forces
    force_ext, state = compute_total_force_ext(setup, state, setup.forces)

    f_next, rho, u, force_tot = _run_multiphase_pipeline(
        setup,
        state.f,
        force_ext=force_ext,
        wetting=state.wetting,
    )
    new_state = state._replace(
        f=f_next,
        rho=rho,
        u=u,
        t=state.t + 1,
    )

    new_state = update_extra_state(setup, state, new_state, force_ext=force_ext)

    return new_state._replace(
        force=force_tot,
    )
