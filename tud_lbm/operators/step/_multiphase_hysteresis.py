"""Hysteresis-enabled multiphase LBM step operator.

Registered as ``update_timestep:multiphase_hysteresis`` via the operator registry.

This step function is used when both wetting and hysteresis are configured.
The wetting parameters (phi_left, phi_right, d_rho_left, d_rho_right) are
optimized every timestep by the hysteresis optimizer, using wetting parametric
operators from setup (gradient_density_wetting, laplacian_density_wetting).
"""

from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
from tud_lbm.operators.force import compute_total_force_ext
from tud_lbm.operators.step._common import _multiphase_pipeline
from tud_lbm.pipeline.state import update_extra_state
from tud_lbm.registry import update_timestep_operator

if TYPE_CHECKING:
    import jax.numpy as jnp
    from jax import Array
    from tud_lbm.operators.protocols import DifferentialOperator
    from tud_lbm.operators.wetting._params import WettingParams
    from tud_lbm.pipeline.setup import SimulationSetup
    from tud_lbm.pipeline.state.state import State


# ── Part A: Helper to build operators from live wetting state ──


def _make_wetting_ops(setup: SimulationSetup, wetting: State) -> tuple[DifferentialOperator, DifferentialOperator]:
    """Build (grid)->result operators from live wetting parameters.

    Wraps the wetting parametric operators (gradient_density_wetting, laplacian_density_wetting)
    by injecting live wetting parameters from state.wetting.

    Args:
        setup: :class:`~tud_lbm.pipeline.setup.SimulationSetup`.
        wetting: Current :class:`~tud_lbm.pipeline.state.state.WettingState`.

    Returns:
        Tuple of (gradient_closure, laplacian_closure) where each is
        a callable (grid) -> result.
    """

    def grad(grid: jnp.ndarray) -> jnp.ndarray:
        return setup.gradient_density_wetting(
            grid, wetting.phi_left, wetting.phi_right, wetting.d_rho_left, wetting.d_rho_right
        )

    def lap(grid: jnp.ndarray) -> jnp.ndarray:
        return setup.laplacian_density_wetting(
            grid, wetting.phi_left, wetting.phi_right, wetting.d_rho_left, wetting.d_rho_right
        )

    return grad, lap


# ── Part B: Trial step helper for hysteresis optimiser ──


def _trial_step(
    setup: SimulationSetup,
    f_t: jnp.ndarray,
    force_ext: jnp.ndarray,
    params: WettingParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One or more multiphase passes with candidate wetting parameters.

    Called by the hysteresis optimiser to evaluate trial parameter sets.
    Runs a multiphase physics pass with the given wetting parameters,
    returning the post-step populations and density field.
    Multiple steps can be taken per trial to strengthen the optimizer response.

    Args:
        setup: :class:`~tud_lbm.pipeline.setup.SimulationSetup`.
        f_t: Current populations.
        force_ext: External force field.
        params: Candidate :class:`~tud_lbm.operators.wetting._params.WettingParams`.

    Returns:
        Tuple (f_out, rho_out) where f_out is post-BC populations and
        rho_out is the post-step density field.
    """

    def grad(grid: jnp.ndarray) -> jnp.ndarray:
        return setup.gradient_density_wetting(
            grid, params.phi_left, params.phi_right, params.d_rho_left, params.d_rho_right
        )

    def lap(grid: jnp.ndarray) -> jnp.ndarray:
        return setup.laplacian_density_wetting(
            grid, params.phi_left, params.phi_right, params.d_rho_left, params.d_rho_right
        )

    num_steps: int = setup.config.hysteresis_config.get("trial_steps", 2)

    def body_fn(carry_f: jnp.ndarray, _) -> tuple[Array, Array]:  # noqa: ANN001
        f_next, rho_next, _u, _force_tot = _multiphase_pipeline(setup, carry_f, force_ext, grad, lap)
        return f_next, rho_next

    import jax

    f_out, rho_out_all = jax.lax.scan(body_fn, f_t, None, length=num_steps)

    # We only need the rho from the last trial step
    rho = rho_out_all[-1] if num_steps > 1 else rho_out_all[0]

    return f_out, rho


# ── Part C: The hysteresis step function ──


@update_timestep_operator(name="multiphase_hysteresis")
def step_multiphase_hysteresis(setup: SimulationSetup, state: State) -> State:
    """Hysteresis-enabled multiphase LBM step.

    Used when both wetting and hysteresis are configured.
    The wetting parameters are optimized every timestep via the hysteresis
    optimizer using wetting parametric operators and trial steps.

    The implementation:
    1. Compute external forces
    2. Build operators from live wetting parameters
    3. Run multiphase physics kernel
    4. Optimize wetting parameters via hysteresis
    5. Update extra state (plugins)
    6. Return updated state

    Args:
        setup: Closed-over :class:`~tud_lbm.pipeline.setup.SimulationSetup`.
               setup.gradient_density_wetting and setup.laplacian_density_wetting are
               populated with parametric closures.
        state: Current :class:`~tud_lbm.pipeline.state.state.State`.
               state.wetting is a :class:`~tud_lbm.pipeline.state.state.WettingState`
               with parameters to optimize.

    Returns:
        Updated :class:`~tud_lbm.pipeline.state.state.State` after one time step
        with optimized wetting parameters.
    """
    # 1. Compute external forces
    force_ext, state = compute_total_force_ext(setup, state, setup.forces)

    # 2. Build operators from live wetting parameters
    grad, lap = _make_wetting_ops(setup, state.wetting)

    # 3. Run multiphase physics kernel
    f_out, rho, u, force_tot = _multiphase_pipeline(setup, state.f, force_ext, grad, lap)

    # 4. Create new state (wetting updated by plugin via trial_step_fn in context)
    new_state = state._replace(f=f_out, rho=rho, u=u, force=force_tot, t=state.t + 1)

    # 5. Update extra state — trial_step_fn forwarded so wetting plugin can run optimiser
    return update_extra_state(
        setup,
        state,
        new_state,
        trial_step_fn=partial(_trial_step, setup, f_out, force_ext),
    )
