"""Hysteresis-enabled multiphase LBM step operator.

Registered as ``update_timestep:multiphase_hysteresis`` via the operator registry.

This step function is used when both wetting and hysteresis are configured.
The wetting parameters (phi_left, phi_right, d_rho_left, d_rho_right) are
optimised every timestep by the hysteresis optimiser, using wetting parametric
density operators from setup (gradient_density, laplacian_density), which
accept live wetting parameters.
"""

from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.force._force_aggregator import compute_total_force_ext
from src.operators.step._common import _multiphase_pipeline
from src.pipeline.state import update_extra_state
from src.registry import update_timestep_operator

if TYPE_CHECKING:
    from jax import Array
    from src.operators.protocols import DifferentialOperator
    from src.operators.wetting._params import WettingParams
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State
    from src.pipeline.state.state import WettingState


# ── Part A: Helper to build operators from live wetting state ──


def with_default_wetting(op: DifferentialOperator, wetting: WettingParams) -> DifferentialOperator:
    """Rebind a wetting-aware operator's default parameters to *wetting*.

    The hysteresis step evaluates the whole multiphase pipeline under live or
    candidate parameters; binding them here lets that pipeline keep calling its
    operators as ``op(grid)``. An explicit *wetting* argument still wins.

    Args:
        op: A wetting-aware :class:`~src.operators.protocols.DifferentialOperator`.
        wetting: Parameters to apply when the result is called without any.

    Returns:
        ``bound(grid, wetting=None)`` delegating to *op*.
    """
    default = wetting

    def bound(grid: Array, wetting: WettingParams | None = None) -> Array:
        return op(grid, default if wetting is None else wetting)

    return bound


def _make_wetting_ops(
    setup: SimulationSetup, wetting: WettingState | WettingParams
) -> tuple[DifferentialOperator, DifferentialOperator]:
    """Bind the density operators to the given wetting parameters.

    The wetting-aware ``setup.gradient_density`` / ``setup.laplacian_density``
    accept live :class:`~src.operators.wetting._params.WettingParams`; this
    rebinds their default so the multiphase pipeline can call them as
    ``op(grid)``. A plain (non-wetting) operator raises :class:`TypeError` when
    first called with those parameters.

    Args:
        setup: :class:`~src.pipeline.setup.SimulationSetup`.
        wetting: Current :class:`~src.pipeline.state.state.WettingState`, or
            candidate :class:`~src.operators.wetting._params.WettingParams`.

    Returns:
        Tuple of (gradient, laplacian) operators with *wetting* bound.
    """
    from src.operators.wetting._params import WettingParams

    if setup.gradient_density is None:
        msg = "gradient_density is required for wetting operators"
        raise TypeError(msg)
    if setup.laplacian_density is None:
        msg = "laplacian_density is required for wetting operators"
        raise TypeError(msg)

    params = WettingParams(
        d_rho_left=wetting.d_rho_left,
        d_rho_right=wetting.d_rho_right,
        phi_left=wetting.phi_left,
        phi_right=wetting.phi_right,
    )
    return with_default_wetting(setup.gradient_density, params), with_default_wetting(setup.laplacian_density, params)


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
    Multiple steps can be taken per trial to strengthen the optimiser response.

    Args:
        setup: :class:`~src.pipeline.setup.SimulationSetup`.
        f_t: Current populations.
        force_ext: External force field.
        params: Candidate :class:`~src.operators.wetting._params.WettingParams`.

    Returns:
        Tuple (f_out, rho_out) where f_out is post-BC populations and
        rho_out is the post-step density field.
    """
    if setup.config.hysteresis_config is None:
        msg = "hysteresis_config is required for trial step"
        raise TypeError(msg)

    grad, lap = _make_wetting_ops(setup, params)

    num_steps: int = setup.config.hysteresis_config["trial_steps"]

    def body_fn(carry_f: jnp.ndarray, _) -> tuple[Array, Array]:  # noqa: ANN001
        f_next, rho_next, _u, _force_tot, _pressure = _multiphase_pipeline(setup, carry_f, force_ext, grad, lap)
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
    The wetting parameters are optimised every timestep via the hysteresis
    optimiser using wetting parametric operators and trial steps.

    The implementation:
    1. Compute external forces
    2. Build operators from live wetting parameters
    3. Run multiphase physics kernel
    4. Optimise wetting parameters via hysteresis
    5. Update extra state (plugins)
    6. Return updated state

    Args:
        setup: Closed-over :class:`~src.pipeline.setup.SimulationSetup`.
               setup.gradient_density and setup.laplacian_density are
               wetting-aware and accept live wetting parameters.
        state: Current :class:`~src.pipeline.state.state.State`.
               state.wetting is a :class:`~src.pipeline.state.state.WettingState`
               with parameters to optimise.

    Returns:
        Updated :class:`~src.pipeline.state.state.State` after one time step
        with optimised wetting parameters.
    """
    # 1. Compute external forces
    force_ext, state = compute_total_force_ext(setup, state, setup.forces)

    # 2. Build operators from live wetting parameters
    if state.wetting is None:
        msg = "state.wetting is required for hysteresis step"
        raise TypeError(msg)
    grad, lap = _make_wetting_ops(setup, state.wetting)

    # 3. Run multiphase physics kernel
    f_out, rho, u, force_tot, pressure = _multiphase_pipeline(setup, state.f, force_ext, grad, lap)

    # 4. Create new state (wetting updated by plugin via trial_step_fn in context)
    new_state = state._replace(
        f=f_out, rho=rho, u=u, force=force_tot, force_ext=force_ext, pressure=pressure, t=state.t + 1
    )

    # 5. Update extra state — trial_step_fn forwarded so wetting plugin can run optimiser
    _force_ext_arr = force_ext if force_ext is not None else jnp.zeros(1)
    return update_extra_state(
        setup,
        state,
        new_state,
        force_ext=force_ext,
        trial_step_fn=partial(_trial_step, setup, f_out, _force_ext_arr),
    )
