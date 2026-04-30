"""Wetting differential-operator shims for live parameter injection.

Builds grid-only gradient and laplacian closures that inject live
wetting parameters (phi, d_rho) from the current :class:`WettingState`.

Internal helper — not registered in the operator registry.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp

if TYPE_CHECKING:
    from tud_lbm.pipeline.setup import SimulationSetup
    from tud_lbm.pipeline.state import WettingState


def _make_wetting_differential_ops(setup: SimulationSetup, wetting_state: WettingState) -> tuple:
    """Build grid-only gradient and laplacian shims from live wetting params.

    Extracts ``(phi_l, phi_r, d_rho_l, d_rho_r)`` from *wetting_state*,
    closes over them together with ``setup.gradient_density`` /
    ``setup.laplacian_density``, and returns two callables each with the
    signature ``grid -> result`` expected by
    :func:`~tud_lbm.operators.macroscopic.compute_macroscopic_multiphase`.

    When a chemical step is configured the four quantities become per-column
    1-D arrays of shape ``(nx,)`` split at ``step_ix``.  Columns left of the
    step carry the ``_pre`` wetting values; columns right carry ``_post``.
    Using ``jnp.where`` (not Python conditionals or ``float()`` casts) keeps
    the JAX tracer graph intact so ``jax.value_and_grad`` can differentiate
    through the full chain during hysteresis optimisation.

    Args:
        setup: Closed-over :class:`~tud_lbm.pipeline.setup.SimulationSetup`.
        wetting_state: Current :class:`~tud_lbm.pipeline.state.state.WettingState`.

    Returns:
        ``(gradient_density_shim, laplacian_density_shim)`` — two callables,
        each ``(grid) → result``.
    """
    csc = setup.config.chemical_step_config
    if csc is not None:
        # Build per-column wetting arrays split at the chemical step.
        # step_ix is a Python int (static), so `cols < step_ix` is a concrete
        # boolean array — no tracer leak.  The pre/post values are JAX
        # arrays/tracers, so jnp.where keeps both branches in the graph.
        nx = setup.config.grid_shape[0]
        step_ix = int(float(csc["chemical_step_location"]) * nx)
        cols = jnp.arange(nx)
        phi_l = jnp.where(cols < step_ix, wetting_state.phi_left_pre, wetting_state.phi_left_post)
        phi_r = jnp.where(cols < step_ix, wetting_state.phi_right_pre, wetting_state.phi_right_post)
        d_rho_l = jnp.where(cols < step_ix, wetting_state.d_rho_left_pre, wetting_state.d_rho_left_post)
        d_rho_r = jnp.where(cols < step_ix, wetting_state.d_rho_right_pre, wetting_state.d_rho_right_post)
    else:
        phi_l = wetting_state.phi_left
        phi_r = wetting_state.phi_right
        d_rho_l = wetting_state.d_rho_left
        d_rho_r = wetting_state.d_rho_right

    def wetting_gradient_density(grid: jnp.ndarray) -> jnp.ndarray:
        return setup.gradient_density(grid, phi_l, phi_r, d_rho_l, d_rho_r)

    def wetting_laplacian_density(grid: jnp.ndarray) -> jnp.ndarray:
        return setup.laplacian_density(grid, phi_l, phi_r, d_rho_l, d_rho_r)

    return wetting_gradient_density, wetting_laplacian_density
