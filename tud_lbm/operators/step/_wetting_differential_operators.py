"""Wetting differential-operator shims for live parameter injection.

Builds grid-only gradient and laplacian closures that inject live
wetting parameters (phi, d_rho) from the current :class:`WettingState`.

Internal helper — not registered in the operator registry.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from tud_lbm.operators.wetting import build_wetting_fn

if TYPE_CHECKING:
    import jax.numpy as jnp
    from tud_lbm.pipeline.setup import SimulationSetup
    from tud_lbm.pipeline.state import WettingState


def _make_wetting_differential_ops(setup: SimulationSetup, wetting_state: WettingState) -> tuple:
    """Build grid-only gradient and laplacian shims from live wetting params.

    Extracts (phi_l, phi_r, d_rho_l, d_rho_r) from wetting_state,
    closes over them together with setup.gradient_density / setup.laplacian_density
    (the wetting-aware factory closures with baked static params), and returns
    two callables each with the signature ``grid -> result`` expected by
    :func:`~tud_lbm.operators.macroscopic.compute_macroscopic_multiphase`.

    Args:
        setup: Closed-over :class:`~tud_lbm.pipeline.setup.SimulationSetup`.
        wetting_state: Current :class:`~tud_lbm.pipeline.state.state.WettingState`.

    Returns:
        ``(gradient_density_shim, laplacian_density_shim)`` — two callable wrappers,
        each ``(grid) → result``.
    """
    # Extract live wetting parameters from the state
    _resolve_wetting_fields = build_wetting_fn("resolve_wetting_fields")
    # Build a mapping that includes both legacy scalar keys and the new
    # per-region pre/post keys so resolve_wetting_fields can choose the
    # most specific available layout.
    mapping = {
        "phi_l": wetting_state.phi_left,
        "phi_r": wetting_state.phi_right,
        "d_rho_l": wetting_state.d_rho_left,
        "d_rho_r": wetting_state.d_rho_right,
        "phi_left_pre": getattr(wetting_state, "phi_left_pre", None),
        "phi_left_post": getattr(wetting_state, "phi_left_post", None),
        "d_rho_left_pre": getattr(wetting_state, "d_rho_left_pre", None),
        "d_rho_left_post": getattr(wetting_state, "d_rho_left_post", None),
        "phi_right_pre": getattr(wetting_state, "phi_right_pre", None),
        "phi_right_post": getattr(wetting_state, "phi_right_post", None),
        "d_rho_right_pre": getattr(wetting_state, "d_rho_right_pre", None),
        "d_rho_right_post": getattr(wetting_state, "d_rho_right_post", None),
    }

    # If a chemical step is configured, pass spatial info so resolver builds
    # per-column arrays split at step_x.
    csc = setup.config.chemical_step_config
    if csc is not None:
        nx = setup.config.grid_shape[0]
        step_x = float(csc["chemical_step_location"]) * nx
        phi_l, phi_r, d_rho_l, d_rho_r = _resolve_wetting_fields(mapping, nx=nx, step_x=step_x)
    else:
        phi_l, phi_r, d_rho_l, d_rho_r = _resolve_wetting_fields(mapping)

    def wetting_gradient_density(grid: jnp.ndarray) -> jnp.ndarray:
        """Gradient shim that injects live wetting parameters."""
        return setup.gradient_density(grid, phi_l, phi_r, d_rho_l, d_rho_r)

    def wetting_laplacian_density(grid: jnp.ndarray) -> jnp.ndarray:
        """Laplacian shim that injects live wetting parameters."""
        return setup.laplacian_density(grid, phi_l, phi_r, d_rho_l, d_rho_r)

    return wetting_gradient_density, wetting_laplacian_density
