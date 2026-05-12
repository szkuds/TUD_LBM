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
    phi_l, phi_r, d_rho_l, d_rho_r = _resolve_wetting_fields(
        {
            "phi_l": wetting_state.phi_left,
            "phi_r": wetting_state.phi_right,
            "d_rho_l": wetting_state.d_rho_left,
            "d_rho_r": wetting_state.d_rho_right,
        },
    )

    def wetting_gradient_density(grid: jnp.ndarray) -> jnp.ndarray:
        """Gradient shim that injects live wetting parameters."""
        return setup.gradient_density(grid, phi_l, phi_r, d_rho_l, d_rho_r)

    def wetting_laplacian_density(grid: jnp.ndarray) -> jnp.ndarray:
        """Laplacian shim that injects live wetting parameters."""
        return setup.laplacian_density(grid, phi_l, phi_r, d_rho_l, d_rho_r)

    return wetting_gradient_density, wetting_laplacian_density
