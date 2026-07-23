"""Wetting ghost-cell applicator builder.

Composes edge configuration and per-edge application into a single
closure that applies wetting ghost-cell corrections to a padded
density array.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
import jax.numpy as jnp
from src.operators.wetting._apply_edge import _apply_wetting_edge
from src.operators.wetting._edge_config import _resolve_wetting_edges
from src.registry import wetting_operator

if TYPE_CHECKING:
    from collections.abc import Callable


@wetting_operator(name="applicator")
def build_wetting_applicator(
    rho_l: float,
    rho_v: float,
    width: int,
    bc_config: dict[str, Any] | None = None,
) -> Callable[..., jnp.ndarray]:
    """Build a wetting ghost-cell applicator with baked-in static parameters.

    Args:
        rho_l: Liquid density.
        rho_v: Vapour density.
        width: Interface width in lattice units.
        bc_config: Boundary-condition config dict, e.g.
            ``{"bottom": "wetting", "top": "bounce-back", ...}``.
            ``None`` defaults to bottom-only wetting.

    Returns:
        ``(gp, phi_l, phi_r, d_rho_l, d_rho_r) → gp``
    """
    _rho_l = jnp.array(float(rho_l))
    _rho_v = jnp.array(float(rho_v))
    _width = int(width)
    _bc_config = bc_config if bc_config is not None else {}

    # Pre-compute which edges need wetting and their perpendicular BC types.
    _edges = _resolve_wetting_edges(_bc_config)

    def apply(
        gp: jnp.ndarray,
        phi_l: jnp.ndarray,
        phi_r: jnp.ndarray,
        d_rho_l: jnp.ndarray,
        d_rho_r: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply wetting ghost-cell corrections to a padded density array.

        Args:
            gp: Padded density field, shape ``(nx + 2, ny + 2)``.
            phi_l: Wetting potential (left/bottom side of interface).
            phi_r: Wetting potential (right/top side of interface).
            d_rho_l: Density offset (left/bottom side).
            d_rho_r: Density offset (right/top side).

        Returns:
            Updated padded field with ghost-cell rows/columns set.
        """
        for edge, perp_start_periodic, perp_end_periodic in _edges:
            gp = _apply_wetting_edge(
                gp,
                edge,
                perp_start_periodic,
                perp_end_periodic,
                _rho_l,
                _rho_v,
                phi_l,
                phi_r,
                d_rho_l,
                d_rho_r,
            )
        return gp

    return apply
