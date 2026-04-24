"""Wetting-aware gradient — addon layer on the base gradient.

Registered as ``("differential", "gradient_wetting")``.
Auto-discovered alongside the base operators by ``auto_load_operators``.

Imports the base ``grad_core`` from ``_gradient`` and wetting utilities
from ``operators.wetting``. The base ``_gradient`` module has zero
knowledge of wetting.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from tud_lbm.operators.differential._gradient import grad_core
from tud_lbm.operators.differential._pad_utils import _apply_stencil_padding
from tud_lbm.operators.differential._pad_utils import to_2d
from tud_lbm.operators.wetting import build_wetting_fn
from tud_lbm.registry import register_operator

if TYPE_CHECKING:
    import jax.numpy as jnp


@register_operator("differential", name="gradient_wetting")
def build_wetting_gradient(
    w: jnp.ndarray,
    c: jnp.ndarray,
    pad_mode: tuple[str, ...] | list[str],
    bc_config: dict | None = None,
    rho_l: float | None = None,
    rho_v: float | None = None,
    width: int | None = None,
) -> callable:
    """Return a wetting-corrected gradient closure.

    Closes over static config (w, c, pad_mode, bc_config, rho_l, rho_v, width).
    The returned callable accepts only the grid and dynamic wetting parameters,
    returning shape ``(nx, ny, nz, 1, 2)``.

    Args:
        w:         Lattice weights ``(1, 1, 1, q, 1)``.
        c:         Lattice velocities ``(1, 1, 1, q, 2)``.
        pad_mode:  ``(right_y, left_y, bottom_x, top_x)``.
        bc_config: Boundary-condition edge map, e.g.
                   ``{"bottom": "wetting", "top": "bounce-back"}``.
        rho_l:     Liquid density (baked into closure at build time).
        rho_v:     Vapour density (baked into closure at build time).
        width:     Interface width in lattice units (baked into closure at build time).

    Returns:
        ``grad(grid, phi_l, phi_r, d_rho_l, d_rho_r) → (nx,ny,nz,1,2)``
    """
    _pad_mode = tuple(pad_mode)
    _build_wetting_applicator = build_wetting_fn("applicator")
    _apply_wetting = _build_wetting_applicator(rho_l, rho_v, width, bc_config)

    def _grad(
        grid: jnp.ndarray,
        phi_l: jnp.ndarray,
        phi_r: jnp.ndarray,
        d_rho_l: jnp.ndarray,
        d_rho_r: jnp.ndarray,
    ) -> jnp.ndarray:
        gp = _apply_stencil_padding(to_2d(grid), _pad_mode)
        gp = _apply_wetting(gp, phi_l, phi_r, d_rho_l, d_rho_r)

        # Pass FULL padded array to grad_core.
        return grad_core(gp, w, c)

    return _grad
