"""Wetting-aware Laplacian — addon layer on the base Laplacian.

Registered as ``("differential", "laplacian_wetting")``.
Auto-discovered alongside the base operators by ``auto_load_operators``.

Imports the Laplacian stencil logic and wetting utilities.
The base ``_laplacian`` module has zero knowledge of wetting.
"""

from __future__ import annotations
import jax.numpy as jnp
from tud_lbm.operators.differential._laplacian import lap_core
from tud_lbm.operators.differential._pad_utils import _apply_stencil_padding
from tud_lbm.operators.differential._pad_utils import to_2d
from tud_lbm.operators.wetting import build_wetting_fn
from registry import register_operator


@register_operator("differential", name="laplacian_wetting")
def build_wetting_laplacian(
    w: jnp.ndarray,
    pad_mode: tuple[str, ...] | list[str],
    bc_config: dict | None = None,
    rho_l: float | None = None,
    rho_v: float | None = None,
    width: int | None = None,
):
    """Return a wetting-corrected Laplacian closure.

    Closes over static config (w, pad_mode, bc_config, rho_l, rho_v, width).
    The returned callable accepts only the grid and dynamic wetting parameters,
    returning shape ``(nx, ny, 1, 1)``.

    Args:
        w:         Lattice weights ``(q,)``.
        pad_mode:  ``(right_y, left_y, bottom_x, top_x)``.
        bc_config: Boundary-condition edge map, e.g.
                   ``{"bottom": "wetting", "top": "bounce-back"}``.
        rho_l:     Liquid density (baked into closure at build time).
        rho_v:     Vapour density (baked into closure at build time).
        width:     Interface width in lattice units (baked into closure at build time).

    Returns:
        ``lap(grid, phi_l, phi_r, d_rho_l, d_rho_r) → (nx, ny, 1, 1)``
    """
    _pad_mode = tuple(pad_mode)
    _build_wetting_applicator = build_wetting_fn("applicator")
    _apply_wetting = _build_wetting_applicator(rho_l, rho_v, width, bc_config)

    def _lap(
        grid: jnp.ndarray,
        phi_l: jnp.ndarray,
        phi_r: jnp.ndarray,
        d_rho_l: jnp.ndarray,
        d_rho_r: jnp.ndarray,
    ) -> jnp.ndarray:
        """Wetting-corrected Laplacian of a scalar field.

        Args:
            grid: Scalar field, shape ``(nx, ny, 1, 1)`` or ``(nx, ny)``.
            phi_l: Contact angle (left edge), scalar or 0-d array.
            phi_r: Contact angle (right edge), scalar or 0-d array.
            d_rho_l: Density offset (left edge), scalar or 0-d array.
            d_rho_r: Density offset (right edge), scalar or 0-d array.

        Returns:
            Laplacian field, shape ``(nx, ny, 1, 1)``.
        """
        grid_2d = to_2d(grid)
        gp = _apply_stencil_padding(grid_2d, _pad_mode)

        # Wetting ghost-cell correction on the padded array
        # (rho_l, rho_v, width now baked into the applicator)
        gp = _apply_wetting(gp, phi_l, phi_r, d_rho_l, d_rho_r)

        # Pass FULL padded array to lap_core.
        return lap_core(gp, w)

    return _lap
