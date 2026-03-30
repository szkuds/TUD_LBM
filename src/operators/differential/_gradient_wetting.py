"""Wetting-aware gradient — addon layer on the base gradient.

Registered as ``("differential", "gradient_wetting")``.
Auto-discovered alongside the base operators by ``auto_load_operators``.

Imports the base ``grad_core`` from ``_gradient`` and wetting utilities
from ``operators.wetting``. The base ``_gradient`` module has zero
knowledge of wetting.
"""

from __future__ import annotations
import jax.numpy as jnp
from registry import register_operator
from operators.differential._pad_utils import apply_stencil_padding, to_2d
from operators.differential._gradient import grad_core
from operators.wetting.wetting_util import apply_wetting_to_all_edges


@register_operator("differential", name="gradient_wetting")
def build_wetting_gradient(
    w: jnp.ndarray,
    c: jnp.ndarray,
    pad_mode: tuple[str, ...] | list[str],
    bc_config: dict | None = None,
):
    """Return a wetting-corrected gradient closure.

    Closes over static config (w, c, pad_mode, bc_config).
    The returned callable accepts the grid plus dynamic wetting
    parameters and returns shape ``(nx, ny, 1, 2)``.

    Args:
        w:         Lattice weights ``(q,)``.
        c:         Lattice velocities ``(2, q)``.
        pad_mode:  ``(right_y, left_y, bottom_x, top_x)``.
        bc_config: Boundary-condition edge map, e.g.
                   ``{"bottom": "wetting", "top": "bounce-back"}``.

    Returns:
        ``grad(grid, phi_l, phi_r, d_rho_l, d_rho_r, rho_l, rho_v, width) → (nx,ny,1,2)``
    """
    _pad_mode = tuple(pad_mode)

    def _grad(
        grid: jnp.ndarray,
        phi_l: jnp.ndarray,
        phi_r: jnp.ndarray,
        d_rho_l: jnp.ndarray,
        d_rho_r: jnp.ndarray,
        rho_l: jnp.ndarray,
        rho_v: jnp.ndarray,
        width: jnp.ndarray,
    ) -> jnp.ndarray:
        gp = apply_stencil_padding(to_2d(grid), _pad_mode)

        # Wetting ghost-cell correction on the padded array
        gp = apply_wetting_to_all_edges(
            gp, rho_l, rho_v, phi_l, phi_r,
            d_rho_l, d_rho_r, width, bc_config,
        )

        # Pass FULL padded array to grad_core.
        # BUG FIX: old code stripped ghost cells before calling
        # _grad_core, which broke the [2:, 1:-1] neighbour slicing.
        return grad_core(gp, w, c)

    return _grad

