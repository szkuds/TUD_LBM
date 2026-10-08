"""Wetting-aware gradient — addon layer on the base gradient.

Resolved through :func:`~src.operators.differential.build_wetting_gradient_fn`.

Imports the base ``grad_core`` from ``_gradient`` and wetting utilities
from ``operators.wetting``. The base ``_gradient`` module has zero
knowledge of wetting.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators.differential._gradient import grad_core_2d
from src.operators.differential._pad_utils import _apply_stencil_padding
from src.operators.differential._pad_utils import to_2d

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
    import jax.numpy as jnp
    from src.config.chemical_step import ChemicalStepWall
    from src.operators.protocols import DifferentialOperator
    from src.operators.wetting._params import WettingParams


def build_wetting_gradient(
    w: jnp.ndarray,
    c: jnp.ndarray,
    pad_mode: Sequence[str],
    bc_config: dict[str, Any] | None = None,
    *,
    rho_l: float,
    rho_v: float,
    default: WettingParams,
    step: ChemicalStepWall | None = None,
) -> DifferentialOperator:
    """Return a wetting-corrected gradient closure.

    Closes over static config (w, c, pad_mode, bc_config, rho_l, rho_v) and
    the *default* wetting parameters. The returned operator takes the grid and,
    optionally, live wetting parameters, returning shape ``(nx, ny, nz, 1, 2)``.

    Args:
        w:         Lattice weights ``(1, 1, 1, q, 1)``.
        c:         Lattice velocities ``(1, 1, 1, q, 2)``.
        pad_mode:  ``(top, bottom, right, left)`` padding modes.
        bc_config: Boundary-condition edge map, e.g.
                   ``{"bottom": "wetting", "top": "bounce-back"}``.
        rho_l:     Liquid density (baked into closure at build time).
        rho_v:     Vapour density (baked into closure at build time).
        default:   Wetting parameters applied when the operator is called
                   without *wetting*.
        step:      The chemical-step wall, or ``None``; splits the wall
                   modification by surface.

    Returns:
        ``grad(grid, wetting=None) → (nx, ny, nz, 1, 2)``
    """
    from src.operators.wetting._applicator import build_wetting_applicator

    _pad_mode = tuple(pad_mode)
    _apply_wetting = build_wetting_applicator(rho_l, rho_v, bc_config, step)

    def _grad(grid: jnp.ndarray, wetting: WettingParams | None = None) -> jnp.ndarray:
        params = default if wetting is None else wetting
        grid_padded = _apply_stencil_padding(to_2d(grid), _pad_mode)
        grid_padded = _apply_wetting(
            grid_padded, params.phi_left, params.phi_right, params.d_rho_left, params.d_rho_right
        )

        # Pass FULL padded array to grad_core.
        return grad_core_2d(grid_padded, w, c)

    return _grad
