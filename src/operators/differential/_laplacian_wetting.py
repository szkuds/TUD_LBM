"""Wetting-aware Laplacian — addon layer on the base Laplacian.

Resolved through :func:`~src.operators.differential.build_wetting_laplacian_fn`.

Imports the Laplacian stencil logic and wetting utilities.
The base ``_laplacian`` module has zero knowledge of wetting.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators.differential._laplacian import lap_core_2d
from src.operators.differential._pad_utils import _apply_stencil_padding
from src.operators.differential._pad_utils import to_2d

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
    import jax.numpy as jnp
    from src.config.chemical_step import ChemicalStepWall
    from src.operators.protocols import DifferentialOperator
    from src.operators.wetting._params import WettingParams


def build_wetting_laplacian(
    w: jnp.ndarray,
    pad_mode: Sequence[str],
    bc_config: dict[str, Any] | None = None,
    *,
    rho_l: float,
    rho_v: float,
    default: WettingParams,
    step: ChemicalStepWall | None = None,
) -> DifferentialOperator:
    """Return a wetting-corrected Laplacian closure.

    Closes over static config (w, pad_mode, bc_config, rho_l, rho_v) and the
    *default* wetting parameters. The returned operator takes the grid and,
    optionally, live wetting parameters, returning shape ``(nx, ny, nz, 1, 1)``.

    Args:
        w:         Lattice weights ``(1, 1, 1, q, 1)``.
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
        ``lap(grid, wetting=None) → (nx, ny, nz, 1, 1)``
    """
    from src.operators.wetting._applicator import build_wetting_applicator

    _pad_mode = tuple(pad_mode)
    _apply_wetting = build_wetting_applicator(rho_l, rho_v, bc_config, step)

    def _lap(grid: jnp.ndarray, wetting: WettingParams | None = None) -> jnp.ndarray:
        params = default if wetting is None else wetting
        grid_padded = _apply_stencil_padding(to_2d(grid), _pad_mode)

        # Wetting ghost-cell correction on the padded array
        # (rho_l, rho_v, width baked into the applicator)
        grid_padded = _apply_wetting(
            grid_padded, params.phi_left, params.phi_right, params.d_rho_left, params.d_rho_right
        )

        # Pass FULL padded array to lap_core.
        return lap_core_2d(grid_padded, w)

    return _lap
