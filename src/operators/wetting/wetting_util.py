"""Wetting ghost-cell utilities — pure helper functions.

These functions support the wetting boundary treatment used by
:func:`~operators.differential.gradient.make_wetting_gradient`.

``resolve_wetting_fields`` extracts per-side wetting scalars from a
``wetting_params`` dict (which may contain either plain scalars or
per-side arrays depending on whether a chemical step is used).

``build_wetting_applicator`` creates a closure that applies wetting ghost-cell
corrections to a padded density field. The returned closure writes ghost-cell
rows so that the LBM-stencil gradient "sees" the wetting boundary condition at
whichever edges are marked ``"wetting"`` in ``bc_config``. Static parameters
(liquid/vapour densities, interface width, boundary configuration) are baked in
at build time, so the closure only requires dynamic wetting parameters per call.

All operations are pure Python / NumPy pre-computations that run *before*
``@jit``, so there are no tracing constraints.
"""

from __future__ import annotations
from typing import Any
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def resolve_wetting_fields(
    wetting_params: dict[str, Any],
    chemical_step: int | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Extract per-side wetting scalars from a *wetting_params* dict.

    Supports two layouts:

    * **Scalar** — ``{"phi_l": <float>, "phi_r": <float>,
      "d_rho_l": <float>, "d_rho_r": <float>, ...}``
    * **Array with chemical step** — ``{"phi": <array>, "drho": <array>,
      ...}`` where index 0 is the left half and index 1 is the right half,
      selected by *chemical_step*.

    Args:
        wetting_params: Dict containing at minimum ``phi_l``, ``phi_r``,
            ``d_rho_l``, ``d_rho_r`` (scalar layout) **or** ``phi``,
            ``drho`` (array layout) keys.
        chemical_step: Optional step index (0 or 1) for chemical-step
            simulations.  When ``None`` the scalar layout is assumed.

    Returns:
        ``(phi_l, phi_r, d_rho_l, d_rho_r)`` — scalars or 0-d arrays.
    """
    if chemical_step is not None:
        phi = wetting_params["phi"]
        d_rho = wetting_params["d_rho"]
        phi_l = phi[0] if chemical_step == 0 else phi[1]
        phi_r = phi[1] if chemical_step == 0 else phi[0]
        d_rho_l = d_rho[0] if chemical_step == 0 else d_rho[1]
        d_rho_r = d_rho[1] if chemical_step == 0 else d_rho[0]
    else:
        phi_l = wetting_params["phi_l"]
        phi_r = wetting_params["phi_r"]
        d_rho_l = wetting_params["d_rho_l"]
        d_rho_r = wetting_params["d_rho_r"]

    return phi_l, phi_r, d_rho_l, d_rho_r


def build_wetting_applicator(
    rho_l: float,
    rho_v: float,
    width: int,
    bc_config: dict[str, Any] | None = None,
):
    """Build a wetting ghost-cell applicator closure with baked-in static parameters.

    Creates a closure that applies wetting ghost-cell corrections to a padded density
    array. Static parameters (rho_l, rho_v, width, bc_config) are baked in at build time,
    so the returned closure only requires dynamic wetting parameters.

    Args:
        rho_l: Liquid density.
        rho_v: Vapour density.
        width: Interface width in lattice units.
        bc_config: Boundary-condition config dict, e.g.
            ``{"bottom": "wetting", "top": "bounce-back", ...}``.
            ``None`` defaults to bottom-only injection.

    Returns:
        A closure ``(gp, phi_l, phi_r, d_rho_l, d_rho_r) -> gp`` that applies
        wetting ghost-cell corrections to the padded density field.
    """
    _rho_l = float(rho_l)
    _rho_v = float(rho_v)
    _width = int(width)
    _bc_config = bc_config

    def apply(
        gp: jnp.ndarray,
        phi_l: Any,
        phi_r: Any,
        d_rho_l: Any,
        d_rho_r: Any,
    ) -> jnp.ndarray:
        """Apply wetting ghost-cell correction to a padded density array.

        The padded array *gp* has shape ``(nx + 2, ny + 2)`` (one ghost cell on
        each side). This function overwrites the ghost rows for each edge
        marked ``"wetting"`` in *bc_config* with a wetting density value
        derived from the liquid/vapour densities and the per-side wetting
        parameters.

        The ghost-cell value at column *i* is:

        .. code-block:: text

            rho_ghost[i] = phi * rho_l + (1 - phi) * rho_v + d_rho * profile[i]

        where *phi* is ``phi_l`` for the left half of the domain and ``phi_r``
        for the right half, *d_rho* is the corresponding density offset, and
        *profile* is a smooth step function of width *width*.

        Args:
            gp: Padded density field, shape ``(nx + 2, ny + 2)``.
            phi_l: Wetting potential (left side).
            phi_r: Wetting potential (right side).
            d_rho_l: Density offset (left side).
            d_rho_r: Density offset (right side).

        Returns:
            Updated padded field with ghost-cell rows set.
        """
        nx2, _ny2 = gp.shape  # nx+2, ny+2
        nx = nx2 - 2
        half = nx // 2

        # Build per-column phi and d_rho arrays
        phi_col = jnp.where(
            jnp.arange(nx) < half,
            jnp.full(nx, phi_l),
            jnp.full(nx, phi_r),
        )
        d_rho_col = jnp.where(
            jnp.arange(nx) < half,
            jnp.full(nx, d_rho_l),
            jnp.full(nx, d_rho_r),
        )

        # Smooth tanh profile along x (centred at the domain mid-point)
        xs = jnp.arange(nx, dtype=jnp.float32)
        profile = 0.5 * (1.0 + jnp.tanh((xs - nx / 2.0) / _width))

        # Ghost-cell density
        rho_ghost = phi_col * _rho_l + (1.0 - phi_col) * _rho_v + d_rho_col * profile

        # Determine which edges to inject wetting ghost cells for.
        if _bc_config is not None:
            edges_to_process = [e for e in ("bottom", "top") if _bc_config.get(e) == "wetting"]
        else:
            edges_to_process = ["bottom"]  # legacy default

        # Write into the ghost row(s) of the padded array.
        # The interior columns in gp are indices 1 .. nx (padded by 1 on each side).
        for edge in edges_to_process:
            if edge == "bottom":
                gp = gp.at[1:-1, 0].set(rho_ghost)
            elif edge == "top":
                gp = gp.at[1:-1, -1].set(rho_ghost)

        return gp

    return apply
