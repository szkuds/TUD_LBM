"""Resolve per-side wetting scalars from a wetting_params dict.

Supports scalar and per-chemical-step layouts.

NOTE: This function is no longer called from the wetting differential-operator
shim, which now inlines the chemical-step split directly.  It is kept here in
case other callers exist, but can be removed once confirmed unused.
"""

from __future__ import annotations
from typing import Any
import jax.numpy as jnp
from tud_lbm.registry import wetting_operator


@wetting_operator(name="resolve_wetting_fields")
def resolve_wetting_fields(
    wetting_params: dict[str, Any],
    *,
    nx: int | None = None,
    step_x: float | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Extract per-side wetting scalars from a *wetting_params* dict.

    Returns:
        ``(phi_l, phi_r, d_rho_l, d_rho_r)`` — each a scalar or a 1-D array
        of length ``nx`` when per-region keys and spatial arguments are
        supplied.  Supported input layouts:

        - **Legacy scalar**: ``{"phi_l", "phi_r", "d_rho_l", "d_rho_r"}``
        - **Per-region**: ``{"phi_left_pre", "phi_left_post", ...}`` (all eight
          keys).  When *nx* and *step_x* are also provided the result is a
          per-column array split at ``step_x``; otherwise the ``_pre`` scalar
          is returned for each side.
    """
    # 1) Explicit per-side scalar layout
    if all(k in wetting_params for k in ("phi_l", "phi_r", "d_rho_l", "d_rho_r")):
        return (
            wetting_params["phi_l"],
            wetting_params["phi_r"],
            wetting_params["d_rho_l"],
            wetting_params["d_rho_r"],
        )

    # 2) Per-region explicit scalars
    pre_post_keys = (
        "phi_left_pre",
        "phi_left_post",
        "d_rho_left_pre",
        "d_rho_left_post",
        "phi_right_pre",
        "phi_right_post",
        "d_rho_right_pre",
        "d_rho_right_post",
    )
    if all(k in wetting_params for k in pre_post_keys):
        if nx is not None and step_x is not None:
            # Build per-column arrays split at step_x.
            # Values are kept as-is (no float() cast) so JAX tracer lineage is
            # preserved when this function is called inside an autodiff context.
            step_ix = int(step_x)
            cols = jnp.arange(nx)
            phi_l = jnp.where(cols < step_ix, wetting_params["phi_left_pre"], wetting_params["phi_left_post"])
            phi_r = jnp.where(cols < step_ix, wetting_params["phi_right_pre"], wetting_params["phi_right_post"])
            d_rho_l = jnp.where(cols < step_ix, wetting_params["d_rho_left_pre"], wetting_params["d_rho_left_post"])
            d_rho_r = jnp.where(cols < step_ix, wetting_params["d_rho_right_pre"], wetting_params["d_rho_right_post"])
            return phi_l, phi_r, d_rho_l, d_rho_r
        # Without spatial info, return the pre-step scalar for each side.
        return (
            wetting_params["phi_left_pre"],
            wetting_params["phi_right_pre"],
            wetting_params["d_rho_left_pre"],
            wetting_params["d_rho_right_pre"],
        )

    # Fallback: alternative key names
    alt = {
        "phi_l": ("phi_l", "phi_left"),
        "phi_r": ("phi_r", "phi_right"),
        "d_rho_l": ("d_rho_l", "d_rho_left"),
        "d_rho_r": ("d_rho_r", "d_rho_right"),
    }

    def _pick(*keys: str) -> object:
        for k in keys:
            if k in wetting_params:
                return wetting_params[k]
        msg = "Missing wetting parameter keys"
        raise KeyError(msg)

    return (_pick(*alt["phi_l"]), _pick(*alt["phi_r"]), _pick(*alt["d_rho_l"]), _pick(*alt["d_rho_r"]))
