"""Resolve per-side wetting scalars from a wetting_params dict.

Supports scalar and per-chemical-step layouts.
"""

from __future__ import annotations
from typing import Any
import jax.numpy as jnp
from tud_lbm.registry import wetting_operator


@wetting_operator(name="resolve_wetting_fields")
def resolve_wetting_fields(
    wetting_params: dict[str, Any],
    *,
    chemical_step: int | None = None,
    nx: int | None = None,
    step_x: float | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Extract per-side wetting scalars from a *wetting_params* dict.

    Returns:
        ``(phi_l, phi_r, d_rho_l, d_rho_r)`` — each may be a scalar or
        a 1D array of length ``nx`` when chemical-step / per-region keys
        are supplied. Supported input layouts:

        - Legacy scalar: ``{"phi_l": .., "phi_r": .., "d_rho_l": .., "d_rho_r": ..}``
        - Array layout: ``{"phi": [pre, post], "d_rho": [pre, post]}`` with optional ``chemical_step`` index
        - New per-region scalars: explicit keys like ``phi_left_pre``, ``phi_left_post``, etc.
    """
    # 1) Explicit per-side scalar layout
    if all(k in wetting_params for k in ("phi_l", "phi_r", "d_rho_l", "d_rho_r")):
        return (
            wetting_params["phi_l"],
            wetting_params["phi_r"],
            wetting_params["d_rho_l"],
            wetting_params["d_rho_r"],
        )

    # 2) Array layout for chemical_step: {'phi': [pre, post], 'd_rho': [pre, post]}
    if "phi" in wetting_params and "d_rho" in wetting_params:
        phi = wetting_params["phi"]
        d_rho = wetting_params["d_rho"]
        # If chemical_step supplied, choose mapping accordingly (step swaps sides)
        if chemical_step is not None:
            idx = int(chemical_step) % len(phi)
            phi_l = phi[idx]
            phi_r = phi[1 - idx] if len(phi) > 1 else phi[0]
            d_rho_l = d_rho[idx]
            d_rho_r = d_rho[1 - idx] if len(d_rho) > 1 else d_rho[0]
        else:
            # Default: phi[0] -> left, phi[1] -> right
            phi_l = phi[0]
            phi_r = phi[1] if len(phi) > 1 else phi[0]
            d_rho_l = d_rho[0]
            d_rho_r = d_rho[1] if len(d_rho) > 1 else d_rho[0]
        return (phi_l, phi_r, d_rho_l, d_rho_r)

    # 3) New per-region explicit scalars
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
        # If nx and step_x are provided, build per-x arrays split at step_x
        if nx is not None and step_x is not None:
            step_ix = int(step_x)

            phi_left_pre = float(wetting_params["phi_left_pre"])
            phi_left_post = float(wetting_params["phi_left_post"])
            d_rho_left_pre = float(wetting_params["d_rho_left_pre"])
            d_rho_left_post = float(wetting_params["d_rho_left_post"])

            # For left edge, columns < step_ix use 'pre', >= step_ix use 'post'
            cols = jnp.arange(nx)
            left_phi_col = jnp.where(cols < step_ix, phi_left_pre, phi_left_post)
            left_drho_col = jnp.where(cols < step_ix, d_rho_left_pre, d_rho_left_post)

            phi_right_pre = float(wetting_params["phi_right_pre"])
            phi_right_post = float(wetting_params["phi_right_post"])
            d_rho_right_pre = float(wetting_params["d_rho_right_pre"])
            d_rho_right_post = float(wetting_params["d_rho_right_post"])
            right_phi_col = jnp.where(cols < step_ix, phi_right_pre, phi_right_post)
            right_drho_col = jnp.where(cols < step_ix, d_rho_right_pre, d_rho_right_post)

            return (left_phi_col, right_phi_col, left_drho_col, right_drho_col)
        # Without spatial info, return scalar pre/post pairs (defaults to pre)
        return (
            wetting_params["phi_left_pre"],
            wetting_params["phi_right_pre"],
            wetting_params["d_rho_left_pre"],
            wetting_params["d_rho_right_pre"],
        )

    # Fallback: try to extract legacy keys with alternative names
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
