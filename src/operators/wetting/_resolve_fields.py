"""Resolve per-side wetting scalars from a wetting_params dict.

Supports scalar and per-chemical-step layouts.
"""

from __future__ import annotations

from typing import Any

from registry import wetting_operator


@wetting_operator(name="resolve_wetting_fields")
def resolve_wetting_fields(
    wetting_params: dict[str, Any],
    chemical_step: int | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Extract per-side wetting scalars from a *wetting_params* dict.

    Supports two layouts:

    * **Scalar** — keys ``phi_l``, ``phi_r``, ``d_rho_l``, ``d_rho_r``.
    * **Array with chemical step** — keys ``phi``, ``d_rho`` (each a
      two-element sequence), indexed by *chemical_step*.

    Returns:
        ``(phi_l, phi_r, d_rho_l, d_rho_r)``
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
