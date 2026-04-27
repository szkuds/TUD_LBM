"""Interface-localised wetting density modification.

Provides :func:`_apply_wetting_modification`, which adjusts ghost-cell
densities at the liquid–vapour interface so that the LBM-stencil
gradient "sees" the desired wetting boundary condition.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

# Density thresholds for detecting the interface region
_HIGH_FRAC = 0.95
_LOW_FRAC = 0.05


def _apply_wetting_modification(
    edge_slice: jnp.ndarray,
    rho_l: float,
    rho_v: float,
    phi_l: Any,
    phi_r: Any,
    d_rho_l: Any,
    d_rho_r: Any,
    width: int,
) -> jnp.ndarray:
    """Apply wetting density modification at the liquid-vapour interface.

    Only modifies ghost-cell values that lie within the interface region
    (between the density thresholds). The interface is split into left
    and right contact-line regions, each receiving its own phi/d_rho.

    Args:
        edge_slice: Ghost-row densities, shape ``(n,)``.
        rho_l, rho_v: Liquid and vapour densities.
        phi_l, phi_r: Wetting potentials for left and right contact lines.
        d_rho_l, d_rho_r: Density offsets for left and right contact lines.
        width: Interface width for splitting left/right regions.

    Returns:
        Modified edge slice.
    """
    rho_upper = _HIGH_FRAC * rho_l + _LOW_FRAC * rho_v
    rho_lower = _LOW_FRAC * rho_l + _HIGH_FRAC * rho_v

    # Interface mask: points between the density thresholds
    in_interface = (edge_slice < rho_upper) & (edge_slice > rho_lower)

    # Find transition indices to split left/right contact-line regions
    mask_int = jnp.array(edge_slice < rho_upper, dtype=jnp.int32)
    diff = jnp.diff(mask_int)

    # Left transition: where density drops below upper threshold (diff == -1)
    left_idx = jnp.where(diff == -1, size=1, fill_value=0)[0] + width
    # Right transition: where density rises above upper threshold (diff == 1)
    right_idx = jnp.where(diff == 1, size=1, fill_value=0)[0] - width

    indices = jnp.arange(edge_slice.shape[0])
    is_left_region = in_interface & (indices < right_idx[0])
    is_right_region = in_interface & (indices > left_idx[0])

    # Wetting modification: phi * rho - d_rho, clamped to density bounds
    modified_left = jnp.clip(phi_l * edge_slice - d_rho_l, rho_lower, rho_upper)
    modified_right = jnp.clip(phi_r * edge_slice - d_rho_r, rho_lower, rho_upper)

    result = jnp.where(is_right_region, modified_right, edge_slice)
    return jnp.where(is_left_region, modified_left, result)
