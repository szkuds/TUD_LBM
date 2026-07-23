"""Interface-localised wetting density modification.

Provides :func:`_apply_wetting_modification`, which adjusts ghost-cell
densities at the liquid-vapour interface so that the LBM-stencil
gradient "sees" the desired wetting boundary condition.
"""

from __future__ import annotations
import jax.numpy as jnp

# Density thresholds for detecting the interface region
_HIGH_FRAC = 0.95
_LOW_FRAC = 0.05


def _apply_wetting_modification(
    edge_slice: jnp.ndarray,
    rho_l: jnp.ndarray,
    rho_v: jnp.ndarray,
    phi_l: jnp.ndarray,
    phi_r: jnp.ndarray,
    d_rho_l: jnp.ndarray,
    d_rho_r: jnp.ndarray,
) -> jnp.ndarray:
    """Apply wetting density modification at the liquid-vapour interface.

    Only modifies ghost-cell values that lie within the interface region
    (between the density thresholds). The interface is split into left
    and right contact-line regions, each receiving its own phi/d_rho.

    Args:
        edge_slice: Edge density slice.
        rho_l: Liquid density.
        rho_v: Vapour density.
        phi_l: Left contact angle parameter.
        phi_r: Right contact angle parameter.
        d_rho_l: Left density modification.
        d_rho_r: Right density modification.
        width: Ghost-cell width.

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

    mask_int_outer: jnp.ndarray = jnp.array(edge_slice < rho_upper, dtype=jnp.int64)
    mask_int_inter: jnp.ndarray = jnp.array(edge_slice < rho_lower, dtype=jnp.int64)
    mask_int: jnp.ndarray = mask_int_outer - mask_int_inter
    indices = jnp.arange(edge_slice.shape[0])
    mask_centre: jnp.ndarray = jnp.sum(mask_int * indices) / jnp.count_nonzero(mask_int * indices)

    is_left_region = jnp.bool(mask_int) & (indices < mask_centre)
    # Right region: interface points right of (left contact line + width buffer).
    is_right_region = jnp.bool(mask_int) & (indices > mask_centre)

    # Wetting modification: phi * rho - d_rho, clamped to density bounds
    modified_left = jnp.clip(phi_l * edge_slice - d_rho_l, rho_lower, rho_upper)
    modified_right = jnp.clip(phi_r * edge_slice - d_rho_r, rho_lower, rho_upper)

    result = jnp.where(is_right_region, modified_right, edge_slice)
    return jnp.where(is_left_region, modified_left, result)
