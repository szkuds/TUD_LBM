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

    The split is **positional** — by index relative to the interface-band
    centroid — so it is identical for a droplet and a bubble. The measurement
    side (:func:`~src.operators.wetting._contact_angle.compute_contact_angle`)
    labels left/right positionally for the same reason; keeping the two in
    lock-step is what guarantees ``phi_l`` addresses the contact line reported
    as ``cll_left``.

    Args:
        edge_slice: Ghost-row densities, shape ``(n,)``.
        rho_l: Liquid density.
        rho_v: Vapour density.
        phi_l: Wetting potential for the left contact line.
        phi_r: Wetting potential for the right contact line.
        d_rho_l: Density offset for the left contact line.
        d_rho_r: Density offset for the right contact line.

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
