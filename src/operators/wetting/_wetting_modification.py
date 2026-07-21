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
    rho_l: float | jnp.ndarray,
    rho_v: float | jnp.ndarray,
    phi_l: float | jnp.ndarray,
    phi_r: float | jnp.ndarray,
    d_rho_l: float | jnp.ndarray,
    d_rho_r: float | jnp.ndarray,
    width: int,
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

    # Interface mask: points between the density thresholds
    in_interface = (edge_slice < rho_upper) & (edge_slice > rho_lower)

    # Split the wall into left/right contact-line regions by the *positions*
    # of the two interface transitions, not by their sign. A droplet (liquid on
    # the wall) and a bubble (vapour on the wall) have opposite density profiles,
    # so the sign of ``diff`` at each contact line flips between them; keying off
    # the sign made bubble runs silently no-op. Sorting transitions by position
    # keeps the leftmost transition as the left contact line for both topologies.
    mask_int = jnp.array(edge_slice < rho_upper, dtype=jnp.int32)
    diff = jnp.diff(mask_int)

    # The two transitions in ascending index order: left contact line, then right.
    transitions = jnp.where(jnp.abs(diff) == 1, size=2, fill_value=0)[0]
    left_cl = transitions[0]
    right_cl = transitions[1]

    indices = jnp.arange(edge_slice.shape[0])
    # Left region: interface points left of (right contact line - width buffer).
    is_left_region = in_interface & (indices < right_cl - width)
    # Right region: interface points right of (left contact line + width buffer).
    is_right_region = in_interface & (indices > left_cl + width)

    # Wetting modification: phi * rho - d_rho, clamped to density bounds
    modified_left = jnp.clip(phi_l * edge_slice - d_rho_l, rho_lower, rho_upper)
    modified_right = jnp.clip(phi_r * edge_slice - d_rho_r, rho_lower, rho_upper)

    result = jnp.where(is_right_region, modified_right, edge_slice)
    return jnp.where(is_left_region, modified_left, result)
