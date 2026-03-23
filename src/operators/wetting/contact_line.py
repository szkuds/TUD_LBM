"""Contact-line-location computation — pure function.

Ported from :class:`simulation_operators.wetting.ContactLineLocation`.
Computes left and right contact-line locations (CLL) at the solid
boundary from the density field and the measured contact angles.

All operations are JAX-compatible and jittable.
"""

from __future__ import annotations

import jax.numpy as jnp

from registry import wetting_operator


@wetting_operator(name="contact_line_location")
def compute_contact_line_location(
    rho: jnp.ndarray,
    ca_left: jnp.ndarray,
    ca_right: jnp.ndarray,
    rho_mean: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute contact-line locations at the solid wall.

    The algorithm finds the liquid–vapour transition at the wall row
    (``j=0``), interpolates the interface x-position, and projects
    down to the solid using the measured contact angle.

    Args:
        rho: Density field, shape ``(nx, ny, 1, 1)``.
        ca_left: Left contact angle in degrees (scalar).
        ca_right: Right contact angle in degrees (scalar).
        rho_mean: Mean density ``(rho_l + rho_v) / 2``.

    Returns:
        ``(cll_left, cll_right)`` — contact-line x-positions
        (scalar ``jnp.ndarray``).
    """
    rho_2d = rho[:, :, 0, 0]  # (nx, ny)
    array_j0 = rho_2d[:, 0]

    mask_j0 = jnp.array(array_j0 < rho_mean, dtype=jnp.int32)
    diff_j0 = jnp.diff(mask_j0)

    # Left transition
    idx_left = jnp.where(diff_j0 == -1, size=1, fill_value=0)[0][0]
    # Right transition
    idx_right = jnp.where(diff_j0 == 1, size=1, fill_value=0)[0][0] + 1

    # Sub-cell interpolation
    x_left_j0 = idx_left + (
        (rho_mean - array_j0[idx_left]) / (array_j0[idx_left + 1] - array_j0[idx_left])
    )
    x_right_j0 = idx_right - (
        (rho_mean - array_j0[idx_right])
        / (array_j0[idx_right - 1] - array_j0[idx_right])
    )

    # Project to solid surface using measured contact angle
    cll_left = x_left_j0 - 1.0 / (2.0 * jnp.tan(jnp.deg2rad(ca_left)))
    cll_right = x_right_j0 + 1.0 / (2.0 * jnp.tan(jnp.deg2rad(ca_right)))

    return cll_left, cll_right
