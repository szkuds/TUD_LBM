"""Contact-angle computation — pure function.

Ported from :class:`simulation_operators.wetting.ContactAngle`.
Computes left and right contact angles from the density field using
linear interpolation at the liquid-vapour interface.

All operations are JAX-compatible and jittable.
"""

from __future__ import annotations
import math
import jax.numpy as jnp
from src.operators.wetting._canonical_view import to_canonical
from src.registry import wetting_operator


@wetting_operator(name="contact_angle")
def compute_contact_angle(
    rho: jnp.ndarray,
    rho_mean: float | jnp.ndarray,
    *,
    edge: str = "bottom",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute contact angles (left and right) from a density field.

    The field is first mapped into the wall-aligned canonical frame for
    *edge* (see :func:`~src.operators.wetting._canonical_view.to_canonical`),
    so the wetting wall is at column 0 with liquid above. The algorithm then
    finds the liquid-vapour transition at the two fluid rows nearest the wall
    (``j=1`` and ``j=2``), interpolates the interface tangential position, and
    derives the contact angle from the slope.

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        rho_mean: Mean density ``(rho_l + rho_v) / 2``.
        edge: Wetting wall — ``"bottom"`` (default), ``"top"``, ``"left"``,
            or ``"right"``.

    Returns:
        ``(ca_left, ca_right)`` — contact angles in **degrees**
        (scalar ``jnp.ndarray``).
    """
    if rho.shape[2] != 1:
        msg = "Contact angle computation only implemented in 2D (nz=1)"
        raise ValueError(msg)

    rho_2d = to_canonical(rho[:, :, 0, 0, 0], edge)  # (tangential, normal)

    array_j0 = rho_2d[:, 1]
    array_j1 = rho_2d[:, 2]

    # Binary masks: True where density is below rho_mean (vapour)
    mask_j0 = jnp.array(array_j0 < rho_mean, dtype=jnp.int32)
    mask_j1 = jnp.array(array_j1 < rho_mean, dtype=jnp.int32)

    diff_j0 = jnp.diff(mask_j0)
    diff_j1 = jnp.diff(mask_j1)

    # Droplet's left edge: liquid → vapour scanning +tangential (diff == -1)
    idx_left_j0 = jnp.where(diff_j0 == -1, size=1, fill_value=0)[0][0]
    idx_left_j1 = jnp.where(diff_j1 == -1, size=1, fill_value=0)[0][0]

    # Droplet's right edge: vapour → liquid (diff == +1), then +1
    idx_right_j0 = jnp.where(diff_j0 == 1, size=1, fill_value=0)[0][0] + 1
    idx_right_j1 = jnp.where(diff_j1 == 1, size=1, fill_value=0)[0][0] + 1

    # Linear interpolation for sub-cell interface location
    x_left_j0 = idx_left_j0 + ((rho_mean - array_j0[idx_left_j0]) / (array_j0[idx_left_j0 + 1] - array_j0[idx_left_j0]))
    x_left_j1 = idx_left_j1 + ((rho_mean - array_j1[idx_left_j1]) / (array_j1[idx_left_j1 + 1] - array_j1[idx_left_j1]))
    x_right_j0 = idx_right_j0 - (
        (rho_mean - array_j0[idx_right_j0]) / (array_j0[idx_right_j0 - 1] - array_j0[idx_right_j0])
    )
    x_right_j1 = idx_right_j1 - (
        (rho_mean - array_j1[idx_right_j1]) / (array_j1[idx_right_j1 - 1] - array_j1[idx_right_j1])
    )

    ca_left = jnp.rad2deg(math.pi / 2.0 + jnp.arctan(x_left_j0 - x_left_j1))
    ca_right = jnp.rad2deg(math.pi / 2.0 + jnp.arctan(x_right_j1 - x_right_j0))

    return ca_left, ca_right
