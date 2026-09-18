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
from src.operators.wetting._interface_crossings import interface_crossings
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

    Both angles are measured **through the dispersed phase** — the liquid for a
    droplet, the vapour for a bubble. That is what makes the measurement mean
    the same thing for both topologies: the dispersed phase spreading over the
    wall is always a *decreasing* angle, so one hysteresis window applies
    unchanged to a droplet and a bubble.

    The convention falls out of the geometry rather than needing a correction.
    :func:`~src.operators.wetting._interface_crossings.interface_crossings`
    returns the two crossings in ascending tangential order, so the dispersed
    phase is by construction the region *between* them; the mirrored argument
    order below then measures into that region from either side.

    ``left``/``right`` are positional along the canonical tangential axis,
    matching how the wetting applicator splits the interface.

    To recover the liquid angle for a bubble, take ``180° − θ``.

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        rho_mean: Mean density ``(rho_l + rho_v) / 2``.
        edge: Wetting wall — ``"bottom"`` (default), ``"top"``, ``"left"``,
            or ``"right"``.

    Returns:
        ``(ca_left, ca_right)`` — dispersed-phase contact angles in **degrees**
        (scalar ``jnp.ndarray``).
    """
    if rho.shape[2] != 1:
        msg = "Contact angle computation only implemented in 2D (nz=1)"
        raise ValueError(msg)

    rho_2d = to_canonical(rho[:, :, 0, 0, 0], edge)  # (tangential, normal)

    # Interface positions at the two fluid rows nearest the wall.
    x_left_j0, x_right_j0, _ = interface_crossings(rho_2d[:, 1], rho_mean)
    x_left_j1, x_right_j1, _ = interface_crossings(rho_2d[:, 2], rho_mean)

    # Slope across the unit row spacing. The mirrored argument order measures
    # into the region between the two crossings from either side, so both
    # angles subtend the dispersed phase for a droplet and a bubble alike.
    ca_left = jnp.rad2deg(math.pi / 2.0 + jnp.arctan(x_left_j0 - x_left_j1))
    ca_right = jnp.rad2deg(math.pi / 2.0 + jnp.arctan(x_right_j1 - x_right_j0))

    return ca_left, ca_right
