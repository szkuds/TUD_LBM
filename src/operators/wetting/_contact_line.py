"""Contact-line-location computation — pure function.

Ported from :class:`simulation_operators.wetting.ContactLineLocation`.
Computes left and right contact-line locations (CLL) at the solid
boundary from the density field and the measured contact angles.

All operations are JAX-compatible and jittable.
"""

from __future__ import annotations
import jax.numpy as jnp
from src.operators.wetting._canonical_view import to_canonical
from src.operators.wetting._interface_crossings import interface_crossings
from src.registry import wetting_operator


@wetting_operator(name="contact_line_location")
def compute_contact_line_location(
    rho: jnp.ndarray,
    ca_left: jnp.ndarray,
    ca_right: jnp.ndarray,
    rho_mean: float | jnp.ndarray,
    *,
    edge: str = "bottom",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute contact-line locations at the solid wall.

    The field is first mapped into the wall-aligned canonical frame for
    *edge* (see :func:`~src.operators.wetting._canonical_view.to_canonical`).
    The algorithm then finds the liquid-vapour transition at the wall row
    (``j=0``), interpolates the interface tangential position, and projects
    down to the solid using the measured contact angle.

    ``left``/``right`` are positional along the canonical tangential axis, so
    ``cll_left < cll_right`` holds for a droplet and a bubble alike.

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        ca_left: Left contact angle in degrees (scalar), measured through the
            dispersed phase as returned by
            :func:`~src.operators.wetting._contact_angle.compute_contact_angle`.
        ca_right: Right contact angle in degrees (scalar), same convention.
        rho_mean: Mean density ``(rho_l + rho_v) / 2``.
        edge: Wetting wall — ``"bottom"`` (default), ``"top"``, ``"left"``,
            or ``"right"``. The returned coordinate is tangential to this
            wall (x for bottom/top, y for left/right).

    Returns:
        ``(cll_left, cll_right)`` — contact-line tangential positions
        (scalar ``jnp.ndarray``).
    """
    if rho.shape[2] != 1:
        msg = "Contact line location computation only implemented in 2D (nz=1)"
        raise ValueError(msg)

    rho_2d = to_canonical(rho[:, :, 0, 0, 0], edge)  # (tangential, normal)
    x_left_j0, x_right_j0, _ = interface_crossings(rho_2d[:, 0], rho_mean)

    # Project the half-cell from the wall row down to the solid along the
    # interface slope. The angles subtend the dispersed phase, which is the
    # region between the two crossings, so the projection carries the same sign
    # for a droplet and a bubble: a dispersed phase that spreads (theta < 90
    # deg) widens toward the wall, one that overhangs (theta > 90 deg) narrows,
    # and cot(theta) changes sign at 90 deg to match.
    cll_left = x_left_j0 - 1.0 / (2.0 * jnp.tan(jnp.deg2rad(ca_left)))
    cll_right = x_right_j0 + 1.0 / (2.0 * jnp.tan(jnp.deg2rad(ca_right)))

    return cll_left, cll_right
