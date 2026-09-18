"""Threshold crossings of a wall-parallel density row, and phase topology.

Both :mod:`~src.operators.wetting._contact_angle` and
:mod:`~src.operators.wetting._contact_line` need the same two things from a row
of the canonical view: where the density crosses ``rho_mean``, and which phase
is the *dispersed* one.

**Left/right is positional.** ``x_left`` is the leftmost crossing along the
canonical tangential axis and ``x_right`` the next one, regardless of which
direction the density steps. This is deliberate: the wetting *applicator*
(:func:`~src.operators.wetting._wetting_modification._apply_wetting_modification`)
splits the interface positionally too, so keying the measurement off the sign of
the transition — as the original droplet-only code did — cross-wired the two for
a bubble, feeding ``phi_left`` into the physically right contact line.

**Topology.** Scanning along +tangential, a droplet (liquid dispersed in vapour)
crosses low→high then high→low; a bubble (vapour dispersed in liquid) does the
reverse. The sign of the leftmost crossing therefore identifies the topology,
and that is what ``is_bubble`` reports.

All operations are JAX-compatible and jittable: the crossing search uses a
static ``size=2``, and ``edge`` is resolved at trace time.
"""

from __future__ import annotations
import jax.numpy as jnp
from src.operators.wetting._canonical_view import to_canonical


def interface_crossings(
    row: jnp.ndarray,
    rho_mean: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Locate the two liquid-vapour crossings of *row*.

    Args:
        row: Densities along the canonical tangential axis, shape ``(n,)``.
        rho_mean: Threshold density ``(rho_l + rho_v) / 2``.

    Returns:
        ``(x_left, x_right, is_bubble)`` — the two sub-cell crossing positions
        in ascending tangential order, and a boolean scalar that is True when
        the dispersed phase is vapour.

    Note:
        A row with fewer than two crossings (no interface, or a dispersed phase
        detached from the wall) collapses both positions to index 0 rather than
        raising — the search is padded with ``fill_value=0`` to keep the shape
        static under ``jit``.
    """
    mask = jnp.asarray(row < rho_mean, dtype=jnp.int32)  # 1 == vapour
    diff = jnp.diff(mask)
    idx = jnp.where(jnp.abs(diff) == 1, size=2, fill_value=0)[0]
    idx_left, idx_right = idx[0], idx[1]

    # One forward interpolation covers both transition signs: the crossing sits
    # between samples ``i`` and ``i + 1`` either way, so the mirrored backward
    # form the droplet-only code used for the right side was redundant.
    x_left = idx_left + (rho_mean - row[idx_left]) / (row[idx_left + 1] - row[idx_left])
    x_right = idx_right + (rho_mean - row[idx_right]) / (row[idx_right + 1] - row[idx_right])

    # diff == +1 is liquid → vapour, i.e. density stepping high → low.
    is_bubble = diff[idx_left] > 0

    return x_left, x_right, is_bubble


def detect_bubble(
    rho: jnp.ndarray,
    rho_mean: float | jnp.ndarray,
    *,
    edge: str = "bottom",
) -> jnp.ndarray:
    """Return True when the phase dispersed at the wetting wall is vapour.

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        rho_mean: Threshold density ``(rho_l + rho_v) / 2``.
        edge: Wetting wall — ``"bottom"`` (default), ``"top"``, ``"left"``,
            or ``"right"``.

    Returns:
        Boolean JAX scalar.
    """
    rho_2d = to_canonical(rho[:, :, 0, 0, 0], edge)
    return interface_crossings(rho_2d[:, 0], rho_mean)[2]
