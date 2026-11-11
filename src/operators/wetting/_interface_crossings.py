"""Threshold crossings of a wall-parallel density row, and phase topology.

Both :mod:`~src.operators.wetting._contact_angle` and
:mod:`~src.operators.wetting._contact_line` need the same two things from a row
of the canonical view: where the density crosses ``rho_mean``, and which phase
is the *dispersed* one.

**Left/right is positional.** ``x_left`` is the lower crossing along the
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

**The pair is the inclusion's, not the first two.** A row can cross more than
twice: at a chemical step the destination surface's pull condenses a one-cell
patch of the dispersed phase on the wall just ahead of the leading line. Taking
the first two crossings in ascending order dropped such a patch when it lay at
higher ``x`` (a droplet sliding toward ``+x``) but took it *as* the inclusion when
it lay at lower ``x`` (a bubble rising toward ``-x``): the measured left contact
line jumped onto the patch and the run and its x-mirror parted. The pair is the
one bounding the longest closed run of the phase *not* at the row's start (the
continuous phase touches the row ends), which is the inclusion whichever side a
patch is on. With at most two crossings it is the same pair as before, bit for
bit.

All operations are JAX-compatible and jittable: the search keeps static shapes,
and ``edge`` is resolved at trace time.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from src.config.boundary import Edge
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

    With more than two crossings, the pair bounding the longest closed run of the
    phase not at ``row[0]`` (the inclusion) is returned, the lowest one on a tie.

    Note:
        A row with fewer than two crossings (no interface, or a dispersed phase
        detached from the wall) falls back to the first crossings padded with
        index 0 rather than raising — ``fill_value=0`` keeps the shape static
        under ``jit``.
    """
    mask = jnp.asarray(row < rho_mean, dtype=jnp.int32)  # 1 == vapour
    diff = jnp.diff(mask)
    idx_left, idx_right = _inclusion_pair(jnp.abs(diff) == 1, mask[1:] != mask[0])

    # One forward interpolation covers both transition signs: the crossing sits
    # between samples ``i`` and ``i + 1`` either way, so the mirrored backward
    # form the droplet-only code used for the right side was redundant.
    x_left = idx_left + (rho_mean - row[idx_left]) / (row[idx_left + 1] - row[idx_left])
    x_right = idx_right + (rho_mean - row[idx_right]) / (row[idx_right + 1] - row[idx_right])

    # diff == +1 is liquid → vapour, i.e. density stepping high → low.
    is_bubble = diff[idx_left] > 0

    return x_left, x_right, is_bubble


def _inclusion_pair(is_crossing: jnp.ndarray, enters_dispersed: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Indices ``(i, j)`` of the consecutive crossings bounding the longest dispersed run.

    *is_crossing* flags each ``i`` where the row crosses between samples ``i`` and
    ``i + 1``; *enters_dispersed* is True where sample ``i + 1`` is in the phase not
    at the row's start. ``j`` is the next crossing after ``i``. Without such a run,
    the first two crossings padded with ``0``, as the search always gave.
    """
    n = is_crossing.shape[0]
    positions = jnp.arange(n)
    flagged = jnp.where(is_crossing, positions, n)
    # following[i]: the first crossing strictly after i, or n when there is none.
    following = jnp.concatenate([jax.lax.cummin(flagged[::-1])[::-1][1:], jnp.array([n])])
    length = jnp.where(is_crossing & enters_dispersed & (following < n), following - positions, -1)
    best = jnp.argmax(length)
    fallback = jnp.where(is_crossing, size=2, fill_value=0)[0]
    found = length[best] > 0
    return jnp.where(found, best, fallback[0]), jnp.where(found, following[best], fallback[1])


def detect_bubble(
    rho: jnp.ndarray,
    rho_mean: float | jnp.ndarray,
    *,
    edge: Edge = Edge.BOTTOM,
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
