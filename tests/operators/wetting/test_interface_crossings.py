"""The crossing pair of a wall row is the inclusion's, whichever side a stray patch is on.

At a chemical step the destination surface condenses a one-cell patch of the
dispersed phase just ahead of the leading line. The first two crossings in
ascending order dropped a patch at higher x but took one at lower x as the
inclusion, so a bubble rising toward -x lost its leading line at the step.
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
import pytest
from src.operators.wetting._interface_crossings import interface_crossings

_MEAN = 0.5


def _row(*runs: tuple[int, int], n: int = 64, background: float = 0.1, inside: float = 0.9) -> jnp.ndarray:
    """A row at *background* with each ``[start, stop)`` run at *inside*."""
    row = np.full(n, background)
    for start, stop in runs:
        row[start:stop] = inside
    return jnp.asarray(row)


def _pair(row: jnp.ndarray) -> tuple[float, float, bool]:
    x_left, x_right, is_bubble = interface_crossings(row, _MEAN)
    return float(x_left), float(x_right), bool(is_bubble)


@pytest.mark.parametrize("patch", [(8, 9), (50, 51)], ids=["patch-below", "patch-above"])
def test_a_patch_either_side_leaves_the_droplet_pair(patch):
    assert _pair(_row((20, 34), patch)) == _pair(_row((20, 34)))


@pytest.mark.parametrize("patch", [(8, 9), (50, 51)], ids=["patch-below", "patch-above"])
def test_a_patch_either_side_leaves_the_bubble_pair(patch):
    bubble = _row((20, 34), patch, background=0.9, inside=0.1)
    x_left, x_right, is_bubble = _pair(bubble)
    assert (x_left, x_right, is_bubble) == _pair(_row((20, 34), background=0.9, inside=0.1))
    assert is_bubble


def _first_two(row: jnp.ndarray) -> tuple[float, float, bool]:
    """The search before the inclusion pair: the first two crossings, padded with 0."""
    mask = jnp.asarray(row < _MEAN, dtype=jnp.int32)
    diff = jnp.diff(mask)
    i, j = jnp.where(jnp.abs(diff) == 1, size=2, fill_value=0)[0]
    x = [k + (_MEAN - row[k]) / (row[k + 1] - row[k]) for k in (i, j)]
    return float(x[0]), float(x[1]), bool(diff[i] > 0)


@pytest.mark.parametrize(
    "row",
    [
        _row((20, 34)),
        _row((20, 34), background=0.9, inside=0.1),
        _row((40, 64)),  # one crossing: the inclusion runs off the row
        _row((0, 12), background=0.9, inside=0.1),
        _row(),  # no interface
    ],
    ids=["droplet", "bubble", "one-crossing", "one-crossing-bubble", "none"],
)
def test_at_most_two_crossings_give_the_first_two_bit_for_bit(row):
    with np.errstate(divide="ignore", invalid="ignore"):
        np.testing.assert_array_equal(_pair(row), _first_two(row))
