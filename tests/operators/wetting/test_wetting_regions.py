"""The contact-line regions the wetting BC modifies.

The predecessor selected ghost cells by the absolute window
``[0.05 rho_l + 0.95 rho_v, 0.95 rho_l + 0.05 rho_v)`` and split it at its index
centroid. That leaves ``0.05 (rho_l - rho_v)`` of headroom above the band, and on
a gravitating wall the liquid equilibrates hydrostatically below the prescribed
``rho_l`` by more than that — a measured inclined bubble run
(``force_g = 5e-06`` at 60°, ``rho_l = 12.18``, headroom 0.608) developed a drop
of 0.76, bulk liquid entered the band, the region grew from 54 to 158 of 200 wall
cells and the run diverged. The tests here pin the properties that make that
impossible: the region is anchored on the contact line, bounded by a window,
confined to one contiguous run, and its density bounds are measured rather than
prescribed.

They also pin the property the *first* attempt at that fix broke. Measuring the
bounds per contact line splits the two ceilings whenever the two flanks hold
different amounts of liquid, which injects more density on one side than the
other and walks the inclusion along the wall. The bounds are therefore measured
over both anchor windows together and shared.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.operators.wetting._wetting_modification import _apply_wetting_modification
from src.operators.wetting._wetting_modification import anchor_window_half_width
from src.operators.wetting._wetting_modification import wetting_regions

_RHO_L, _RHO_V = 12.18, 0.015
_N = 200
#: Contact lines of the baseline profile, matching the measured run's ~126/~168.
_LEFT, _RIGHT = 126.0, 168.0
_WIDTH = 6.0


def _wall_row(
    *,
    liquid_drop: float = 0.0,
    tilt: float = 0.0,
    right_flank_drop: float = 0.0,
    rho_l: float = _RHO_L,
    left: float = _LEFT,
    right: float = _RIGHT,
) -> jnp.ndarray:
    """A bubble footprint on a wall: vapour between the two contact lines.

    ``liquid_drop`` lowers the liquid everywhere and ``tilt`` adds a further
    linear fall along the wall, the two ways gravity moves a wetting wall's
    liquid density away from the prescribed ``rho_l``.

    ``right_flank_drop`` lowers the liquid beyond the right contact line only,
    leaving both interfaces the same shape. That is the measured geometry behind
    the drift: an inclusion crowded against the far wall, whose right flank holds
    too little liquid for a window centred there to see bulk.
    """
    x = np.arange(_N, dtype=float)
    vapour_fraction = 0.5 * (np.tanh((x - left) / _WIDTH) - np.tanh((x - right) / _WIDTH))
    flank = 0.5 * (1.0 + np.tanh((x - (right + _WIDTH)) / _WIDTH))
    liquid = rho_l - liquid_drop - tilt * (x / (_N - 1)) - right_flank_drop * flank
    return jnp.asarray(liquid + (_RHO_V - liquid) * vapour_fraction)


def _extent(region) -> int:
    return int(np.asarray(region.mask).sum())


def test_regions_sit_on_the_two_contact_lines():
    left, right = wetting_regions(_wall_row(), _RHO_L, _RHO_V)

    assert float(left.anchor) == pytest.approx(_LEFT, abs=1.0)
    assert float(right.anchor) == pytest.approx(_RIGHT, abs=1.0)

    half_width = anchor_window_half_width(_N)
    for region, anchor in ((left, _LEFT), (right, _RIGHT)):
        cells = np.flatnonzero(np.asarray(region.mask))
        assert cells.size > 0
        assert np.abs(cells - anchor).max() <= half_width
        # One contiguous run, not a scattering of cells.
        assert np.array_equal(cells, np.arange(cells.min(), cells.max() + 1))

    assert not np.any(np.asarray(left.mask) & np.asarray(right.mask))


def test_bulk_liquid_below_the_old_upper_bound_does_not_enter_the_region():
    """The regression: hydrostatic drop past the old headroom must not widen the region."""
    old_upper = 0.95 * _RHO_L + 0.05 * _RHO_V
    baseline = _wall_row()
    reference = sum(_extent(r) for r in wetting_regions(baseline, _RHO_L, _RHO_V))

    # Liquid drawn below the old threshold, and tilted as an inclined domain tilts it.
    drifted = _wall_row(liquid_drop=0.9, tilt=0.6)
    assert float(jnp.max(drifted)) < old_upper  # the old band would swallow the whole wall

    left, right = wetting_regions(drifted, _RHO_L, _RHO_V)
    widened = _extent(left) + _extent(right)

    assert widened <= 2 * reference
    assert widened < _N // 4
    for region, anchor in ((left, _LEFT), (right, _RIGHT)):
        assert float(region.anchor) == pytest.approx(anchor, abs=2.0)


def test_bounds_are_measured_not_taken_from_the_prescribed_densities():
    drifted = _wall_row(liquid_drop=0.9, tilt=0.6)
    left, right = wetting_regions(drifted, _RHO_L, _RHO_V)

    # The bounds follow the liquid the wall actually holds, which the drop and
    # tilt have pulled below the prescribed pair.
    for region in (left, right):
        assert float(region.rho_lower) < float(region.rho_upper)
        assert float(region.rho_upper) < 0.95 * _RHO_L + 0.05 * _RHO_V


def test_both_contact_lines_share_one_pair_of_bounds():
    """The bounds belong to the wall, not to a side — the fix for the drift.

    Measuring them per contact line splits the two ceilings whenever the flanks
    hold different amounts of liquid: on this row the predecessor put the left
    ceiling 0.76 above the right, and on the measured run that split was 1.62
    (10.82 against 9.20) and walked the bubble 32 cells into the far wall.
    """
    crowded = _wall_row(right_flank_drop=0.8)
    left, right = wetting_regions(crowded, _RHO_L, _RHO_V)

    half_width = anchor_window_half_width(_N)
    row = np.asarray(crowded)
    beside_left = row[int(_LEFT) - half_width : int(_LEFT) + 1].max()
    beside_right = row[int(_RIGHT) : int(_RIGHT) + half_width + 1].max()
    assert beside_left - beside_right > 0.5  # the flanks really do differ

    assert float(left.rho_upper) == float(right.rho_upper)
    assert float(left.rho_lower) == float(right.rho_lower)


def test_a_mirror_symmetric_row_is_modified_symmetrically():
    """Equal phi on a mirrored row must inject equal density on both sides.

    This is the invariant the drift violated: an unbalanced injection is an
    unbalanced tangential force at the wall, and nothing restores it.
    """
    half = (_RIGHT - _LEFT) / 2.0
    centre = (_N - 1) / 2.0
    mirrored = _wall_row(left=centre - half, right=centre + half)
    left, right = wetting_regions(mirrored, _RHO_L, _RHO_V)

    np.testing.assert_array_equal(np.asarray(left.mask), np.asarray(right.mask)[::-1])

    modified = np.asarray(_apply_wetting_modification(mirrored, _RHO_L, _RHO_V, 1.05, 1.05, 0.0, 0.0))
    injected = modified - np.asarray(mirrored)
    assert injected[np.asarray(left.mask)].sum() == pytest.approx(
        injected[np.asarray(right.mask)].sum(),
        rel=1e-9,
    )


def test_a_crowded_flank_does_not_widen_the_region_past_the_window():
    """Shared bounds let a depressed flank into the band; the window still caps it."""
    crowded = _wall_row(right_flank_drop=0.8)
    left, right = wetting_regions(crowded, _RHO_L, _RHO_V)

    half_width = anchor_window_half_width(_N)
    assert _extent(left) + _extent(right) <= 2 * (2 * half_width + 1)
    assert _extent(right) > _extent(left)  # the depressed flank is in band
    for region, anchor in ((left, _LEFT), (right, _RIGHT)):
        cells = np.flatnonzero(np.asarray(region.mask))
        assert np.abs(cells - anchor).max() <= half_width


def test_a_disconnected_in_band_cluster_is_excluded():
    """Connectivity, not just the window, is what rejects bulk that enters the band.

    The patch sits *inside* the left anchor's window and holds an in-band
    density, separated from the contact line by bulk liquid — the shape bulk
    liquid took in the measured run once it crossed the old upper bound.
    """
    row = np.array(_wall_row(), copy=True)
    patch = slice(101, 112)
    row[patch] = 0.55 * (_RHO_L + _RHO_V)

    left, right = wetting_regions(jnp.asarray(row), _RHO_L, _RHO_V)
    assert float(left.rho_lower) < row[patch].max() < float(left.rho_upper)  # in band

    cells = np.flatnonzero(np.asarray(left.mask | right.mask))
    assert cells.size > 0
    assert not np.any((cells >= patch.start) & (cells < patch.stop))


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(jnp.full(_N, _RHO_L), id="all-liquid"),
        pytest.param(jnp.full(_N, _RHO_V), id="all-vapour"),
        pytest.param(jnp.asarray(np.linspace(_RHO_V, _RHO_L, _N)), id="single-crossing"),
    ],
)
def test_a_row_without_two_contact_lines_is_a_no_op(row):
    """No interface to anchor on: the BC leaves the row untouched rather than guessing."""
    left, right = wetting_regions(row, _RHO_L, _RHO_V)
    assert _extent(left) == 0
    assert _extent(right) == 0

    result = _apply_wetting_modification(row, _RHO_L, _RHO_V, 1.3, 1.3, 0.1, 0.1)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(row))
    assert np.all(np.isfinite(np.asarray(result)))


def test_each_side_moves_only_its_own_contact_line():
    row = _wall_row()
    base = np.asarray(_apply_wetting_modification(row, _RHO_L, _RHO_V, 1.0, 1.0, 0.0, 0.0))
    left_only = np.asarray(_apply_wetting_modification(row, _RHO_L, _RHO_V, 1.3, 1.0, 0.0, 0.0))
    right_only = np.asarray(_apply_wetting_modification(row, _RHO_L, _RHO_V, 1.0, 1.3, 0.0, 0.0))

    moved_left = np.flatnonzero(~np.isclose(base, left_only))
    moved_right = np.flatnonzero(~np.isclose(base, right_only))

    assert moved_left.size > 0
    assert moved_right.size > 0
    assert moved_left.max() < moved_right.min()


def test_regions_are_jittable():
    jitted = jax.jit(wetting_regions)
    left, right = jitted(_wall_row(), _RHO_L, _RHO_V)

    eager_left, eager_right = wetting_regions(_wall_row(), _RHO_L, _RHO_V)
    np.testing.assert_array_equal(np.asarray(left.mask), np.asarray(eager_left.mask))
    np.testing.assert_array_equal(np.asarray(right.mask), np.asarray(eager_right.mask))


def test_gradients_stay_finite_when_a_side_is_empty():
    """``jnp.where`` poisons gradients through a non-finite dead branch.

    The hysteresis optimiser differentiates through the applicator, so the local
    bounds must stay finite even for an anchor the contrast floor rejected.
    """
    row = jnp.full(_N, _RHO_L)

    def total(phi: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(_apply_wetting_modification(row, _RHO_L, _RHO_V, phi, phi, 0.0, 0.0))

    grad = jax.grad(total)(jnp.asarray(1.1))
    assert np.isfinite(float(grad))
