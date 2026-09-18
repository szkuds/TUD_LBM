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
confined to one contiguous run, and its density bounds are measured locally.
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
    rho_l: float = _RHO_L,
) -> jnp.ndarray:
    """A bubble footprint on a wall: vapour between the two contact lines.

    ``liquid_drop`` lowers the liquid everywhere and ``tilt`` adds a further
    linear fall along the wall, the two ways gravity moves a wetting wall's
    liquid density away from the prescribed ``rho_l``.
    """
    x = np.arange(_N, dtype=float)
    vapour_fraction = 0.5 * (np.tanh((x - _LEFT) / _WIDTH) - np.tanh((x - _RIGHT) / _WIDTH))
    liquid = rho_l - liquid_drop - tilt * (x / (_N - 1))
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


def test_bounds_are_measured_locally_not_from_the_prescribed_densities():
    drifted = _wall_row(liquid_drop=0.9, tilt=0.6)
    left, right = wetting_regions(drifted, _RHO_L, _RHO_V)

    # Each side's upper bound tracks the liquid density beside *that* contact
    # line, so the tilted wall gives the two sides different bounds.
    assert float(left.rho_upper) > float(right.rho_upper)
    for region in (left, right):
        assert float(region.rho_lower) < float(region.rho_upper)
        assert float(region.rho_upper) < 0.95 * _RHO_L + 0.05 * _RHO_V


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

    The hysteresis optimizer differentiates through the applicator, so the local
    bounds must stay finite even for an anchor the contrast floor rejected.
    """
    row = jnp.full(_N, _RHO_L)

    def total(phi: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(_apply_wetting_modification(row, _RHO_L, _RHO_V, phi, phi, 0.0, 0.0))

    grad = jax.grad(total)(jnp.asarray(1.1))
    assert np.isfinite(float(grad))
