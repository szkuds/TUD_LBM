"""The interface width a restart measures off its init field, and the wall's use of it."""

from __future__ import annotations
import numpy as np
import pytest
from src.config import SimulationConfig
from src.config.init_field import measure_interface_width
from src.config.wetting import chemical_step_regions
from src.config.wetting import wetting_interface_width
from src.config.wetting import wetting_knob_limits

NX, NY = 80, 60
RHO_L, RHO_V = 1.0, 0.01
MULTIPHASE = {"eos": "double-well", "kappa": 0.02, "rho_l": RHO_L, "rho_v": RHO_V, "interface_width": 5}


def _cap(width: float, radius: float = 22.0) -> np.ndarray:
    """A sessile cap on the bottom wall with a ``tanh(2 x / width)`` interface."""
    x, y = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    distance = np.hypot(x - NX / 2, y + 6.0) - radius
    return RHO_V + (RHO_L - RHO_V) * 0.5 * (1.0 - np.tanh(2.0 * distance / width))


@pytest.mark.parametrize("width", [5.0, 12.0])
def test_a_tanh_interface_measures_its_own_width(width):
    """Central differences and the contour band read a 5-cell interface 5% wide, never narrow."""
    measured = measure_interface_width(_cap(width))

    assert measured == pytest.approx(width, rel=0.08)
    assert measured >= width


def test_the_five_dimensional_snapshot_layout_is_measured_alike():
    rho = _cap(5.0)

    assert measure_interface_width(rho[:, :, None, None, None]) == pytest.approx(measure_interface_width(rho))


def test_a_steepened_wall_row_does_not_move_the_width():
    """A wetting edge steepens the rows it acts on; the width is the bulk interface's."""
    rho = _cap(8.0)
    steepened = rho.copy()
    steepened[:, :2] = np.where(rho[:, :2] > 0.5 * (RHO_L + RHO_V), RHO_L, RHO_V)

    assert measure_interface_width(steepened) == pytest.approx(measure_interface_width(rho), rel=0.02)


def test_a_field_without_an_interface_has_no_width():
    assert measure_interface_width(np.full((NX, NY), RHO_L)) is None


def test_a_field_that_is_not_two_dimensional_has_no_width():
    assert measure_interface_width(np.linspace(RHO_V, RHO_L, NX)) is None


def test_a_restart_bounds_its_wall_by_the_init_field_width(tmp_path):
    npz_path = tmp_path / "init_state.npz"
    np.savez(npz_path, rho=_cap(9.0)[:, :, None, None, None])
    config = SimulationConfig(
        multiphase=MULTIPHASE,
        grid_shape=(NX, NY),
        init_type="init_from_file",
        init_dir=str(npz_path),
        initialisation={},
    )

    assert wetting_interface_width(config) == pytest.approx(9.0, rel=0.05)


def test_a_fresh_run_has_no_measured_width():
    assert wetting_interface_width(SimulationConfig(multiphase=MULTIPHASE, grid_shape=(NX, NY))) is None


def test_a_single_phase_run_has_no_width():
    assert wetting_interface_width(SimulationConfig(grid_shape=(NX, NY))) is None


# --- the knob limits: d_rho is a density, scaled by the contrast over the width ---------------


def test_a_fresh_run_limits_d_rho_by_the_configured_contrast_and_width():
    limits = wetting_knob_limits(SimulationConfig(multiphase=MULTIPHASE, grid_shape=(NX, NY)))

    assert limits is not None
    assert limits.phi_max == pytest.approx(1.0 + 5.0 / 5)
    assert limits.d_rho_max == pytest.approx(1.5 * (RHO_L - RHO_V) / 5)


def test_d_rho_limit_grows_with_the_density_contrast():
    """A twelvefold contrast (a Carnahan-Starling bubble against the double-well) needs a twelvefold d_rho."""
    dense = MULTIPHASE | {"rho_l": 12.0, "rho_v": 0.01}

    small = wetting_knob_limits(SimulationConfig(multiphase=MULTIPHASE, grid_shape=(NX, NY)))
    large = wetting_knob_limits(SimulationConfig(multiphase=dense, grid_shape=(NX, NY)))

    assert small is not None
    assert large is not None
    assert large.d_rho_max / small.d_rho_max == pytest.approx((12.0 - 0.01) / (RHO_L - RHO_V))
    assert large.phi_max == small.phi_max


def test_a_restart_limits_d_rho_by_the_init_field_contrast_and_width(tmp_path):
    rho = _cap(9.0)
    npz_path = tmp_path / "init_state.npz"
    np.savez(npz_path, rho=rho[:, :, None, None, None])
    config = SimulationConfig(
        multiphase=MULTIPHASE,
        grid_shape=(NX, NY),
        init_type="init_from_file",
        init_dir=str(npz_path),
        initialisation={},
    )
    width = wetting_interface_width(config)
    assert width is not None

    limits = wetting_knob_limits(config)

    assert limits is not None
    assert limits.d_rho_max == pytest.approx(1.5 * (rho.max() - rho.min()) / width)
    assert limits.phi_max == pytest.approx(1.0 + 5.0 / 5)


def test_a_single_phase_run_has_no_knob_limits():
    assert wetting_knob_limits(SimulationConfig(grid_shape=(NX, NY))) is None


def test_a_bubble_post_step_region_pulls_with_a_third_of_the_d_rho_limit():
    """At the limit the fixed post-step cells drove the wall liquid to the spinodal ahead of the line."""
    bubble = SimulationConfig(
        multiphase=MULTIPHASE,
        grid_shape=(NX, NY),
        boundary_conditions={"top": "wetting"},
        hysteresis={},
        chemical_step={
            "chemical_step_location": 0.5,
            "ca_advancing_pre_step": 120.0,
            "ca_receding_pre_step": 100.0,
            "ca_advancing_post_step": 70.0,
            "ca_receding_post_step": 50.0,
        },
        initialisation={"centres": [[0.7, 1.0]], "radii": [0.2], "dispersed": "vapour"},
        init_type="multiphase_bubbles",
    )
    limits = wetting_knob_limits(bubble)
    regions = chemical_step_regions(bubble)

    assert limits is not None
    assert regions is not None
    assert regions.post_phi == 1.0
    assert regions.post_d_rho == pytest.approx(limits.d_rho_max / 3)
