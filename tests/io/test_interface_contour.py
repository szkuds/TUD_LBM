"""Interface markers and contour geometry (``analysis.interface_contour``)."""

from __future__ import annotations
import numpy as np
import pytest
from src.config import SimulationConfig
from src.simulation_io.analysis.interface_contour import DEFAULT_INTERFACE_LEVELS
from src.simulation_io.analysis.interface_contour import config_rho_mean
from src.simulation_io.analysis.interface_contour import interface_lines
from src.simulation_io.analysis.interface_contour import level_value
from src.simulation_io.analysis.interface_contour import measured_rho_mean
from src.simulation_io.analysis.interface_contour import resolve_interface_levels

_NX, _NY = 64, 48
_CENTRE = (30, 22)
_RADIUS = 12.0


def _multiphase_config(**overrides: object) -> SimulationConfig:
    params: dict[str, object] = {
        "sim_type": "multiphase",
        "grid_shape": (_NX, _NY),
        "eos": "double-well",
        "kappa": 0.02,
        "interface_width": 4,
        "rho_l": 1.0,
        "rho_v": 0.1,
    }
    params.update(overrides)
    return SimulationConfig(**params)  # ty: ignore[invalid-argument-type]


def _droplet(rho_l: float = 1.0, rho_v: float = 0.1) -> np.ndarray:
    """A tanh droplet of radius ``_RADIUS`` in ``(nx, ny)`` layout."""
    x, y = np.meshgrid(np.arange(_NX), np.arange(_NY), indexing="ij")
    distance = np.hypot(x - _CENTRE[0], y - _CENTRE[1])
    return rho_v + (rho_l - rho_v) * 0.5 * (1.0 - np.tanh((distance - _RADIUS) / 2.0))


def test_contour_of_droplet_is_one_closed_circle_at_its_radius():
    rho = _droplet()
    lines = interface_lines(rho, 0.55)

    assert len(lines) == 1
    line = lines[0]
    np.testing.assert_allclose(line[0], line[-1])
    radii = np.hypot(line[:, 0] - _CENTRE[0], line[:, 1] - _CENTRE[1])
    assert abs(float(radii.mean()) - _RADIUS) < 0.1


def test_contour_vertices_use_x_for_the_first_array_axis():
    """Vertices are (x, y) with x indexing ``rho[:, ...]`` — the imshow(rho.T) frame."""
    rho = np.zeros((_NX, _NY))
    rho[40:50, 5:15] = 1.0

    vertices = np.concatenate(interface_lines(rho, 0.5))

    assert vertices[:, 0].min() >= 39.0
    assert vertices[:, 0].max() <= 50.0
    assert vertices[:, 1].min() >= 4.0
    assert vertices[:, 1].max() <= 15.0


def test_measured_level_ignores_interface_overshoot_that_moves_min_max():
    rho = _droplet()
    baseline = measured_rho_mean(rho, 0.55)

    overshoot = rho.copy()
    overshoot[_CENTRE[0] + int(_RADIUS), _CENTRE[1]] = 1.6
    min_max_before = 0.5 * (rho.min() + rho.max())
    min_max_after = 0.5 * (overshoot.min() + overshoot.max())

    assert baseline is not None
    assert measured_rho_mean(overshoot, 0.55) == pytest.approx(baseline, abs=1e-12)
    assert min_max_after - min_max_before > 0.2


def test_bulk_density_drift_moves_measured_level_but_not_config_level():
    config = _multiphase_config()
    baseline = level_value("measured", config, _droplet())
    drifted = level_value("measured", config, _droplet(rho_v=0.3))

    assert level_value("config", config, _droplet(rho_v=0.3)) == pytest.approx(0.55)
    assert baseline is not None
    assert drifted is not None
    # Vapour rose by 0.2, so the bulk midpoint rises by ~0.1.
    assert drifted - baseline == pytest.approx(0.1, abs=0.01)


def test_config_level_is_none_without_coexistence_densities():
    config = SimulationConfig(grid_shape=(_NX, _NY))

    assert config_rho_mean(config) is None
    assert level_value("config", config, _droplet()) is None


def test_measured_level_falls_back_to_min_max_seed_without_config_densities():
    rho = _droplet()
    seed = 0.5 * (float(rho.min()) + float(rho.max()))

    measured = level_value("measured", SimulationConfig(grid_shape=(_NX, _NY)), rho)

    assert measured is not None
    assert measured == pytest.approx(measured_rho_mean(rho, seed))
    # Same bulk split as seeding at the config midpoint would give.
    assert measured == pytest.approx(level_value("measured", _multiphase_config(), rho))


def test_uniform_field_has_no_measured_level():
    assert measured_rho_mean(np.full((_NX, _NY), 0.4), 0.4) is None


def test_levels_default_to_both_markers():
    assert resolve_interface_levels(_multiphase_config()) == DEFAULT_INTERFACE_LEVELS
    assert DEFAULT_INTERFACE_LEVELS == ("config", "measured")


def test_selected_levels_are_kept_in_order():
    config = _multiphase_config(interface_levels=["measured"])

    assert resolve_interface_levels(config) == ("measured",)


def test_unknown_level_is_rejected_with_the_choices():
    config = _multiphase_config(interface_levels=["config", "median"])

    with pytest.raises(ValueError, match=r"median.*config.*measured"):
        resolve_interface_levels(config)
    with pytest.raises(ValueError, match="median"):
        level_value("median", config, _droplet())
