"""Tests for physical parameter overview generation."""

from __future__ import annotations
import math
from types import SimpleNamespace
import numpy as np
import pytest
from src.config import SimulationConfig
from src.simulation_io.analysis.physical_parameters import build_overview
from src.simulation_io.analysis.physical_parameters import write_physical_parameters
from src.simulation_io.analysis.physical_parameters.physical_parameters import _contact_line_length_from_rho
from src.simulation_io.analysis.physical_parameters.physical_parameters import _droplet_area_from_rho
from src.simulation_io.analysis.physical_parameters.physical_parameters import _get_contact_line_length_from_file
from src.simulation_io.analysis.physical_parameters.physical_parameters import _get_setup_droplet_area
from src.simulation_io.analysis.physical_parameters.physical_parameters import _resolve_gravity_inclination
from src.simulation_io.analysis.physical_parameters.physical_parameters import _resolve_gravity_value


def _mp_config(**kwargs) -> SimulationConfig:
    base = {
        "sim_type": "multiphase",
        "grid_shape": (40, 20),
        "eos": "double-well",
        "kappa": 0.02,
        "rho_l": 1.0,
        "rho_v": 0.5,
        "interface_width": 2,
        "gravity_force": {"force_g": 1e-6},
    }
    base.update(kwargs)
    return SimulationConfig(**base)  # ty: ignore[invalid-argument-type]


def test_build_overview_uses_droplet_area_length_when_available():
    cfg = _mp_config(initialisation={"centres": [[0.5, 0.1]], "radii": [0.4]})

    text = build_overview(cfg)

    assert "gamma (surface tension):" in text
    assert "Oh (Ohnesorge number):" in text
    assert "Bo (Bond number):" in text
    assert "sqrt(A/pi), init geometry" in text


def test_build_overview_falls_back_to_grid_x_length_when_droplet_missing():
    cfg = _mp_config(initialisation={"centres": [], "radii": []})

    text = build_overview(cfg)

    assert "Oh (Ohnesorge number):" in text
    assert "Bo (Bond number):" in text
    assert "grid_x" in text


def test_write_physical_parameters_creates_output_file(tmp_path):
    cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
    out = tmp_path / "nested" / "physical_parameters.txt"

    write_physical_parameters(cfg, out)

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "PHYSICAL PARAMETER OVERVIEW" in content
    assert "Collision" in content


def test_build_overview_splits_bond_number_for_inclined_gravity():
    cfg = _mp_config(
        gravity_force={"force_g": 1e-6, "inclination_angle_deg": 30.0},
        initialisation={"centres": [[0.5, 0.1]], "radii": [0.4]},
    )

    text = build_overview(cfg)

    assert "Bo (Bond number):" in text
    assert "Bo_perp (Bond normal):" in text
    assert "Bo_parallel (Bond tangential):" in text


def test_build_overview_uses_init_from_file_length_scale(tmp_path):
    rho = np.full((40, 20, 1, 1, 1), 0.5)
    rho[10:30, 0, 0, 0, 0] = 1.0
    npz_path = tmp_path / "init_state.npz"
    np.savez(npz_path, rho=rho)

    cfg = _mp_config(init_type="init_from_file", init_dir=str(npz_path), initialisation={})
    text = build_overview(cfg)

    # 20 liquid cells -> L_eff = sqrt(20/pi) ~= 2.523
    expected = math.sqrt(20.0 / math.pi)
    assert f"L_eff={expected:.4g} (sqrt(A/pi), init_from_file)" in text


def test_build_overview_falls_back_when_init_from_file_rho_missing(tmp_path):
    npz_path = tmp_path / "init_state_missing_rho.npz"
    np.savez(npz_path, u=np.zeros((40, 20, 1, 1, 2)))

    cfg = _mp_config(init_type="init_from_file", init_dir=str(npz_path), initialisation={})
    text = build_overview(cfg)

    assert "L=40.0 (grid_x)" in text


def test_resolve_gravity_value_prefers_scalar_g_over_force_dict():
    cfg = _mp_config(g=2e-6, gravity_force={"force_g": 1e-6})
    assert _resolve_gravity_value(cfg) == pytest.approx(2e-6)


def test_resolve_gravity_value_supports_masked_force():
    cfg = _mp_config(gravity_force=None, gravity_masked_force={"force_g": 3e-6})
    assert _resolve_gravity_value(cfg) == pytest.approx(3e-6)


def test_resolve_gravity_value_rejects_both_force_variants():
    cfg = SimpleNamespace(g=None, gravity_force={"force_g": 1e-6}, gravity_masked_force={"force_g": 2e-6})
    with pytest.raises(ValueError, match="Only one gravity force"):
        _resolve_gravity_value(cfg)  # ty: ignore[invalid-argument-type]


def test_resolve_gravity_inclination_defaults_to_zero_when_missing_key():
    cfg = _mp_config(gravity_force={"force_g": 1e-6})
    assert _resolve_gravity_inclination(cfg) == pytest.approx(0.0)


def test_setup_droplet_area_full_circle_away_from_walls():
    # r = 0.25 * min(40, 20) = 5; nearest wall at distance 10 >= r -> full circle
    cfg = _mp_config(initialisation={"centres": [[0.5, 0.5]], "radii": [0.25]})
    area = _get_setup_droplet_area(cfg)
    assert area == pytest.approx(math.pi * 5.0**2)


def test_setup_droplet_area_clipped_by_nearest_wall():
    # r = 0.4 * 20 = 8; centre 2 lu from the bottom wall -> circular segment removed
    cfg = _mp_config(initialisation={"centres": [[0.5, 0.1]], "radii": [0.4]})
    r, d = 8.0, 2.0
    expected = math.pi * r**2 - (r**2 * math.acos(d / r) - d * math.sqrt(r**2 - d**2))
    area = _get_setup_droplet_area(cfg)
    assert area == pytest.approx(expected)


def test_setup_droplet_area_returns_none_without_droplet():
    cfg = _mp_config(initialisation={"centres": [], "radii": []})
    assert _get_setup_droplet_area(cfg) is None


def test_droplet_area_from_rho_counts_liquid_cells():
    rho = np.full((40, 20, 1, 1, 1), 0.5)
    rho[10:30, 3:8, 0, 0, 0] = 1.0
    assert _droplet_area_from_rho(rho, rho_mean=0.75) == pytest.approx(20.0 * 5.0)


def test_droplet_area_from_rho_returns_none_for_all_vapour_field():
    rho = np.full((40, 20, 1, 1, 1), 0.5)
    assert _droplet_area_from_rho(rho, rho_mean=0.75) is None


def test_contact_line_length_from_rho_returns_none_for_degenerate_profile():
    rho = np.ones((12, 6, 1, 1, 1)) * 0.5
    assert _contact_line_length_from_rho(rho, rho_mean=0.5) is None


def test_get_contact_line_length_from_file_returns_none_for_missing_file(tmp_path):
    cfg = _mp_config(init_type="init_from_file", init_dir=str(tmp_path / "missing.npz"), initialisation={})
    assert _get_contact_line_length_from_file(cfg) is None
