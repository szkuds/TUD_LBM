"""Tests for physical parameter overview generation."""

from __future__ import annotations
from tud_lbm.config import SimulationConfig
from tud_lbm.io.physical_parameters import build_overview
from tud_lbm.io.physical_parameters import write_physical_parameters


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
    return SimulationConfig(**base)


def test_build_overview_uses_contact_line_length_when_available():
    cfg = _mp_config(initialisation={"centres": [[0.5, 0.1]], "radii": [0.4]})

    text = build_overview(cfg)

    assert "gamma (surface tension):" in text
    assert "Oh (Ohnesorge number):" in text
    assert "Bo (Bond number):" in text
    assert "contact line" in text


def test_build_overview_falls_back_to_grid_x_length_when_contact_line_missing():
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
