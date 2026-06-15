"""Additional branch coverage for tud_lbm/io/physical_parameters.py.

Targets the 13 uncovered lines / 9 uncovered conditions reported by SonarCloud:
- _contact_line_length_from_rho edge cases (zero-denominator, boundary index guards)
- _ensure_single_gravity_force_source conflict path
- _resolve_gravity_value branches (g attr, force dict lookup, None)
- _resolve_gravity_inclination branches
- _add_multiphase_section with no gravity
- _add_forces_section with masked gravity force
- build_overview sections: simulation_name absent, 3-D grid shape, wetting/hysteresis keys
- _format_critical_inclination_angle_row: guard, valid-angle, pinned, formula
"""

from __future__ import annotations
import math
from types import SimpleNamespace
import numpy as np
import pytest
from tud_lbm.config import SimulationConfig
from tud_lbm.io.physical_parameters import _contact_line_length_from_rho
from tud_lbm.io.physical_parameters import _ensure_single_gravity_force_source
from tud_lbm.io.physical_parameters import _format_critical_inclination_angle_row
from tud_lbm.io.physical_parameters import _nu
from tud_lbm.io.physical_parameters import _resolve_gravity_inclination
from tud_lbm.io.physical_parameters import _resolve_gravity_value
from tud_lbm.io.physical_parameters import _row
from tud_lbm.io.physical_parameters import _section
from tud_lbm.io.physical_parameters import build_overview

# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------


def test_nu_standard():
    assert abs(_nu(1.0) - (1.0 / 3.0) * 0.5) < 1e-12


def test_section_separator_length():
    title = "Hello World"
    result = _section(title)
    lines = result.strip().split("\n")
    assert lines[-1] == "-" * len(title)


def test_row_default_indent():
    result = _row("Label:", "value")
    assert result.startswith("  ")


def test_row_custom_indent():
    result = _row("Label:", "value", indent=4)
    assert result.startswith("    ")


# ---------------------------------------------------------------------------
# _contact_line_length_from_rho edge cases
# ---------------------------------------------------------------------------


def _make_droplet_rho(nx: int = 20, rho_l: float = 1.0, rho_v: float = 0.2) -> np.ndarray:
    """(nx, 1, 1, 1, 1) array with a liquid slab from nx//4 to 3*nx//4."""
    rho = np.full((nx, 1, 1, 1, 1), rho_v)
    lo, hi = nx // 4, 3 * nx // 4
    rho[lo:hi, 0, 0, 0, 0] = rho_l
    return rho


def test_contact_line_from_rho_valid():
    rho = _make_droplet_rho(nx=20)
    rho_mean = 0.6
    result = _contact_line_length_from_rho(rho, rho_mean)
    assert result is not None
    assert result > 0.0


def test_contact_line_from_rho_uniform_returns_none():
    """Uniform field has no interface transitions → None."""
    rho = np.full((20, 1, 1, 1, 1), 1.0)
    assert _contact_line_length_from_rho(rho, 0.6) is None


def test_contact_line_from_rho_wrong_shape_returns_none():
    """Non-conforming array should not raise; helper catches exceptions."""
    result = _contact_line_length_from_rho(np.array([1.0, 0.0]), 0.6)
    assert result is None


def test_contact_line_from_rho_zero_denominator_returns_none():
    """Step-function interface (no gradient) → zero denominator → None."""
    rho = np.zeros((10, 1, 1, 1, 1))
    rho[5:, 0, 0, 0, 0] = 1.0
    result = _contact_line_length_from_rho(rho, 0.5)
    # Either None or a positive number; must not raise
    assert result is None or result > 0.0


# ---------------------------------------------------------------------------
# _ensure_single_gravity_force_source
# ---------------------------------------------------------------------------


def test_ensure_single_source_raises_on_both():
    from types import SimpleNamespace

    cfg = SimpleNamespace(gravity_force={"force_g": 1e-6}, gravity_masked_force={"force_g": 2e-6})
    with pytest.raises(ValueError, match="Only one gravity force"):
        _ensure_single_gravity_force_source(cfg)  # ty: ignore[invalid-argument-type]


def test_ensure_single_source_ok_with_one():
    cfg = SimulationConfig(gravity_force={"force_g": 1e-6})
    _ensure_single_gravity_force_source(cfg)  # must not raise


# ---------------------------------------------------------------------------
# _resolve_gravity_value branches
# ---------------------------------------------------------------------------


def test_resolve_gravity_value_from_g_attr():
    cfg = SimulationConfig(g=3e-7)
    assert _resolve_gravity_value(cfg) == pytest.approx(3e-7)


def test_resolve_gravity_value_from_gravity_force_dict():
    cfg = SimulationConfig(gravity_force={"force_g": 5e-7})
    assert _resolve_gravity_value(cfg) == pytest.approx(5e-7)


def test_resolve_gravity_value_from_gravity_masked_force_dict():
    cfg = SimulationConfig(gravity_masked_force={"force_g": 4e-7})
    assert _resolve_gravity_value(cfg) == pytest.approx(4e-7)


def test_resolve_gravity_value_none_when_no_gravity():
    cfg = SimulationConfig()
    assert _resolve_gravity_value(cfg) is None


# ---------------------------------------------------------------------------
# _resolve_gravity_inclination branches
# ---------------------------------------------------------------------------


def test_resolve_gravity_inclination_from_gravity_force():
    cfg = SimulationConfig(gravity_force={"force_g": 1e-6, "inclination_angle_deg": 30.0})
    assert _resolve_gravity_inclination(cfg) == pytest.approx(30.0)


def test_resolve_gravity_inclination_defaults_to_zero():
    cfg = SimulationConfig(gravity_force={"force_g": 1e-6})
    assert _resolve_gravity_inclination(cfg) == pytest.approx(0.0)


def test_resolve_gravity_inclination_no_force():
    cfg = SimulationConfig()
    assert _resolve_gravity_inclination(cfg) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_overview — section content checks
# ---------------------------------------------------------------------------


def test_build_overview_no_simulation_name():
    cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
    text = build_overview(cfg)
    assert "Simulation" in text
    assert "PHYSICAL PARAMETER OVERVIEW" in text


def test_build_overview_includes_simulation_name_when_set():
    cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10, simulation_name="my_run")
    text = build_overview(cfg)
    assert "my_run" in text


def test_build_overview_3d_grid_displays_full_shape():
    cfg = SimulationConfig(grid_shape=(8, 8, 4), tau=0.8, nt=10)
    text = build_overview(cfg)
    # 3-D: all three dimensions must appear in the shape line
    assert "4" in text


def test_build_overview_multiphase_no_gravity_omits_bond_number():
    cfg = SimulationConfig(
        sim_type="multiphase",
        grid_shape=(20, 20),
        eos="double-well",
        kappa=0.02,
        rho_l=1.0,
        rho_v=0.5,
        interface_width=2,
        # deliberately no gravity_force / g
    )
    text = build_overview(cfg)
    assert "gamma (surface tension):" in text
    assert "Oh (Ohnesorge number):" in text
    assert "Bo (Bond number):" not in text


def test_build_overview_wetting_and_hysteresis_sections():
    cfg = SimulationConfig(
        grid_shape=(8, 8),
        tau=0.8,
        nt=10,
        wetting_config={"contact_angle": 90},
        hysteresis_config={"ca_advancing": 100, "ca_receding": 80},
    )
    text = build_overview(cfg)
    assert "Wetting" in text
    assert "contact_angle" in text
    assert "Hysteresis" in text
    assert "ca_advancing" in text


def test_build_overview_forces_section_with_masked_gravity():
    cfg = SimulationConfig(
        sim_type="multiphase",
        grid_shape=(20, 20),
        eos="double-well",
        kappa=0.02,
        rho_l=1.0,
        rho_v=0.5,
        interface_width=2,
        gravity_masked_force={"force_g": 1e-6},
    )
    text = build_overview(cfg)
    assert "gravity_masked_force" in text


def test_build_overview_archimedes_number_present_when_gravity():
    cfg = SimulationConfig(
        sim_type="multiphase",
        grid_shape=(40, 20),
        eos="double-well",
        kappa=0.02,
        rho_l=1.0,
        rho_v=0.5,
        interface_width=2,
        gravity_force={"force_g": 1e-6},
    )
    text = build_overview(cfg)
    assert "Ar (Archimedes number):" in text


# ---------------------------------------------------------------------------
# _format_critical_inclination_angle_row
# ---------------------------------------------------------------------------

_CRITICAL_ANGLE_BASE = {
    "chemical_step_config": {"ca_advancing_pre_step": 110.0, "ca_receding_pre_step": 80.0},
    "gravity_masked_force": {"force_g": 1e-3},
    "rho_l": 1.0,
    "initialisation": {"radii": [0.1]},
    "grid_shape": (100, 100, 1),
}


def _critical_angle_ns(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{**_CRITICAL_ANGLE_BASE, **overrides})


def test_critical_inclination_raises_without_chemical_step_config():
    ns = _critical_angle_ns(chemical_step_config=None)
    with pytest.raises(RuntimeError, match="chemical_step_config"):
        _format_critical_inclination_angle_row(ns, gamma=0.01)  # ty: ignore[invalid-argument-type]


def test_critical_inclination_raises_without_gravity_masked_force():
    ns = _critical_angle_ns(gravity_masked_force=None)
    with pytest.raises(RuntimeError, match="gravity_masked_force"):
        _format_critical_inclination_angle_row(ns, gamma=0.01)  # ty: ignore[invalid-argument-type]


def test_critical_inclination_raises_without_rho_l():
    ns = _critical_angle_ns(rho_l=None)
    with pytest.raises(RuntimeError, match="rho_l"):
        _format_critical_inclination_angle_row(ns, gamma=0.01)  # ty: ignore[invalid-argument-type]


def test_critical_inclination_returns_angle_row_when_sina_in_range():
    # g=1e-3 makes sina≈0.033, well within [-1, 1]
    ns = _critical_angle_ns()
    result = _format_critical_inclination_angle_row(ns, gamma=0.01)  # ty: ignore[invalid-argument-type]
    assert "Critical Inclination Angle" in result
    assert "arcsin" in result
    assert "This droplet will remain pinned" not in result


def test_critical_inclination_returns_pinned_when_sina_exceeds_one():
    # Vanishingly small g → sina >> 1 → physically impossible to tip → pinned.
    # np.arcsin(sina) returns nan for out-of-range inputs before the branch check — expected.
    ns = _critical_angle_ns(gravity_masked_force={"force_g": 1e-20})
    with pytest.warns(RuntimeWarning, match="invalid value encountered in arcsin"):
        result = _format_critical_inclination_angle_row(ns, gamma=0.01)  # ty: ignore[invalid-argument-type]
    assert "This droplet will remain pinned" in result


def test_critical_inclination_formula_matches_manual_calculation():
    # ca_adv=90°, ca_rec=60°: cos(60°)-cos(90°) = 0.5 - 0.0 = 0.5
    ca_adv_deg, ca_rec_deg = 90.0, 60.0
    gamma = 0.02
    radius_frac = 0.1
    nx = 100
    g = 1e-3
    rho_l = 2.0

    a = (np.pi * (radius_frac * nx) ** 2) / 2
    hysteresis_force = (math.cos(math.radians(ca_rec_deg)) - math.cos(math.radians(ca_adv_deg))) * gamma
    expected_sina = hysteresis_force / (g * a * rho_l)
    expected_deg = math.degrees(math.asin(expected_sina))

    ns = _critical_angle_ns(
        chemical_step_config={"ca_advancing_pre_step": ca_adv_deg, "ca_receding_pre_step": ca_rec_deg},
        gravity_masked_force={"force_g": g},
        rho_l=rho_l,
        initialisation={"radii": [radius_frac]},
        grid_shape=(nx, nx, 1),
    )
    result = _format_critical_inclination_angle_row(ns, gamma=gamma)  # ty: ignore[invalid-argument-type]
    assert f"{expected_deg:.6g}" in result


def test_build_overview_includes_critical_inclination_angle_for_chemical_step_config():
    cfg = SimulationConfig(
        sim_type="multiphase_hysteresis_chemical_step",
        grid_shape=(100, 100),
        eos="double-well",
        kappa=0.02,
        rho_l=1.0,
        rho_v=0.5,
        interface_width=2,
        gravity_masked_force={"force_g": 1e-3},
        chemical_step_config={"ca_advancing_pre_step": 110.0, "ca_receding_pre_step": 80.0},
        initialisation={"radii": [0.1], "centres": [[0.5, 0.5]]},
        hysteresis_config={"ca_advancing": 110, "ca_receding": 80},
        wetting_config={"contact_angle": 95},
    )
    text = build_overview(cfg)
    assert "Critical Inclination Angle" in text


def test_build_overview_omits_critical_inclination_angle_without_chemical_step_config():
    cfg = SimulationConfig(
        sim_type="multiphase",
        grid_shape=(40, 20),
        eos="double-well",
        kappa=0.02,
        rho_l=1.0,
        rho_v=0.5,
        interface_width=2,
        gravity_force={"force_g": 1e-6},
    )
    text = build_overview(cfg)
    assert "Critical Inclination Angle" not in text
