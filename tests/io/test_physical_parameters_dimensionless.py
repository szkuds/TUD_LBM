"""Tests for the numeric dimensionless-number helpers in physical_parameters.py."""

from __future__ import annotations
import math
from tud_lbm.config import SimulationConfig
from tud_lbm.io.analysis.physical_parameters import BondNumbers
from tud_lbm.io.analysis.physical_parameters import build_overview
from tud_lbm.io.analysis.physical_parameters import compute_bond_numbers
from tud_lbm.io.analysis.physical_parameters import compute_dimensionless_numbers
from tud_lbm.io.analysis.physical_parameters import compute_ohnesorge_number
from tud_lbm.io.analysis.physical_parameters.physical_parameters import _nu
from tud_lbm.io.analysis.physical_parameters.physical_parameters import compute_archimedes_number
from tud_lbm.io.analysis.physical_parameters.physical_parameters import compute_reynolds_number


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


def test_compute_ohnesorge_number_matches_hand_computed_value():
    cfg = _mp_config(tau=0.8, rho_l=1.0)
    gamma = 0.001
    length = 10.0

    oh = compute_ohnesorge_number(cfg, gamma, length)

    nu = _nu(0.8)
    expected = nu / (gamma * length * 1.0) ** 0.5
    assert math.isclose(oh, expected, rel_tol=1e-12)


def test_compute_bond_numbers_matches_hand_computed_value():
    drho = 0.5
    gamma = 0.001
    g_val = 1e-6
    length = 10.0
    angle_deg = 30.0

    bn = compute_bond_numbers(drho, gamma, g_val, length, angle_deg)

    bo_expected = (drho * length**2 * g_val) / gamma
    assert isinstance(bn, BondNumbers)
    assert math.isclose(bn.bo, bo_expected, rel_tol=1e-12)
    assert math.isclose(bn.bo_perp, bo_expected * math.cos(math.radians(angle_deg)), rel_tol=1e-12)
    assert math.isclose(bn.bo_parallel, bo_expected * math.sin(math.radians(angle_deg)), rel_tol=1e-12)


def test_compute_bond_numbers_zero_angle_has_no_parallel_component():
    bn = compute_bond_numbers(0.5, 0.001, 1e-6, 10.0, angle_deg=0.0)
    assert math.isclose(bn.bo_parallel, 0.0, abs_tol=1e-15)
    assert math.isclose(bn.bo_perp, bn.bo, rel_tol=1e-12)


def test_compute_dimensionless_numbers_resolves_all_fields_when_inputs_available():
    cfg = _mp_config(gravity_masked_force=None, gravity_force={"force_g": 1e-6, "inclination_angle_deg": 30.0})

    dn = compute_dimensionless_numbers(cfg)

    assert dn.oh is not None
    assert dn.bo is not None
    assert dn.bo_perp is not None
    assert dn.bo_parallel is not None
    assert dn.bo_parallel > 0.0


def test_compute_dimensionless_numbers_all_none_without_gravity():
    cfg = _mp_config(gravity_force=None)

    dn = compute_dimensionless_numbers(cfg)

    assert dn.oh is None
    assert dn.bo is None
    assert dn.bo_perp is None
    assert dn.bo_parallel is None


_CS_EOS_PARAMS = {"a_eos": 1.0, "b_eos": 4.0, "r_eos": 1.0, "t_eos": 0.07}


def test_compute_dimensionless_numbers_all_none_for_calibration_only_eos_without_measurement():
    cfg = _mp_config(eos="carnahan-starling", **_CS_EOS_PARAMS)

    dn = compute_dimensionless_numbers(cfg)

    assert dn == (None, None, None, None)


def test_compute_dimensionless_numbers_uses_measured_surface_tension():
    cfg = _mp_config(eos="carnahan-starling", extra={"surface_tension": 0.002}, **_CS_EOS_PARAMS)

    dn = compute_dimensionless_numbers(cfg)

    assert dn.oh is not None
    assert dn.bo_parallel is not None


def test_format_rows_unchanged_after_refactor():
    cfg = _mp_config(gravity_force={"force_g": 1e-6, "inclination_angle_deg": 30.0})

    text = build_overview(cfg)

    assert "Oh (Ohnesorge number):" in text
    assert "Bo (Bond number):" in text
    assert "Bo_perp (Bond normal):" in text
    assert "Bo_parallel (Bond tangential):" in text
    assert "Ar (Archimedes number):" in text
    assert "Re (Reynolds number):" in text


def test_compute_reynolds_number_is_sqrt_of_archimedes():
    drho, g_val, length, nu, rho_l = 0.5, 1e-6, 10.0, 0.1, 1.0

    ar = compute_archimedes_number(drho, g_val, length, nu, rho_l)
    re = compute_reynolds_number(drho, g_val, length, nu, rho_l)

    assert math.isclose(re, math.sqrt(ar), rel_tol=1e-12)
