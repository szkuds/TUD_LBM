"""Tests for the numeric dimensionless-number helpers in physical_parameters.py."""

from __future__ import annotations
import math
from src.config import SimulationConfig
from src.simulation_io.analysis.physical_parameters import build_overview
from src.simulation_io.analysis.physical_parameters import compute_dimensionless_numbers
from src.simulation_io.analysis.physical_parameters import dimensionless_keys
from src.simulation_io.analysis.physical_parameters import resolve_dimensionless_inputs
from src.simulation_io.analysis.physical_parameters._inputs import DimensionlessInputs
from src.simulation_io.analysis.physical_parameters.numbers._bond import BondNumbers
from src.simulation_io.analysis.physical_parameters.numbers._bond import compute_bond_numbers
from src.simulation_io.analysis.physical_parameters.numbers._buoyancy import archimedes_number
from src.simulation_io.analysis.physical_parameters.numbers._buoyancy import reynolds_number
from src.simulation_io.analysis.physical_parameters.numbers._ohnesorge import ohnesorge_number
from src.simulation_io.analysis.physical_parameters.physical_parameters import _nu
from tests.support.config import merge_overrides


def _mp_config(**kwargs) -> SimulationConfig:
    base = {
        "multiphase": {"eos": "double-well", "kappa": 0.02, "rho_l": 1.0, "rho_v": 0.5, "interface_width": 2},
        "forces": {"gravity_force": {"force_g": 1e-6}},
        "grid_shape": (40, 20),
    }
    merge_overrides(base, kwargs)
    return SimulationConfig(**base)  # ty: ignore[invalid-argument-type]


def test_ohnesorge_number_matches_hand_computed_value():
    cfg = _mp_config(multiphase={"rho_l": 1.0}, tau=0.8)
    inputs = resolve_dimensionless_inputs(cfg)
    assert inputs is not None

    oh = ohnesorge_number(inputs)

    expected = _nu(0.8) / (inputs.gamma * inputs.length * 1.0) ** 0.5
    assert oh is not None
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
    cfg = _mp_config(forces={"gravity_force": {"force_g": 1e-6, "inclination_angle_deg": 30.0}})

    dn = compute_dimensionless_numbers(cfg)

    assert all(dn.get(key) is not None for key in ("oh", "la", "bo", "bo_perp", "bo_parallel", "ar", "re"))
    bo_parallel = dn.get("bo_parallel")
    ar, re = dn.get("ar"), dn.get("re")
    assert bo_parallel is not None
    assert bo_parallel > 0.0
    assert ar is not None
    assert re is not None
    assert math.isclose(re, math.sqrt(ar), rel_tol=1e-12)
    assert dn.inclination_deg == 30.0


def test_only_the_gravity_driven_numbers_are_none_without_gravity():
    """Oh and La need no gravity, so a gravity-free run still reports them.

    Every number used to be dropped together, which contradicted the overview
    file — it printed the Oh row for exactly these configs.
    """
    cfg = _mp_config(forces={})

    dn = compute_dimensionless_numbers(cfg)

    assert dn.get("oh") is not None
    assert dn.get("la") is not None
    assert all(dn.get(key) is None for key in ("bo", "bo_perp", "bo_parallel", "ar", "re"))
    assert dn.inclination_deg is None


_CS_EOS = {"eos": "carnahan-starling", "a_eos": 1.0, "b_eos": 4.0, "r_eos": 1.0, "t_eos": 0.07}


def test_compute_dimensionless_numbers_all_none_for_calibration_only_eos_without_measurement():
    cfg = _mp_config(multiphase=_CS_EOS)

    dn = compute_dimensionless_numbers(cfg)

    assert all(dn.get(key) is None for key in dimensionless_keys())
    assert dn.inclination_deg is None


def test_compute_dimensionless_numbers_uses_measured_surface_tension():
    cfg = _mp_config(multiphase=_CS_EOS, extra={"surface_tension": 0.002})

    dn = compute_dimensionless_numbers(cfg)

    assert dn.get("oh") is not None
    assert dn.get("bo_parallel") is not None


def test_format_rows_unchanged_after_refactor():
    cfg = _mp_config(forces={"gravity_force": {"force_g": 1e-6, "inclination_angle_deg": 30.0}})

    text = build_overview(cfg)

    assert "Oh (Ohnesorge number):" in text
    assert "La (Laplace number):" in text
    assert "Bo (Bond number):" in text
    assert "Bo_perp (Bond normal):" in text
    assert "Bo_parallel (Bond tangential):" in text
    assert "Ar (Archimedes number):" in text
    assert "Re (Reynolds number):" in text


def test_reynolds_number_is_sqrt_of_archimedes():
    inputs = DimensionlessInputs(
        gamma=1e-3,
        gamma_source="analytical",
        drho=0.5,
        drho_source="config",
        length=10.0,
        length_label="L",
        nu=0.1,
        rho_l=1.0,
        g=1e-6,
        angle_deg=0.0,
    )

    ar = archimedes_number(inputs)
    re = reynolds_number(inputs)

    assert ar is not None
    assert re is not None
    assert math.isclose(re, math.sqrt(ar), rel_tol=1e-12)


def test_laplace_number_is_the_reciprocal_square_of_ohnesorge():
    """The identity that justifies keeping La out of automatic legend labels.

    Both are built from the same :class:`DimensionlessInputs`, so this holds by
    construction rather than by coincidence of parameters.
    """
    cfg = _mp_config(forces={"gravity_force": {"force_g": 1e-6, "inclination_angle_deg": 30.0}}, tau=0.8)

    dn = compute_dimensionless_numbers(cfg)

    la, oh = dn.get("la"), dn.get("oh")
    assert la is not None
    assert oh is not None
    assert math.isclose(la, 1.0 / oh**2, rel_tol=1e-12)
