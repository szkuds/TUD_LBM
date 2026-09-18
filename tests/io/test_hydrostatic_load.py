"""The ``hs`` number: does this domain demand more of its liquid than the branch can carry?

The two reference rows are measured, not invented. A double-well regime-map run
survives indefinitely at Hs ≈ 0.51; a Carnahan-Starling bubble run at Hs ≈ 1.53
depleted its top wall from 11.83 to 5.09 and left its intended regime. Separating
those two is the whole job of this number, so both are pinned here.
"""

from __future__ import annotations
import pytest
from src.config import DictAdapter
from src.simulation_io.analysis.physical_parameters import build_overview
from src.simulation_io.analysis.physical_parameters import compute_dimensionless_numbers
from src.simulation_io.analysis.physical_parameters import resolve_dimensionless_inputs
from src.simulation_io.analysis.physical_parameters.numbers._hydrostatic import hydrostatic_load

#: The failing run: bubble_chem_step/2026-09-14/17-09-16, measured Hs = 1.53.
_CS_RUN: dict[str, object] = {
    "sim_type": "multiphase",
    "grid_shape": (201, 101),
    "tau": 0.99,
    "eos": "carnahan-starling",
    "kappa": 0.01,
    "rho_l": 12.18,
    "rho_v": 0.015,
    "interface_width": 5,
    "a_eos": 0.00031459670905604266,
    "b_eos": 0.1490857142857143,
    "r_eos": 1.0,
    "t_eos": 0.00039808421247983624,
    "gravity_force": {"force_g": 5e-6, "inclination_angle_deg": 60},
    "extra": {"surface_tension": 0.0725411},
}

#: A surviving regime-map run: 26_09_08_regime_map_infill, measured Hs = 0.51.
_DW_RUN: dict[str, object] = {
    "sim_type": "multiphase",
    "grid_shape": (201, 51),
    "tau": 0.99,
    "eos": "double-well",
    "kappa": 0.04,
    "rho_l": 1.0,
    "rho_v": 0.001,
    "interface_width": 5,
    "gravity_masked_force": {"force_g": 6.26e-6, "inclination_angle_deg": 60},
}


def _hs(**overrides: object) -> float | None:
    base = dict(_CS_RUN)
    base.update(overrides)
    return compute_dimensionless_numbers(DictAdapter().load(base)).get("hs")


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        pytest.param(_CS_RUN, 1.53, id="carnahan-starling-run-that-failed"),
        pytest.param(_DW_RUN, 0.51, id="double-well-run-that-survived"),
    ],
)
def test_reproduces_the_measured_runs(run, expected):
    """The pass/fail pair this number exists to tell apart.

    The ``Hs = 1`` threshold is the point of the number, so each value is also
    checked to fall on the side of it that the measured run landed on.
    """
    value = compute_dimensionless_numbers(DictAdapter().load(dict(run))).get("hs")

    assert value is not None
    assert value == pytest.approx(expected, rel=0.02)
    assert (value < 1.0) == (expected < 1.0)


def test_scales_linearly_with_gravity():
    """Capacity is a property of the EOS alone, so Hs carries all of g's dependence."""
    base = _hs(gravity_force={"force_g": 5e-6, "inclination_angle_deg": 60})
    half = _hs(gravity_force={"force_g": 2.5e-6, "inclination_angle_deg": 60})

    assert base is not None
    assert half == pytest.approx(base / 2.0, rel=1e-6)


def test_an_inclined_domain_demands_more_than_an_upright_one():
    """The along-slope term spans nx, which is the longer side here."""
    upright = _hs(gravity_force={"force_g": 5e-6, "inclination_angle_deg": 0})
    inclined = _hs(gravity_force={"force_g": 5e-6, "inclination_angle_deg": 60})

    assert upright is not None
    assert inclined is not None
    assert inclined > upright


def test_no_gravity_leaves_it_unresolved():
    assert _hs(gravity_force=None) is None


def test_a_liquid_density_at_the_eos_divergence_is_unresolved():
    """``b*rho_l = 4`` is the Carnahan-Starling pole: no liquid branch to integrate."""
    assert (
        _hs(rho_l=1.0, rho_v=0.5, a_eos=1.0, b_eos=4.0, r_eos=1.0, t_eos=0.07, extra={"surface_tension": 0.002}) is None
    )


def test_an_eos_without_a_registered_pressure_is_unresolved():
    """Membership of the ``pressure`` kind is the capability gate, and it may be absent."""
    inputs = resolve_dimensionless_inputs(DictAdapter().load(dict(_CS_RUN)))
    assert inputs is not None
    assert inputs.pressure is not None

    assert hydrostatic_load(inputs._replace(pressure=None)) is None


def test_the_overview_reports_the_row_only_with_gravity():
    with_gravity = build_overview(DictAdapter().load(dict(_CS_RUN)))
    without = build_overview(DictAdapter().load({**_CS_RUN, "gravity_force": None}))

    assert "Hs (hydrostatic load):" in with_gravity
    assert "Hs (hydrostatic load):" not in without
