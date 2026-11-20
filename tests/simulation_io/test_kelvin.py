"""The Kelvin equation: the EOS-exact pair at a curved interface, and its first-order pressure shift.

The reference values are measured: the 2026-10-17 bubble R070 (cap radius 54.3)
held its vapour at 0.01175, and the largest calibration droplet (R 100) at
0.0203, against a Maxwell vapour density of 0.016804.
"""

from __future__ import annotations
import math
import pytest
from src.config import DictAdapter
from src.config.initialisation import DispersedPhase
from src.operators.macroscopic.eos import coexistence_densities
from src.operators.macroscopic.eos import kelvin_densities
from src.simulation_io.analysis.physical_parameters import build_overview
from src.simulation_io.analysis.physical_parameters import compute_dimensionless_numbers
from src.simulation_io.analysis.physical_parameters.physical_parameters import _cap_radius
from tests.support.config import merge_overrides

_CS = {
    "a_eos": 0.00031459670905604266,
    "b_eos": 0.1490857142857143,
    "r_eos": 1.0,
    "t_eos": 0.00039808421247983624,
}
_SIGMA = 0.0890473
_EOS = "carnahan-starling"

#: A half-disk bubble of radius 0.5 * 101 on the top wall of a 401x101 box.
_BUBBLE: dict[str, object] = {
    "multiphase": {"eos": _EOS, "kappa": 0.015, "rho_l": 12.18, "rho_v": 0.015, "interface_width": 5, **_CS},
    "boundary_conditions": {"top": "wetting", "bottom": "bounce-back", "left": "bounce-back", "right": "bounce-back"},
    "grid_shape": (401, 101),
    "tau": 0.99,
    "init_type": "multiphase_bubbles",
    "initialisation": {"centres": [[0.7, 1]], "radii": [0.5], "dispersed": "vapour"},
    "wetting": {"phi_left": 1.05, "phi_right": 1.05},
    "extra": {"surface_tension": _SIGMA},
}


def _bubble(**overrides: object):
    base = dict(_BUBBLE)
    merge_overrides(base, overrides)
    return DictAdapter().load(base)


def _kelvin(radius: float, dispersed: DispersedPhase) -> tuple[float, float] | None:
    return kelvin_densities(_EOS, _CS, sigma=_SIGMA, radius=radius, dispersed=dispersed)


def test_no_pressure_jump_is_the_maxwell_pair():
    assert coexistence_densities(_EOS, _CS, pressure_jump=0.0) == coexistence_densities(_EOS, _CS)


def test_a_bubble_holds_its_vapour_below_coexistence_as_measured():
    pair = _kelvin(54.3, DispersedPhase.VAPOUR)

    assert pair is not None
    assert pair[0] == pytest.approx(0.01175, rel=0.02)
    assert pair[1] < 12.1833  # the liquid around a bubble is at the lower pressure


def test_a_droplet_holds_its_vapour_above_coexistence_as_measured():
    pair = _kelvin(100.0, DispersedPhase.LIQUID)  # the largest calibration droplet

    assert pair is not None
    assert pair[0] == pytest.approx(0.0203, rel=0.03)
    assert pair[1] > 12.1833  # the liquid of a droplet is at the higher pressure


def test_the_shift_vanishes_for_a_flat_interface():
    rho_v, rho_l = coexistence_densities(_EOS, _CS) or (0.0, 0.0)

    pair = _kelvin(1e9, DispersedPhase.VAPOUR)

    assert pair == pytest.approx((rho_v, rho_l), rel=1e-6)


def test_an_inclusion_too_small_to_exist_has_no_pair():
    assert _kelvin(5.0, DispersedPhase.VAPOUR) is None


def test_an_eos_without_a_coexistence_solver_has_no_kelvin_pair():
    assert kelvin_densities("double-well", {}, sigma=0.01, radius=30.0, dispersed=DispersedPhase.VAPOUR) is None


@pytest.mark.parametrize("theta_deg", [60.0, 90.0, 110.0, 150.0])
def test_cap_radius_inverts_the_cap_geometry(theta_deg):
    theta, radius = math.radians(theta_deg), 37.0
    area = radius**2 * (theta - math.sin(theta) * math.cos(theta))

    assert _cap_radius(area, 2.0 * radius * math.sin(theta)) == pytest.approx(radius, rel=1e-9)


def _shift(config) -> float | None:
    return compute_dimensionless_numbers(config).get("kelvin_shift")


def test_the_shift_is_kelvins_first_order_term():
    """``gamma rho_v / ((rho_l - rho_v) P r)`` with the saturation pressure of the EOS, negative for a bubble."""
    rho_v, rho_l = coexistence_densities(_EOS, _CS) or (0.0, 0.0)
    p_sat = 6.6175e-6  # the Maxwell tie line of this fluid

    shift = _shift(_bubble())

    assert shift == pytest.approx(-_SIGMA * rho_v / ((rho_l - rho_v) * p_sat * 50.5), rel=1e-3)


def test_the_shift_agrees_with_the_exact_pair_to_its_own_order():
    """The exact vapour sits at ``ln(rho_v(r) / rho_v) ~ shift`` for a nearly ideal vapour."""
    rho_v, _ = coexistence_densities(_EOS, _CS) or (0.0, 0.0)
    pair = _kelvin(50.5, DispersedPhase.VAPOUR)  # the half disk's own radius
    shift = _shift(_bubble())

    assert shift is not None
    assert pair is not None
    assert shift == pytest.approx(math.log(pair[0] / rho_v), rel=0.1)


def test_the_shift_is_positive_for_a_droplet_and_negative_for_a_bubble():
    droplet = _shift(_bubble(initialisation={"centres": [[0.7, 1]], "radii": [0.5], "dispersed": "liquid"}))
    bubble = _shift(_bubble())

    assert droplet is not None
    assert bubble is not None
    assert droplet == pytest.approx(-bubble)
    assert droplet > 0.0


def test_the_shift_falls_with_the_radius():
    small = _shift(_bubble())
    large = _shift(
        _bubble(grid_shape=(801, 201), initialisation={"centres": [[0.7, 1]], "radii": [0.5], "dispersed": "vapour"})
    )

    assert small is not None
    assert large is not None
    assert large == pytest.approx(small * 50.5 / 100.5, rel=1e-6)


def test_a_double_well_has_no_saturation_pressure_to_compare_with():
    config = DictAdapter().load(
        {
            "multiphase": {"eos": "double-well", "kappa": 0.04, "rho_l": 1.0, "rho_v": 0.1, "interface_width": 5},
            "grid_shape": (201, 101),
            "initialisation": {"centres": [[0.5, 0.5]], "radii": [0.25]},
            "init_type": "multiphase_bubbles",
        }
    )

    assert _shift(config) is None


def test_the_overview_prints_the_kelvin_pair_and_warns_for_a_small_bubble():
    overview = build_overview(_bubble())

    assert "Kelvin shift (p_r-P)/P:" in overview
    assert "Kelvin densities (exact):" in overview
    assert "concave: bubble" in overview
    assert "WARNING: the vapour pressure is shifted by -0." in overview


def test_a_large_bubble_is_reported_without_the_warning():
    overview = build_overview(
        _bubble(grid_shape=(1201, 401), initialisation={"centres": [[0.7, 1]], "radii": [0.5], "dispersed": "vapour"})
    )

    assert "Kelvin densities (exact):" in overview
    assert "WARNING" not in overview
