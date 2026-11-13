"""An omitted ``rho_l``/``rho_v`` is filled with the Maxwell pair of an EOS that has a coexistence solver."""

from __future__ import annotations
import pytest
from src.config.multiphase import multiphase_params
from src.simulation_io.readers import DictAdapter

_CS = {
    "eos": "carnahan-starling",
    "kappa": 0.015,
    "interface_width": 4,
    "a_eos": 0.00031459670905604266,
    "b_eos": 0.1490857142857143,
    "r_eos": 1.0,
    "t_eos": 0.00039808421247983624,
}
#: The Maxwell pair of _CS, rounded as the config rounds it.
_RHO_V, _RHO_L = 0.0168045, 12.1833


def _densities(section: dict) -> tuple[float, float]:
    config = DictAdapter().load({"grid_shape": (16, 16), "nt": 10, "multiphase": section})
    params = multiphase_params(config)
    assert params is not None
    return params.rho_v, params.rho_l


def test_omitted_densities_are_the_maxwell_pair():
    assert _densities(dict(_CS)) == (_RHO_V, _RHO_L)


def test_configured_densities_are_kept():
    assert _densities(_CS | {"rho_v": 0.015, "rho_l": 12.18}) == (0.015, 12.18)


def test_only_the_missing_density_is_filled():
    assert _densities(_CS | {"rho_v": 0.015}) == (0.015, _RHO_L)


def test_double_well_still_requires_both_densities():
    adapter = DictAdapter()
    raw = {
        "grid_shape": (16, 16),
        "multiphase": {"eos": "double-well", "kappa": 0.015, "rho_v": 0.1, "interface_width": 4},
    }
    with pytest.raises(ValueError, match="'rho_l' is required"):
        adapter.load(raw)


def test_supercritical_fluid_without_densities_raises():
    adapter = DictAdapter()
    raw = {"grid_shape": (16, 16), "multiphase": _CS | {"t_eos": 1.0}}
    with pytest.raises(ValueError, match="critical temperature"):
        adapter.load(raw)


# ``equilibrium_reference_density``: rho_0 of an equilibrium that carries a pressure.

_DW = {"eos": "double-well", "kappa": 0.02, "rho_l": 1.0, "rho_v": 0.1, "interface_width": 4}


def _reference_density(section: dict, equilibrium: str = "wb_hydro_inc") -> float | None:
    raw = {"grid_shape": (16, 16), "nt": 10, "equilibrium": equilibrium, "multiphase": section}
    params = multiphase_params(DictAdapter().load(raw))
    assert params is not None
    return params.equilibrium_reference_density


def test_the_reference_density_defaults_to_the_maxwell_vapour_density():
    # Also with a configured pair off coexistence: rho_0 follows the fluid, not the config.
    assert _reference_density(dict(_CS)) == _RHO_V
    assert _reference_density(_CS | {"rho_v": 0.015, "rho_l": 12.18}) == _RHO_V


def test_the_reference_density_of_a_double_well_is_its_vapour_density():
    assert _reference_density(dict(_DW)) == 0.1


def test_a_configured_reference_density_is_kept():
    assert _reference_density(_CS | {"equilibrium_reference_density": 0.02}) == 0.02


def test_the_default_equilibrium_has_no_reference_density():
    assert _reference_density(dict(_CS), equilibrium="wb") is None


def test_a_reference_density_without_its_equilibrium_is_rejected():
    adapter = DictAdapter()
    raw = {"grid_shape": (16, 16), "multiphase": _DW | {"equilibrium_reference_density": 0.1}}
    with pytest.raises(ValueError, match="needs an equilibrium that carries a pressure"):
        adapter.load(raw)


@pytest.mark.parametrize("rho_0", [0.0, -0.1, 1.0, 2.0])
def test_the_reference_density_lies_between_zero_and_the_liquid_density(rho_0):
    adapter = DictAdapter()
    raw = {
        "grid_shape": (16, 16),
        "equilibrium": "wb_hydro_inc",
        "multiphase": _DW | {"equilibrium_reference_density": rho_0},
    }
    with pytest.raises(ValueError, match="equilibrium_reference_density"):
        adapter.load(raw)


def test_a_single_phase_run_cannot_use_the_pressure_carrying_equilibrium():
    adapter = DictAdapter()
    raw = {"grid_shape": (16, 16), "equilibrium": "wb_hydro_inc"}
    with pytest.raises(ValueError, match="needs a \\[multiphase\\] section"):
        adapter.load(raw)
