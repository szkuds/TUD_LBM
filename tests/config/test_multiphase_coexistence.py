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
