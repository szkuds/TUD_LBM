"""The wetting addon: its initial substate and which post_step the config sections select."""

from __future__ import annotations
from types import SimpleNamespace
from typing import Any
import jax.numpy as jnp
import pytest
from src.addons import _wetting
from src.config.simulation_config import SimulationConfig
from src.lattice.lattice import build_lattice

_MULTIPHASE: dict[str, Any] = {
    "multiphase": {"eos": "double-well", "kappa": 0.017, "rho_l": 1.0, "rho_v": 0.5, "interface_width": 4},
    "grid_shape": (32, 24),
    "tau": 0.99,
    "nt": 3,
    "boundary_conditions": {"bottom": "wetting"},
}
_WETTING: dict[str, Any] = {"phi_left": 1.2, "phi_right": 1.0, "d_rho_left": 0.0, "d_rho_right": 0.0}
_HYSTERESIS: dict[str, Any] = {"ca_advancing": 120.0, "ca_receding": 60.0, "learning_rate": 0.01, "max_iterations": 2}
_CHEMICAL_STEP: dict[str, Any] = {
    "chemical_step_location": 0.5,
    "ca_advancing_pre_step": 120.0,
    "ca_receding_pre_step": 60.0,
    "ca_advancing_post_step": 90.0,
    "ca_receding_post_step": 50.0,
}


@pytest.fixture
def lattice():
    return build_lattice("D2Q9")


@pytest.fixture
def fixed_measurements(monkeypatch):
    """Measurements that return known angles and lines, so init_state is checked in isolation."""
    monkeypatch.setattr(
        _wetting, "compute_contact_angle", lambda rho, rho_mean, **_: (jnp.array(75.0), jnp.array(85.0))
    )
    monkeypatch.setattr(
        _wetting,
        "compute_contact_line_location",
        lambda rho, ca_l, ca_r, rho_mean, **_: (jnp.array(10.0), jnp.array(30.0)),
    )


def _init_setup() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(hysteresis=None, init_type="standard"),
        initial_f_fn=lambda: jnp.ones((4, 4, 1, 9, 1)),
    )


def _addon(lattice, **sections):
    hooks = _wetting.build_wetting_addon(SimulationConfig(**_MULTIPHASE, **sections))
    assert hooks is not None
    return hooks


def test_inactive_without_a_wetting_section(lattice):
    config = SimulationConfig(**{k: v for k, v in _MULTIPHASE.items() if k != "boundary_conditions"})
    assert _wetting.build_wetting_addon(config) is None


def test_init_state_seeds_parameters_from_the_config_and_measures_the_angles(lattice, fixed_measurements):
    addon = _addon(lattice, wetting=_WETTING)
    wetting = addon.init_state(_init_setup())

    assert (float(wetting.phi_left), float(wetting.phi_right)) == (1.2, 1.0)
    assert (float(wetting.ca_left), float(wetting.ca_right)) == (75.0, 85.0)
    assert float(wetting.cll_right) == 30.0


def test_init_state_resumes_the_saved_wetting_state_on_a_hysteresis_restart(lattice, fixed_measurements, monkeypatch):
    saved = {
        "phi_left": 1.0,
        "phi_right": 1.75,
        "d_rho_left": 0.12,
        "d_rho_right": 0.0,
        "cll_left": 11.5,
        "cll_right": 29.0,
        "latch_left": 1.0,
        "stall_left": 3.0,
    }
    addon = _addon(lattice, wetting=_WETTING, hysteresis=_HYSTERESIS)
    monkeypatch.setattr(_wetting, "restored_wetting", lambda _config: saved)
    fresh = _addon(lattice, wetting=_WETTING).init_state(_init_setup())
    wetting = addon.init_state(_init_setup())

    for key, value in saved.items():
        assert float(getattr(wetting, key)) == value
    # The carry's types are fixed for the scan, whatever the snapshot stored.
    assert [value.dtype for value in wetting] == [value.dtype for value in fresh]
    assert (int(wetting.latch_right), int(wetting.stall_right)) == (0, 0)
    assert (float(wetting.ca_left), float(wetting.ca_right)) == (75.0, 85.0)  # re-measured, not restored


@pytest.mark.parametrize(
    ("sections", "expected"),
    [
        pytest.param({"hysteresis": _HYSTERESIS}, "update_wetting_state", id="hysteresis"),
        pytest.param(
            {
                "hysteresis": _HYSTERESIS,
                "chemical_step": {**_CHEMICAL_STEP, "chemical_step_pre_side": "low"},
                "initialisation": {"dispersed": "liquid"},
            },
            "update_wetting_state_chemical_step",
            id="chemical_step",
        ),
    ],
)
def test_the_config_sections_select_the_hysteresis_update(monkeypatch, sections, expected):
    """``sim_type`` selects nothing: [chemical_step] alone switches to the per-side windows."""
    from src.pipeline.runner import init_state
    from src.pipeline.setup import build_setup

    called: list[str] = []
    for name in ("update_wetting_state", "update_wetting_state_chemical_step"):

        def record(wetting, *_args, _name=name, **_kwargs):
            called.append(_name)
            return wetting

        monkeypatch.setattr(_wetting, name, record)

    setup = build_setup(SimulationConfig(**_MULTIPHASE, wetting=_WETTING, **sections))
    state = init_state(setup)
    post_step = dict(setup.addons)["wetting"].post_step
    assert post_step is not None
    post_step(setup, state, state)

    assert called == [expected]


def test_a_fixed_wetting_wall_only_remeasures():
    """Without [hysteresis] the post_step keeps the wall and updates the measured angles."""
    from src.pipeline.runner import init_state
    from src.pipeline.setup import build_setup

    setup = build_setup(SimulationConfig(**_MULTIPHASE, wetting=_WETTING))
    state = init_state(setup)
    seeded = state.ext["wetting"]._replace(ca_left=jnp.array(-1.0))
    state = state._replace(ext={"wetting": seeded})

    post_step = dict(setup.addons)["wetting"].post_step
    assert post_step is not None
    after = post_step(setup, state, state).ext["wetting"]

    assert float(after.phi_left) == float(seeded.phi_left)
    assert float(after.ca_left) != -1.0


def test_a_single_phase_run_cannot_carry_a_wetting_wall(lattice):
    config = SimulationConfig(grid_shape=(8, 8), boundary_conditions={"bottom": "wetting"}, wetting=_WETTING)
    with pytest.raises(ValueError, match="multiphase"):
        _wetting.build_wetting_addon(config)
