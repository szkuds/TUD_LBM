"""Tests for wetting extra-state plugin wiring."""

from __future__ import annotations
from types import SimpleNamespace
import jax.numpy as jnp
from src.pipeline.state.state import State
from src.pipeline.state.state import WettingState


def _state_template(*, wetting: WettingState | None) -> State:
    return State(
        f=jnp.ones((4, 4, 1, 9, 1)),
        rho=jnp.ones((4, 4, 1, 1, 1)),
        u=jnp.zeros((4, 4, 1, 1, 2)),
        t=jnp.array(0),
        wetting=wetting,
    )


def test_cfg_value_reads_alias_and_default():
    from src.operators.wetting._extra_state import _cfg_value

    assert _cfg_value({"phi_l": 1.2}, "phi_left", "phi_l", default=1.0) == 1.2
    assert _cfg_value({}, "phi_left", "phi_l", default=1.0) == 1.0


def test_is_active_detects_wetting_or_hysteresis():
    from src.operators.wetting._extra_state import WettingExtraStatePlugin

    assert WettingExtraStatePlugin.is_active(SimpleNamespace(wetting_config={}))  # ty: ignore[invalid-argument-type]
    assert WettingExtraStatePlugin.is_active(SimpleNamespace(wetting_config=None, hysteresis_config={}))  # ty: ignore[invalid-argument-type]
    assert not WettingExtraStatePlugin.is_active(SimpleNamespace(wetting_config=None, hysteresis_config=None))  # ty: ignore[invalid-argument-type]


def test_init_state_uses_defaults_when_wetting_cfg_missing(monkeypatch):
    from src.operators.wetting import _extra_state as mod

    monkeypatch.setattr(mod, "compute_contact_angle", lambda rho, rho_mean, **_: (jnp.array(75.0), jnp.array(85.0)))
    monkeypatch.setattr(
        mod,
        "compute_contact_line_location",
        lambda rho, ca_l, ca_r, rho_mean, **_: (jnp.array(10.0), jnp.array(30.0)),
    )

    setup = SimpleNamespace(
        config=SimpleNamespace(wetting_config=None),
        initial_f_fn=lambda: jnp.ones((4, 4, 1, 9, 1)),
        multiphase_params=SimpleNamespace(rho_l=1.0, rho_v=0.5),
        wetting_edge="bottom",
    )

    out = mod.WettingExtraStatePlugin.init_state(setup)  # ty: ignore[invalid-argument-type]
    wet = out["wetting"]
    assert float(wet.phi_left) == 1.0
    assert float(wet.phi_right) == 1.0
    assert float(wet.ca_left) == 75.0
    assert float(wet.cll_right) == 30.0


def test_update_state_early_return_and_update_path():
    from src.operators.wetting._extra_state import WettingExtraStatePlugin

    new_state = _state_template(wetting=None)
    prev_state = _state_template(wetting=None)
    setup = SimpleNamespace(wetting_fn=None)

    assert WettingExtraStatePlugin.update_state(setup, prev_state, new_state) is new_state  # ty: ignore[invalid-argument-type]

    previous_wetting = WettingState(*(jnp.array(v) for v in (1, 1, 0, 0, 80, 81, 2, 3)))
    prev_state = _state_template(wetting=previous_wetting)
    new_state = _state_template(wetting=previous_wetting)

    calls = {}

    def _fake_wetting_fn(wetting, rho, setup_obj, trial_step_fn=None, t=None):
        calls["trial"] = trial_step_fn
        calls["t"] = t
        return wetting._replace(phi_left=jnp.array(1.5))

    marker = object()
    setup = SimpleNamespace(wetting_fn=_fake_wetting_fn)
    updated = WettingExtraStatePlugin.update_state(setup, prev_state, new_state, trial_step_fn=marker)  # ty: ignore[invalid-argument-type]

    assert updated.wetting is not None
    assert float(updated.wetting.phi_left) == 1.5
    assert calls["trial"] is marker
    # The timestep is forwarded so the wetting debug trace can stamp and
    # rate-limit its rows.
    assert int(calls["t"]) == int(new_state.t)


def test_init_state_raises_when_initial_f_fn_none():
    import pytest
    from src.operators.wetting._extra_state import WettingExtraStatePlugin

    setup = SimpleNamespace(
        config=SimpleNamespace(wetting_config=None),
        initial_f_fn=None,
        multiphase_params=SimpleNamespace(rho_l=1.0, rho_v=0.5),
    )
    with pytest.raises(TypeError, match="initial_f_fn is required"):
        WettingExtraStatePlugin.init_state(setup)  # ty: ignore[invalid-argument-type]
