"""Unit tests for hysteresis-enabled multiphase step operator."""

from __future__ import annotations
from functools import partial
from types import SimpleNamespace
import jax.numpy as jnp
from tud_lbm.operators.step import _multiphase_hysteresis as mh
from tud_lbm.operators.wetting._params import WettingParams
from tud_lbm.pipeline.state import State
from tud_lbm.pipeline.state import WettingState


def _make_wetting_state() -> WettingState:
    return WettingState(
        phi_left=jnp.array(1.0),
        phi_right=jnp.array(2.0),
        d_rho_left=jnp.array(3.0),
        d_rho_right=jnp.array(4.0),
        ca_left=jnp.array(0.0),
        ca_right=jnp.array(0.0),
        cll_left=jnp.array(0.0),
        cll_right=jnp.array(0.0),
    )


def test_make_wetting_ops_forwards_live_wetting_params():
    """Closure operators should forward all wetting parameters to setup operators."""
    calls: dict[str, tuple[float, float, float, float]] = {}

    def grad_wetting(grid, phi_left, phi_right, d_rho_left, d_rho_right):
        calls["grad"] = (float(phi_left), float(phi_right), float(d_rho_left), float(d_rho_right))
        return grid + 1.0

    def lap_wetting(grid, phi_left, phi_right, d_rho_left, d_rho_right):
        calls["lap"] = (float(phi_left), float(phi_right), float(d_rho_left), float(d_rho_right))
        return grid - 1.0

    setup = SimpleNamespace(
        gradient_density_wetting=grad_wetting,
        laplacian_density_wetting=lap_wetting,
    )
    wetting = _make_wetting_state()

    grad, lap = mh._make_wetting_ops(setup, wetting)  # ty: ignore[invalid-argument-type]

    _ = grad(jnp.array(5.0))
    _ = lap(jnp.array(5.0))

    assert calls["grad"] == (1.0, 2.0, 3.0, 4.0)
    assert calls["lap"] == (1.0, 2.0, 3.0, 4.0)


def test_trial_step_defaults_to_two_steps_and_uses_last_rho(monkeypatch):
    """Default trial_steps=2 should select the last scan rho output."""
    grad_param_calls: list[tuple[float, float, float, float]] = []
    lap_param_calls: list[tuple[float, float, float, float]] = []

    def grad_wetting(grid, phi_left, phi_right, d_rho_left, d_rho_right):
        grad_param_calls.append((float(phi_left), float(phi_right), float(d_rho_left), float(d_rho_right)))
        return grid + phi_left + phi_right + d_rho_left + d_rho_right

    def lap_wetting(grid, phi_left, phi_right, d_rho_left, d_rho_right):
        lap_param_calls.append((float(phi_left), float(phi_right), float(d_rho_left), float(d_rho_right)))
        return grid - (phi_left + phi_right + d_rho_left + d_rho_right)

    def fake_pipeline(_setup, carry_f, _force_ext, grad, lap):
        probe = jnp.array(10.0)
        _ = grad(probe)
        _ = lap(probe)
        return carry_f + 1.0, carry_f + 100.0, jnp.array(-1.0), jnp.array(-2.0)

    monkeypatch.setattr(mh, "_multiphase_pipeline", fake_pipeline)

    setup = SimpleNamespace(
        config=SimpleNamespace(hysteresis_config={}),
        gradient_density_wetting=grad_wetting,
        laplacian_density_wetting=lap_wetting,
    )
    params = WettingParams(
        d_rho_left=jnp.array(3.0),
        d_rho_right=jnp.array(4.0),
        phi_left=jnp.array(1.0),
        phi_right=jnp.array(2.0),
    )

    f_out, rho = mh._trial_step(setup, jnp.array(1.0), jnp.array(9.0), params)  # ty: ignore[invalid-argument-type]

    assert float(f_out) == 3.0
    assert float(rho) == 102.0
    assert grad_param_calls
    assert lap_param_calls
    assert grad_param_calls[0] == (1.0, 2.0, 3.0, 4.0)
    assert lap_param_calls[0] == (1.0, 2.0, 3.0, 4.0)


def test_trial_step_single_step_branch(monkeypatch):
    """trial_steps=1 should select rho_out_all[0] branch."""

    def fake_pipeline(_setup, carry_f, _force_ext, _grad, _lap):
        return carry_f + 2.0, carry_f + 50.0, jnp.array(0.0), jnp.array(0.0)

    monkeypatch.setattr(mh, "_multiphase_pipeline", fake_pipeline)

    setup = SimpleNamespace(
        config=SimpleNamespace(hysteresis_config={"trial_steps": 1}),
        gradient_density_wetting=lambda *args: args[0],
        laplacian_density_wetting=lambda *args: args[0],
    )
    params = WettingParams(
        d_rho_left=jnp.array(0.0),
        d_rho_right=jnp.array(0.0),
        phi_left=jnp.array(0.0),
        phi_right=jnp.array(0.0),
    )

    f_out, rho = mh._trial_step(setup, jnp.array(1.0), jnp.array(0.0), params)  # ty: ignore[invalid-argument-type]

    assert float(f_out) == 3.0
    assert float(rho) == 51.0


def test_step_multiphase_hysteresis_new_state_and_trial_partial(monkeypatch):
    """Step should build new state and forward a correctly-bound trial_step_fn."""
    wetting = _make_wetting_state()
    state = State(
        f=jnp.array(1.0),
        rho=jnp.array(2.0),
        u=jnp.array(3.0),
        t=jnp.array(7),
        force=jnp.array(4.0),
        force_ext=jnp.array(5.0),
        wetting=wetting,
    )

    setup = SimpleNamespace(forces=("gravity",))
    force_ext = jnp.array(9.0)
    f_out = jnp.array(11.0)
    rho_out = jnp.array(12.0)
    u_out = jnp.array(13.0)
    force_tot = jnp.array(14.0)

    def fake_compute_total_force_ext(got_setup, got_state, got_forces):
        assert got_setup is setup
        assert got_state is state
        assert got_forces == ("gravity",)
        return force_ext, state

    def grad_sentinel(grid):
        return grid + 1.0

    def lap_sentinel(grid):
        return grid - 1.0

    def fake_make_wetting_ops(got_setup, got_wetting):
        assert got_setup is setup
        assert got_wetting is wetting
        return grad_sentinel, lap_sentinel

    def fake_multiphase_pipeline(got_setup, got_f, got_force_ext, grad, lap):
        assert got_setup is setup
        assert (
            float(got_f),
            float(got_force_ext),
            grad is grad_sentinel,
            lap is lap_sentinel,
        ) == (1.0, 9.0, True, True)
        return f_out, rho_out, u_out, force_tot

    trial_calls: list[tuple[object, object, object, object]] = []

    def fake_trial_step(arg_setup, arg_f_t, arg_force_ext, params):
        trial_calls.append((arg_setup, arg_f_t, arg_force_ext, params))
        return jnp.array(21.0), jnp.array(22.0)

    def fake_update_extra_state(got_setup, prev_state, new_state, **context):
        assert got_setup is setup
        assert prev_state is state
        assert (
            float(new_state.f),
            float(new_state.rho),
            float(new_state.u),
            float(new_state.force),
            float(new_state.force_ext),
            int(new_state.t),
        ) == (11.0, 12.0, 13.0, 14.0, 9.0, 8)
        assert float(context["force_ext"]) == 9.0
        trial_step_fn = context["trial_step_fn"]
        assert isinstance(trial_step_fn, partial)
        assert (trial_step_fn.func is fake_trial_step, trial_step_fn.args == (setup, f_out, force_ext)) == (True, True)

        marker_params = WettingParams(
            d_rho_left=jnp.array(30.0),
            d_rho_right=jnp.array(40.0),
            phi_left=jnp.array(10.0),
            phi_right=jnp.array(20.0),
        )
        trial_f, trial_rho = trial_step_fn(marker_params)
        assert (float(trial_f), float(trial_rho)) == (21.0, 22.0)
        assert trial_calls[-1] == (setup, f_out, force_ext, marker_params)

        return new_state._replace(t=jnp.array(99))

    monkeypatch.setattr(mh, "compute_total_force_ext", fake_compute_total_force_ext)
    monkeypatch.setattr(mh, "_make_wetting_ops", fake_make_wetting_ops)
    monkeypatch.setattr(mh, "_multiphase_pipeline", fake_multiphase_pipeline)
    monkeypatch.setattr(mh, "_trial_step", fake_trial_step)
    monkeypatch.setattr(mh, "update_extra_state", fake_update_extra_state)

    out_state = mh.step_multiphase_hysteresis(setup, state)  # ty: ignore[invalid-argument-type]

    assert int(out_state.t) == 99


# ── Guard branches ───────────────────────────────────────────────────


import pytest  # noqa: E402


class TestMakeWettingOpsGuards:
    """_make_wetting_ops raises TypeError when required operators are absent."""

    def test_raises_when_gradient_density_wetting_none(self):
        setup = SimpleNamespace(gradient_density_wetting=None, laplacian_density_wetting=lambda *_a: None)
        wetting = _make_wetting_state()
        with pytest.raises(TypeError, match="gradient_density_wetting is required"):
            mh._make_wetting_ops(setup, wetting)  # ty: ignore[invalid-argument-type]

    def test_raises_when_laplacian_density_wetting_none(self):
        setup = SimpleNamespace(gradient_density_wetting=lambda *_a: None, laplacian_density_wetting=None)
        wetting = _make_wetting_state()
        with pytest.raises(TypeError, match="laplacian_density_wetting is required"):
            mh._make_wetting_ops(setup, wetting)  # ty: ignore[invalid-argument-type]


class TestTrialStepGuards:
    """_trial_step raises TypeError when required setup fields are absent."""

    def test_raises_when_gradient_density_wetting_none(self):
        setup = SimpleNamespace(
            gradient_density_wetting=None,
            laplacian_density_wetting=lambda *_a: None,
            config=SimpleNamespace(hysteresis_config={}),
        )
        params = WettingParams(
            phi_left=jnp.array(0.0),
            phi_right=jnp.array(0.0),
            d_rho_left=jnp.array(0.0),
            d_rho_right=jnp.array(0.0),
        )
        f_t, force_ext = jnp.array(1.0), jnp.array(0.0)
        with pytest.raises(TypeError, match="gradient_density_wetting is required"):
            mh._trial_step(setup, f_t, force_ext, params)  # ty: ignore[invalid-argument-type]

    def test_raises_when_laplacian_density_wetting_none(self):
        setup = SimpleNamespace(
            gradient_density_wetting=lambda *_a: None,
            laplacian_density_wetting=None,
            config=SimpleNamespace(hysteresis_config={}),
        )
        params = WettingParams(
            phi_left=jnp.array(0.0),
            phi_right=jnp.array(0.0),
            d_rho_left=jnp.array(0.0),
            d_rho_right=jnp.array(0.0),
        )
        f_t, force_ext = jnp.array(1.0), jnp.array(0.0)
        with pytest.raises(TypeError, match="laplacian_density_wetting is required"):
            mh._trial_step(setup, f_t, force_ext, params)  # ty: ignore[invalid-argument-type]

    def test_raises_when_hysteresis_config_none(self):
        setup = SimpleNamespace(
            gradient_density_wetting=lambda *_a: None,
            laplacian_density_wetting=lambda *_a: None,
            config=SimpleNamespace(hysteresis_config=None),
        )
        params = WettingParams(
            phi_left=jnp.array(0.0),
            phi_right=jnp.array(0.0),
            d_rho_left=jnp.array(0.0),
            d_rho_right=jnp.array(0.0),
        )
        f_t, force_ext = jnp.array(1.0), jnp.array(0.0)
        with pytest.raises(TypeError, match="hysteresis_config is required"):
            mh._trial_step(setup, f_t, force_ext, params)  # ty: ignore[invalid-argument-type]


class TestStepHysteresisGuard:
    """step_multiphase_hysteresis raises TypeError when state.wetting is None."""

    def test_raises_when_wetting_none(self):
        state = State(
            f=jnp.array(1.0),
            rho=jnp.array(1.0),
            u=jnp.array(0.0),
            t=jnp.array(0),
            wetting=None,
        )
        setup = SimpleNamespace(forces=None)
        with pytest.raises(TypeError, match=r"state\.wetting is required"):
            mh.step_multiphase_hysteresis(setup, state)  # ty: ignore[invalid-argument-type]
