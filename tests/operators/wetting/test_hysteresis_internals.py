"""Branch coverage for tud_lbm/operators/wetting/hysteresis/hysteresis.py.

Targets the 6 uncovered lines (76.9 → ~100%):
- _import_optax: ImportError path when optax is absent
- _phi_is_active: all four (in_window, above_window, forward_drift) combinations
- _clamp_params: verify clips are applied correctly
- Cost functions: _cost_cll, _cost_ca, _cost_above, _cost_below — both branches
  (|err| < delta and |err| >= delta)
- Gradient-mask helpers: all four mask functions
"""

from __future__ import annotations
from types import SimpleNamespace
import jax.numpy as jnp
import pytest
from tud_lbm.operators.wetting._params import WettingParams
from tud_lbm.operators.wetting.hysteresis.hysteresis import _clamp_params
from tud_lbm.operators.wetting.hysteresis.hysteresis import _cost_above
from tud_lbm.operators.wetting.hysteresis.hysteresis import _cost_below
from tud_lbm.operators.wetting.hysteresis.hysteresis import _cost_ca
from tud_lbm.operators.wetting.hysteresis.hysteresis import _cost_cll
from tud_lbm.operators.wetting.hysteresis.hysteresis import _import_optax
from tud_lbm.operators.wetting.hysteresis.hysteresis import _mask_left_d_rho
from tud_lbm.operators.wetting.hysteresis.hysteresis import _mask_left_phi
from tud_lbm.operators.wetting.hysteresis.hysteresis import _mask_right_d_rho
from tud_lbm.operators.wetting.hysteresis.hysteresis import _mask_right_phi
from tud_lbm.operators.wetting.hysteresis.hysteresis import _phi_is_active
from tud_lbm.operators.wetting.hysteresis.hysteresis import _update_wetting_state_impl
from tud_lbm.operators.wetting.hysteresis.hysteresis import update_wetting_state


def test_import_optax_succeeds_when_available():
    optax = _import_optax()
    assert optax is not None


def test_import_optax_raises_clear_message(mock_optax_missing):
    with pytest.raises(ImportError, match="pip install optax"):
        _import_optax()


# ---------------------------------------------------------------------------
# _phi_is_active — exhaustive truth-table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("in_window", "above_window", "forward_drift", "expected"),
    [
        # above_window=True → phi active regardless of others
        (False, True, False, True),
        (False, True, True, True),
        # in_window & ~forward_drift → phi active (CL drifting backward)
        (True, False, False, True),
        # in_window & forward_drift → d_rho active (CL advancing)
        (True, False, True, False),
        # below window → d_rho active
        (False, False, False, False),
        (False, False, True, False),
    ],
)
def test_phi_is_active_truth_table(in_window, above_window, forward_drift, expected):
    result = bool(
        _phi_is_active(
            jnp.array(in_window),
            jnp.array(above_window),
            jnp.array(forward_drift),
        )
    )
    assert result == expected


# ---------------------------------------------------------------------------
# _clamp_params
# ---------------------------------------------------------------------------


def test_clamp_params_clips_phi_below_minimum():
    p = WettingParams(
        phi_left=jnp.array(0.5),
        phi_right=jnp.array(2.0),
        d_rho_left=jnp.array(-0.1),
        d_rho_right=jnp.array(0.3),
    )
    clamped = _clamp_params(p, jnp.array(5.0))
    assert float(clamped.phi_left) == pytest.approx(1.0)
    assert float(clamped.phi_right) == pytest.approx(1.5)
    assert float(clamped.d_rho_left) == pytest.approx(0.0)
    assert float(clamped.d_rho_right) == pytest.approx(0.3)


def test_clamp_params_leaves_valid_values_unchanged():
    p = WettingParams(
        phi_left=jnp.array(1.2),
        phi_right=jnp.array(1.3),
        d_rho_left=jnp.array(0.1),
        d_rho_right=jnp.array(0.2),
    )
    clamped = _clamp_params(p, jnp.array(5.0))
    assert float(clamped.phi_left) == pytest.approx(1.2)
    assert float(clamped.phi_right) == pytest.approx(1.3)
    assert float(clamped.d_rho_left) == pytest.approx(0.1)
    assert float(clamped.d_rho_right) == pytest.approx(0.2)


def test_clamp_params_bounds_scale_with_interface_width():
    p = WettingParams(
        phi_left=jnp.array(2.0),
        phi_right=jnp.array(2.0),
        d_rho_left=jnp.array(1.0),
        d_rho_right=jnp.array(1.0),
    )
    clamped = _clamp_params(p, jnp.array(10.0))
    assert float(clamped.phi_left) == pytest.approx(1.25)
    assert float(clamped.phi_right) == pytest.approx(1.25)
    assert float(clamped.d_rho_left) == pytest.approx(0.15)
    assert float(clamped.d_rho_right) == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Cost functions — both Huber branches (quadratic and linear)
# ---------------------------------------------------------------------------


class TestCostCll:
    """Tests for the _cost_cll Huber loss function."""

    def test_small_error_quadratic_branch(self):
        # |err| = 0.1 < delta=0.5 → 0.5 * 0.01 = 0.005
        cost = float(_cost_cll(jnp.array(1.0), jnp.array(1.1)))
        assert cost == pytest.approx(0.5 * 0.1**2, rel=1e-5)

    def test_large_error_linear_branch(self):
        # |err| = 2.0 >= delta=0.5 → linear region: delta*(|err| - delta/2)
        cost = float(_cost_cll(jnp.array(0.0), jnp.array(2.0)))
        assert cost > 0  # cost is positive in linear regime


class TestCostCa:
    """Tests for the _cost_ca Huber loss function."""

    def test_small_error_quadratic(self):
        cost = float(_cost_ca(jnp.array(80.0), jnp.array(81.0)))
        assert cost == pytest.approx(0.5 * 1.0**2, rel=1e-5)

    def test_large_error_linear(self):
        # |err| = 20 deg >> delta=5 → linear regime
        cost = float(_cost_ca(jnp.array(80.0), jnp.array(100.0)))
        assert cost > 0  # just verify it doesn't blow up


class TestCostAbove:
    """Tests for the _cost_above one-sided Huber penalty."""

    def test_no_excess_returns_zero(self):
        # ca_current < ca_adv → excess = 0
        cost = float(_cost_above(jnp.array(90.0), jnp.array(80.0)))
        assert cost == pytest.approx(0.0)

    def test_small_excess_quadratic(self):
        cost = float(_cost_above(jnp.array(90.0), jnp.array(92.0)))
        assert cost == pytest.approx(0.5 * 2.0**2, rel=1e-5)

    def test_large_excess_linear(self):
        cost = float(_cost_above(jnp.array(90.0), jnp.array(110.0)))
        assert cost > 0


class TestCostBelow:
    """Tests for the _cost_below one-sided Huber penalty."""

    def test_no_deficit_returns_zero(self):
        # ca_current > ca_rec → deficit = 0
        cost = float(_cost_below(jnp.array(70.0), jnp.array(80.0)))
        assert cost == pytest.approx(0.0)

    def test_small_deficit_quadratic(self):
        cost = float(_cost_below(jnp.array(70.0), jnp.array(68.0)))
        assert cost == pytest.approx(0.5 * 2.0**2, rel=1e-5)

    def test_large_deficit_linear(self):
        cost = float(_cost_below(jnp.array(70.0), jnp.array(50.0)))
        assert cost > 0


# ---------------------------------------------------------------------------
# Gradient mask helpers — each should zero out the non-active components
# ---------------------------------------------------------------------------


def _make_params() -> WettingParams:
    return WettingParams(
        phi_left=jnp.array(1.1),
        phi_right=jnp.array(1.2),
        d_rho_left=jnp.array(0.05),
        d_rho_right=jnp.array(0.08),
    )


def test_mask_left_d_rho():
    m = _mask_left_d_rho(_make_params())
    assert float(m.phi_left) == pytest.approx(0.0)
    assert float(m.phi_right) == pytest.approx(0.0)
    assert float(m.d_rho_left) == pytest.approx(0.05)
    assert float(m.d_rho_right) == pytest.approx(0.0)


def test_mask_left_phi():
    m = _mask_left_phi(_make_params())
    assert float(m.phi_left) == pytest.approx(1.1)
    assert float(m.phi_right) == pytest.approx(0.0)
    assert float(m.d_rho_left) == pytest.approx(0.0)
    assert float(m.d_rho_right) == pytest.approx(0.0)


def test_mask_right_d_rho():
    m = _mask_right_d_rho(_make_params())
    assert float(m.phi_left) == pytest.approx(0.0)
    assert float(m.phi_right) == pytest.approx(0.0)
    assert float(m.d_rho_left) == pytest.approx(0.0)
    assert float(m.d_rho_right) == pytest.approx(0.08)


def test_mask_right_phi():
    m = _mask_right_phi(_make_params())
    assert float(m.phi_left) == pytest.approx(0.0)
    assert float(m.phi_right) == pytest.approx(1.2)
    assert float(m.d_rho_left) == pytest.approx(0.0)
    assert float(m.d_rho_right) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TypeError guard branches
# ---------------------------------------------------------------------------


def _dummy_wetting():
    from tud_lbm.pipeline.state import WettingState

    return WettingState(
        phi_left=jnp.array(0.0),
        phi_right=jnp.array(0.0),
        d_rho_left=jnp.array(0.0),
        d_rho_right=jnp.array(0.0),
        ca_left=jnp.array(0.0),
        ca_right=jnp.array(0.0),
        cll_left=jnp.array(0.0),
        cll_right=jnp.array(0.0),
    )


def _dummy_trial_fn(_p: WettingParams) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.array(0.0), jnp.array(0.0)


class TestUpdateWettingStateGuards:
    """update_wetting_state raises TypeError when hysteresis_config is absent."""

    def test_raises_when_hysteresis_config_none(self):
        setup = SimpleNamespace(config=SimpleNamespace(hysteresis_config=None))
        wetting = _dummy_wetting()
        rho = jnp.ones((4, 4, 1, 1, 1))
        with pytest.raises(TypeError, match="hysteresis_config is required"):
            update_wetting_state(
                wetting,
                rho,
                setup,  # ty: ignore[invalid-argument-type]
                trial_step_fn=_dummy_trial_fn,
            )


class TestUpdateWettingStateImplGuards:
    """_update_wetting_state_impl raises TypeError when multiphase_params or
    hysteresis_config are absent.
    """

    def test_raises_when_multiphase_params_none(self):
        setup = SimpleNamespace(
            multiphase_params=None,
            config=SimpleNamespace(hysteresis_config={"ca_advancing": 110.0, "ca_receding": 85.0}),
        )
        wetting = _dummy_wetting()
        rho = jnp.ones((4, 4, 1, 1, 1))
        ca_adv, ca_rec = jnp.array(110.0), jnp.array(85.0)
        with pytest.raises(TypeError, match="multiphase_params is required"):
            _update_wetting_state_impl(
                wetting,
                rho,
                setup,  # ty: ignore[invalid-argument-type]
                _dummy_trial_fn,
                ca_adv_left=ca_adv,
                ca_rec_left=ca_rec,
                ca_adv_right=ca_adv,
                ca_rec_right=ca_rec,
            )

    def test_raises_when_hysteresis_config_none(self):
        setup = SimpleNamespace(
            multiphase_params=SimpleNamespace(rho_l=1.0, rho_v=0.33),
            config=SimpleNamespace(hysteresis_config=None),
        )
        wetting = _dummy_wetting()
        rho = jnp.ones((4, 4, 1, 1, 1))
        ca_adv, ca_rec = jnp.array(110.0), jnp.array(85.0)
        with pytest.raises(TypeError, match="hysteresis_config is required"):
            _update_wetting_state_impl(
                wetting,
                rho,
                setup,  # ty: ignore[invalid-argument-type]
                _dummy_trial_fn,
                ca_adv_left=ca_adv,
                ca_rec_left=ca_rec,
                ca_adv_right=ca_adv,
                ca_rec_right=ca_rec,
            )
