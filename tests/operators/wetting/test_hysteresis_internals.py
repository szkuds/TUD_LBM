"""Branch coverage for src/operators/wetting/hysteresis/hysteresis.py.

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
from src.operators.wetting._params import WettingParams
from src.operators.wetting.hysteresis.hysteresis import _clamp_params
from src.operators.wetting.hysteresis.hysteresis import _cost_above
from src.operators.wetting.hysteresis.hysteresis import _cost_below
from src.operators.wetting.hysteresis.hysteresis import _cost_ca
from src.operators.wetting.hysteresis.hysteresis import _cost_cll
from src.operators.wetting.hysteresis.hysteresis import _import_optax
from src.operators.wetting.hysteresis.hysteresis import _liquid_is_advancing
from src.operators.wetting.hysteresis.hysteresis import _mask_left_d_rho
from src.operators.wetting.hysteresis.hysteresis import _mask_left_phi
from src.operators.wetting.hysteresis.hysteresis import _mask_right_d_rho
from src.operators.wetting.hysteresis.hysteresis import _mask_right_phi
from src.operators.wetting.hysteresis.hysteresis import _phi_is_active
from src.operators.wetting.hysteresis.hysteresis import _side_hyperparams
from src.operators.wetting.hysteresis.hysteresis import _update_wetting_state_impl
from src.operators.wetting.hysteresis.hysteresis import update_wetting_state


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
    ("in_window", "above_window", "forward_drift", "is_bubble", "expected"),
    [
        # ── Droplet: dispersed angle == liquid angle, so phi (more wetting)
        # lowers the reported angle. These rows are the pre-topology behaviour
        # and must not change.
        # above_window → phi active regardless of drift
        (False, True, False, False, True),
        (False, True, True, False, True),
        # in_window & ~forward_drift → phi active (liquid receding)
        (True, False, False, False, True),
        # in_window & forward_drift → d_rho active (liquid advancing)
        (True, False, True, False, False),
        # below window → d_rho active
        (False, False, False, False, False),
        (False, False, True, False, False),
        # ── Bubble: dispersed angle == 180 - liquid angle, so phi *raises* the
        # reported angle. Both contact-angle branches invert.
        (False, True, False, True, False),
        (False, True, True, True, False),
        (False, False, False, True, True),
        (False, False, True, True, True),
        # The contact-line pinning branches do NOT invert — forward_drift is
        # already liquid-frame and phi/d_rho are liquid-frame knobs, so these
        # two rows match the droplet rows above.
        (True, False, False, True, True),
        (True, False, True, True, False),
    ],
)
def test_phi_is_active_truth_table(in_window, above_window, forward_drift, is_bubble, expected):
    result = bool(
        _phi_is_active(
            jnp.array(in_window),
            jnp.array(above_window),
            jnp.array(forward_drift),
            jnp.array(is_bubble),
        )
    )
    assert result == expected


@pytest.mark.parametrize("forward_drift", [True, False])
def test_phi_is_active_in_window_is_topology_independent(forward_drift):
    """Pinning resists whichever way the liquid moves, for either topology.

    ``_liquid_is_advancing`` has already converted the drift to the liquid
    frame, so applying ``is_bubble`` again here would double-invert it.
    """
    args = (jnp.array(True), jnp.array(False), jnp.array(forward_drift))
    droplet = bool(_phi_is_active(*args, jnp.array(False)))
    bubble = bool(_phi_is_active(*args, jnp.array(True)))
    assert droplet == bubble
    assert droplet != forward_drift


# ---------------------------------------------------------------------------
# _side_hyperparams — per-side learning rate and iteration budget
# ---------------------------------------------------------------------------


_HYPER_CFG = {
    "learning_rate": 0.01,
    "learning_rate_above": 0.05,
    "max_iterations": 10,
    "max_iterations_above": 40,
}


def test_side_hyperparams_uses_defaults_in_window():
    lr, max_iter = _side_hyperparams(_HYPER_CFG, jnp.array(False))
    assert float(lr) == pytest.approx(0.01)
    assert int(max_iter) == 10


def test_side_hyperparams_uses_above_overrides_above_window():
    lr, max_iter = _side_hyperparams(_HYPER_CFG, jnp.array(True))
    assert float(lr) == pytest.approx(0.05)
    assert int(max_iter) == 40


def test_side_hyperparams_sides_are_independent():
    """A side in-window keeps the default budget even when the other is above.

    The previous form OR-ed both sides' above-window flags into a single
    learning rate, so an in-window side inherited the aggressive step.
    """
    lr_in, iter_in = _side_hyperparams(_HYPER_CFG, jnp.array(False))
    lr_above, iter_above = _side_hyperparams(_HYPER_CFG, jnp.array(True))
    assert float(lr_in) < float(lr_above)
    assert int(iter_in) < int(iter_above)


def test_side_hyperparams_max_iterations_honoured_when_above_also_set():
    """Regression: the override used to win unconditionally.

    The old expression was a Python truthiness test, so any truthy
    ``max_iterations_above`` silently discarded ``max_iterations`` for both
    regimes.
    """
    _lr, max_iter = _side_hyperparams(_HYPER_CFG, jnp.array(False))
    assert int(max_iter) == _HYPER_CFG["max_iterations"]


def test_side_hyperparams_above_falls_back_to_max_iterations():
    cfg = {"max_iterations": 7}
    _lr, max_iter = _side_hyperparams(cfg, jnp.array(True))
    assert int(max_iter) == 7


def test_side_hyperparams_empty_config_uses_module_defaults():
    lr, max_iter = _side_hyperparams({}, jnp.array(False))
    assert float(lr) == pytest.approx(0.01)
    assert int(max_iter) == 50


# ---------------------------------------------------------------------------
# _liquid_is_advancing — the droplet/bubble drift inversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("side", "cll_now", "cll_stored", "is_bubble", "expected"),
    [
        # Droplet: the dispersed phase IS the liquid, so its expansion is the
        # liquid advancing. Left CL moves -x, right CL moves +x.
        ("left", 9.0, 10.0, False, True),
        ("left", 11.0, 10.0, False, False),
        ("right", 11.0, 10.0, False, True),
        ("right", 9.0, 10.0, False, False),
        # Bubble: the dispersed phase is the vapour, so the identical motion
        # grows the bubble and the liquid recedes — every case inverts.
        ("left", 9.0, 10.0, True, False),
        ("left", 11.0, 10.0, True, True),
        ("right", 11.0, 10.0, True, False),
        ("right", 9.0, 10.0, True, True),
    ],
)
def test_liquid_is_advancing_truth_table(side, cll_now, cll_stored, is_bubble, expected):
    result = _liquid_is_advancing(
        jnp.array(cll_now),
        jnp.array(cll_stored),
        jnp.array(is_bubble),
        side=side,
    )
    assert bool(result) == expected


def test_liquid_is_advancing_rejects_unknown_side():
    with pytest.raises(ValueError, match="side must be"):
        _liquid_is_advancing(jnp.array(1.0), jnp.array(0.0), jnp.array(False), side="middle")


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
    from src.pipeline.state import WettingState

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
