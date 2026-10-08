"""Unit tests for the hysteresis optimiser in src/operators/wetting/_hysteresis.py.

Covers the pinned/advancing/receding regime and its anchor rule, the phi/d_rho
knob selection for both topologies, the per-side hyperparameters, the clamp,
the gradient masks, the snap-vs-carry start point, the Huber costs, and the
guard branches.
"""

from __future__ import annotations
from types import SimpleNamespace
import jax.numpy as jnp
import pytest
from src.operators.wetting import _hysteresis as hyst
from src.operators.wetting._hysteresis import REGIME_ADVANCING
from src.operators.wetting._hysteresis import REGIME_PINNED
from src.operators.wetting._hysteresis import REGIME_RECEDING
from src.operators.wetting._hysteresis import REGIME_SATURATED
from src.operators.wetting._hysteresis import _clamp_params
from src.operators.wetting._hysteresis import _cost_above
from src.operators.wetting._hysteresis import _cost_below
from src.operators.wetting._hysteresis import _cost_ca
from src.operators.wetting._hysteresis import _cost_cll
from src.operators.wetting._hysteresis import _import_optax
from src.operators.wetting._hysteresis import _initial_params
from src.operators.wetting._hysteresis import _liquid_is_advancing
from src.operators.wetting._hysteresis import _mask_left_d_rho
from src.operators.wetting._hysteresis import _mask_left_phi
from src.operators.wetting._hysteresis import _mask_right_d_rho
from src.operators.wetting._hysteresis import _mask_right_phi
from src.operators.wetting._hysteresis import _phi_is_active
from src.operators.wetting._hysteresis import _regime_code
from src.operators.wetting._hysteresis import _side_hyperparams
from src.operators.wetting._hysteresis import _side_regime
from src.operators.wetting._hysteresis import _update_wetting_state_impl
from src.operators.wetting._hysteresis import update_wetting_state
from src.operators.wetting._params import WettingParams
from src.pipeline.state import WettingState


def _wetting(**overrides: float | jnp.ndarray) -> WettingState:
    values = {
        "phi_left": 1.0,
        "phi_right": 1.0,
        "d_rho_left": 0.0,
        "d_rho_right": 0.0,
        "ca_left": 90.0,
        "ca_right": 90.0,
        "cll_left": 10.0,
        "cll_right": 30.0,
    } | overrides
    return WettingState(**{k: jnp.array(v) for k, v in values.items()})


def test_import_optax_succeeds_when_available():
    optax = _import_optax()
    assert optax is not None


def test_import_optax_raises_clear_message(mock_optax_missing):
    with pytest.raises(ImportError, match="pip install optax"):
        _import_optax()


# ---------------------------------------------------------------------------
# _side_regime — when may a contact line move, and does its anchor follow?
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("side", "ca", "cll", "expected_moving", "expected_advancing"),
    [
        # Window [60, 80], anchor at 10. Left CL expands in -x, right in +x.
        # Above the advancing bound: pinned only while contracting.
        ("left", 90.0, 9.0, True, True),
        ("left", 90.0, 11.0, False, False),  # receding at the advancing bound -> pinned
        ("right", 90.0, 11.0, True, True),
        ("right", 90.0, 9.0, False, False),
        # Below the receding bound: pinned only while expanding.
        ("left", 50.0, 11.0, True, False),
        ("left", 50.0, 9.0, False, False),  # advancing at the receding bound -> pinned
        ("right", 50.0, 9.0, True, False),
        ("right", 50.0, 11.0, False, False),
        # At the anchor: an out-of-window side still targets its bound.
        ("left", 90.0, 10.0, True, True),
        ("right", 50.0, 10.0, True, False),
        # In window: always pinned, whichever way it drifted.
        ("left", 70.0, 9.0, False, False),
        ("right", 70.0, 11.0, False, False),
    ],
)
def test_side_regime_truth_table(side, ca, cll, expected_moving, expected_advancing):
    moving, advancing = _side_regime(jnp.array(ca), jnp.array(cll), jnp.array(10.0), 80.0, 60.0, side=side)
    assert (bool(moving), bool(advancing)) == (expected_moving, expected_advancing)


@pytest.mark.parametrize(
    ("side", "cll", "expected_advancing"),
    [
        # Inverted window adv 50 < rec 100 (a line advancing onto a more wetting
        # surface at a chemical step): CA 70 exceeds both bounds, so the motion
        # decides which one it targets. At the anchor it advances.
        ("left", 9.0, True),
        ("left", 10.0, True),
        ("left", 11.0, False),
        ("right", 11.0, True),
        ("right", 9.0, False),
    ],
)
def test_side_regime_inverted_window_follows_the_motion(side, cll, expected_advancing):
    moving, advancing = _side_regime(jnp.array(70.0), jnp.array(cll), jnp.array(10.0), 50.0, 100.0, side=side)
    assert (bool(moving), bool(advancing)) == (True, expected_advancing)


def test_side_regime_rejects_unknown_side():
    with pytest.raises(ValueError, match="side must be"):
        _side_regime(jnp.array(90.0), jnp.array(1.0), jnp.array(0.0), 80.0, 60.0, side="middle")


@pytest.mark.parametrize(
    ("moving", "advancing", "saturated", "expected"),
    [
        (False, True, False, REGIME_PINNED),
        (True, True, False, REGIME_ADVANCING),
        (True, False, False, REGIME_RECEDING),
        (True, True, True, REGIME_SATURATED),
    ],
)
def test_regime_code(moving, advancing, saturated, expected):
    assert int(_regime_code(jnp.array(moving), jnp.array(advancing), jnp.array(saturated))) == expected


# ---------------------------------------------------------------------------
# _phi_is_active — exhaustive truth-table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pinned", "above_window", "forward_drift", "is_bubble", "expected"),
    [
        # ── Droplet: dispersed angle == liquid angle, so phi (more wetting)
        # lowers the reported angle.
        # moving above window → phi active regardless of drift
        (False, True, False, False, True),
        (False, True, True, False, True),
        # pinned & ~forward_drift → phi active (liquid receding)
        (True, False, False, False, True),
        # pinned & forward_drift → d_rho active (liquid advancing)
        (True, False, True, False, False),
        # moving below window → d_rho active
        (False, False, False, False, False),
        (False, False, True, False, False),
        # ── Bubble: dispersed angle == 180 - liquid angle, so phi *raises* the
        # reported angle. Both contact-angle branches invert.
        (False, True, False, True, False),
        (False, True, True, True, False),
        (False, False, False, True, True),
        (False, False, True, True, True),
        # The pinning branches do NOT invert — forward_drift is already
        # liquid-frame and phi/d_rho are liquid-frame knobs.
        (True, False, False, True, True),
        (True, False, True, True, False),
        # Pinned out of window (above, moving the wrong way): the pin branch
        # decides, not the angle.
        (True, True, False, False, True),
        (True, True, True, False, False),
    ],
)
def test_phi_is_active_truth_table(pinned, above_window, forward_drift, is_bubble, expected):
    result = bool(
        _phi_is_active(
            jnp.array(pinned),
            jnp.array(above_window),
            jnp.array(forward_drift),
            jnp.array(is_bubble),
        )
    )
    assert result == expected


@pytest.mark.parametrize("forward_drift", [True, False])
def test_phi_is_active_pinned_is_topology_independent(forward_drift):
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


def test_side_hyperparams_uses_defaults_when_not_urgent():
    lr, max_iter = _side_hyperparams(_HYPER_CFG, jnp.array(False))
    assert float(lr) == pytest.approx(0.01)
    assert int(max_iter) == 10


def test_side_hyperparams_uses_above_overrides_when_urgent():
    lr, max_iter = _side_hyperparams(_HYPER_CFG, jnp.array(True))
    assert float(lr) == pytest.approx(0.05)
    assert int(max_iter) == 40


def test_side_hyperparams_sides_are_independent():
    lr_l, it_l = _side_hyperparams(_HYPER_CFG, jnp.array(True))
    lr_r, it_r = _side_hyperparams(_HYPER_CFG, jnp.array(False))
    assert (float(lr_l), int(it_l)) != (float(lr_r), int(it_r))


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
        phi_right=jnp.array(2.5),
        d_rho_left=jnp.array(-0.1),
        d_rho_right=jnp.array(0.5),
    )
    clamped = _clamp_params(p, jnp.array(5.0))
    assert float(clamped.phi_left) == pytest.approx(1.0)
    assert float(clamped.phi_right) == pytest.approx(2.0)
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
    assert float(clamped.phi_left) == pytest.approx(1.5)
    assert float(clamped.phi_right) == pytest.approx(1.5)
    assert float(clamped.d_rho_left) == pytest.approx(0.15)
    assert float(clamped.d_rho_right) == pytest.approx(0.15)


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


@pytest.mark.parametrize(
    ("mask", "kept"),
    [
        (_mask_left_d_rho, "d_rho_left"),
        (_mask_left_phi, "phi_left"),
        (_mask_right_d_rho, "d_rho_right"),
        (_mask_right_phi, "phi_right"),
    ],
)
def test_mask_keeps_only_its_parameter(mask, kept):
    params = _make_params()
    m = mask(params)
    for name in WettingParams._fields:
        expected = float(getattr(params, name)) if name == kept else 0.0
        assert float(getattr(m, name)) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _initial_params — snap vs carry of the inactive knob
# ---------------------------------------------------------------------------


def _accumulated_wetting() -> WettingState:
    return _wetting(
        phi_left=1.03,
        phi_right=1.04,
        d_rho_left=0.09,
        d_rho_right=0.11,
        ca_left=100.0,
        ca_right=120.0,
        cll_left=28.0,
        cll_right=79.0,
    )


def test_initial_params_snaps_inactive_knob_to_neutral_by_default():
    # Left: phi active, so d_rho is snapped; right: d_rho active, so phi is snapped.
    p = _initial_params(_accumulated_wetting(), jnp.array(True), jnp.array(False), carry_inactive=False)
    assert float(p.phi_left) == pytest.approx(1.03)
    assert float(p.d_rho_left) == 0.0
    assert float(p.phi_right) == 1.0
    assert float(p.d_rho_right) == pytest.approx(0.11)


@pytest.mark.parametrize("phi_active", [True, False])
def test_initial_params_carry_keeps_both_knobs_whichever_is_active(phi_active):
    wetting = _accumulated_wetting()
    p = _initial_params(wetting, jnp.array(phi_active), jnp.array(phi_active), carry_inactive=True)
    assert p == WettingParams(
        phi_left=wetting.phi_left,
        phi_right=wetting.phi_right,
        d_rho_left=wetting.d_rho_left,
        d_rho_right=wetting.d_rho_right,
    )


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
# _side_cost — a pinned side also keeps its angle inside the window
# ---------------------------------------------------------------------------


def _pinned_cost(ca: float, cll: float) -> float:
    return float(
        hyst._side_cost(
            jnp.array(ca),
            jnp.array(cll),
            anchor=jnp.array(10.0),
            moving=jnp.array(False),
            advancing=jnp.array(False),
            ca_adv=jnp.array(80.0),
            ca_rec=jnp.array(60.0),
        )
    )


def test_pinned_in_window_is_the_pin_alone():
    assert _pinned_cost(70.0, 10.3) == pytest.approx(float(_cost_cll(jnp.array(10.0), jnp.array(10.3))))


@pytest.mark.parametrize("ca", [56.0, 88.0])
def test_pinned_out_of_window_is_penalised_even_on_its_anchor(ca):
    """The logged failure: pinned on its anchor, the angle ran past its bound at zero loss."""
    assert _pinned_cost(ca, 10.0) > 0.0


# ---------------------------------------------------------------------------
# update_wetting_state through a synthetic trial step
# ---------------------------------------------------------------------------
#
# The fake "density" is the measurement itself: [ca_left, ca_right, cll_left,
# cll_right]. The trial step lowers each side's angle as phi rises, as it does
# for a droplet, and detect_bubble is pinned to "droplet".


def _fake_measurements(monkeypatch):
    monkeypatch.setattr(hyst, "compute_contact_angle", lambda rho, _mean, edge: (rho[0], rho[1]))
    monkeypatch.setattr(hyst, "compute_contact_line_location", lambda rho, _ca_l, _ca_r, _mean, edge: (rho[2], rho[3]))
    monkeypatch.setattr(hyst, "detect_bubble", lambda rho, _mean, edge: jnp.array(False))


def _fake_setup() -> SimpleNamespace:
    hc = {
        "ca_advancing": 80.0,
        "ca_receding": 60.0,
        "learning_rate": 0.01,
        "learning_rate_above": 0.01,
        "max_iterations": 3,
        "max_iterations_above": 3,
        "loss_tol": 1e-4,
        "carry_inactive_params": False,
        "saturation_gap": 1.0,
    }
    return SimpleNamespace(
        multiphase_params=SimpleNamespace(rho_l=1.0, rho_v=0.1, interface_width=5),
        wetting_edge="bottom",
        config=SimpleNamespace(hysteresis_config=hc),
    )


def _trial(cll_left: float, cll_right: float):
    def trial_step_fn(p: WettingParams) -> tuple[jnp.ndarray, jnp.ndarray]:
        ca_l = 90.0 - 100.0 * (p.phi_left - 1.0) + 300.0 * p.d_rho_left
        ca_r = 70.0 - 100.0 * (p.phi_right - 1.0) + 300.0 * p.d_rho_right
        return jnp.array(0.0), jnp.stack([ca_l, ca_r, jnp.array(cll_left), jnp.array(cll_right)])

    return trial_step_fn


def test_advancing_side_targets_its_bound_and_its_anchor_follows(monkeypatch):
    _fake_measurements(monkeypatch)
    measured = jnp.array([90.0, 70.0, 9.0, 30.0])  # left above 80 and expanding (9 < anchor 10)

    out = update_wetting_state(
        _wetting(phi_left=1.05),
        measured,
        _fake_setup(),  # ty: ignore[invalid-argument-type]
        trial_step_fn=_trial(9.0, 30.0),
    )

    assert float(out.phi_left) > 1.05  # phi lowers a droplet's angle toward 80
    assert float(out.cll_left) == 9.0  # the anchor follows an allowed advance
    # The in-window right side is pinned on its anchor, already met: untouched.
    assert (float(out.phi_right), float(out.d_rho_right), float(out.cll_right)) == (1.0, 0.0, 30.0)


def test_plain_hysteresis_never_saturates(monkeypatch):
    """Saturation is a chemical-step rule: far above the bound, the plain operator still optimises."""
    _fake_measurements(monkeypatch)
    measured = jnp.array([95.0, 70.0, 9.0, 30.0])  # left 15 above 80, expanding

    out = update_wetting_state(
        _wetting(phi_left=1.05),
        measured,
        _fake_setup(),  # ty: ignore[invalid-argument-type]
        trial_step_fn=_trial(9.0, 30.0),
    )

    assert 1.05 < float(out.phi_left) < 2.0


# ---------------------------------------------------------------------------
# _step_saturation — the post surface starts at the bound
# ---------------------------------------------------------------------------

_W5 = jnp.array(5.0)
_PRE_PHI, _PRE_D_RHO = 1.0, 0.07


def _saturation(*, on_post, held, ca, moving, advancing, is_bubble=False, adv=50.0, rec=30.0):
    saturated, phi, d_rho = hyst._step_saturation(
        hyst.StepSide(jnp.array(on_post), jnp.array(held), _PRE_PHI, _PRE_D_RHO),
        ca=jnp.array(ca),
        ca_adv=jnp.array(adv),
        ca_rec=jnp.array(rec),
        moving=jnp.array(moving),
        advancing=jnp.array(advancing),
        is_bubble=jnp.array(is_bubble),
        gap=1.0,
        w=_W5,
    )
    return bool(saturated), float(phi), float(d_rho)


def test_front_on_post_far_above_its_bound_takes_full_phi():
    assert _saturation(on_post=True, held=False, ca=60.0, moving=True, advancing=True) == (True, 2.0, 0.0)


def test_front_within_one_degree_goes_back_to_the_optimiser():
    assert _saturation(on_post=True, held=False, ca=50.8, moving=True, advancing=True)[0] is False


def test_front_still_on_the_pre_surface_is_not_saturated():
    assert _saturation(on_post=False, held=False, ca=60.0, moving=True, advancing=True)[0] is False


def test_rear_held_on_the_pre_side_keeps_the_pre_surface_pushing():
    """Its own band cells are pre-step cells: they take the configured values, not phi."""
    assert _saturation(on_post=False, held=True, ca=99.0, moving=False, advancing=False, adv=120.0) == (
        True,
        _PRE_PHI,
        _PRE_D_RHO,
    )


def test_rear_held_on_the_post_side_pulls_with_full_phi():
    assert _saturation(on_post=True, held=True, ca=99.0, moving=False, advancing=False, adv=120.0) == (
        True,
        2.0,
        0.0,
    )


def test_rear_held_within_one_degree_of_the_receding_angle_is_optimised():
    assert _saturation(on_post=True, held=True, ca=30.5, moving=False, advancing=False, adv=120.0)[0] is False


def test_receding_below_its_bound_on_post_takes_full_d_rho():
    assert _saturation(on_post=True, held=False, ca=20.0, moving=True, advancing=False) == (True, 1.0, 0.3)


def test_bubble_flips_the_knob():
    assert _saturation(on_post=True, held=False, ca=60.0, moving=True, advancing=True, is_bubble=True) == (
        True,
        1.0,
        0.3,
    )


def test_receding_at_the_advancing_bound_stays_pinned(monkeypatch):
    _fake_measurements(monkeypatch)
    measured = jnp.array([90.0, 70.0, 11.0, 30.0])  # left above 80 but contracting (11 > anchor 10)

    out = update_wetting_state(
        _wetting(phi_left=1.2, d_rho_left=0.05),
        measured,
        _fake_setup(),  # ty: ignore[invalid-argument-type]
        trial_step_fn=_trial(11.0, 30.0),
    )

    assert float(out.cll_left) == 10.0  # the anchor does not ratchet with the receding line
    # Pinned: the objective is the constant CLL error, so phi (the knob that
    # resists a receding droplet) is selected, warm-started and left in place,
    # and d_rho is snapped to neutral.
    assert (float(out.phi_left), float(out.d_rho_left)) == pytest.approx((1.2, 0.0))


# ---------------------------------------------------------------------------
# TypeError guard branches
# ---------------------------------------------------------------------------


def _dummy_trial_fn(_p: WettingParams) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.array(0.0), jnp.array(0.0)


_MEASURED = (jnp.array(90.0), jnp.array(90.0), jnp.array(0.0), jnp.array(0.0))


class TestUpdateWettingStateGuards:
    """update_wetting_state raises TypeError when hysteresis_config is absent."""

    def test_raises_when_hysteresis_config_none(self):
        setup = SimpleNamespace(config=SimpleNamespace(hysteresis_config=None))
        with pytest.raises(TypeError, match="hysteresis_config is required"):
            update_wetting_state(
                _wetting(),
                jnp.ones((4, 4, 1, 1, 1)),
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
        ca_adv, ca_rec = jnp.array(110.0), jnp.array(85.0)
        with pytest.raises(TypeError, match="multiphase_params is required"):
            _update_wetting_state_impl(
                _wetting(),
                jnp.ones((4, 4, 1, 1, 1)),
                setup,  # ty: ignore[invalid-argument-type]
                _dummy_trial_fn,
                ca_adv_left=ca_adv,
                ca_rec_left=ca_rec,
                ca_adv_right=ca_adv,
                ca_rec_right=ca_rec,
                measured=_MEASURED,
            )

    def test_raises_when_hysteresis_config_none(self):
        setup = SimpleNamespace(
            multiphase_params=SimpleNamespace(rho_l=1.0, rho_v=0.33),
            config=SimpleNamespace(hysteresis_config=None),
        )
        ca_adv, ca_rec = jnp.array(110.0), jnp.array(85.0)
        with pytest.raises(TypeError, match="hysteresis_config is required"):
            _update_wetting_state_impl(
                _wetting(),
                jnp.ones((4, 4, 1, 1, 1)),
                setup,  # ty: ignore[invalid-argument-type]
                _dummy_trial_fn,
                ca_adv_left=ca_adv,
                ca_rec_left=ca_rec,
                ca_adv_right=ca_adv,
                ca_rec_right=ca_rec,
                measured=_MEASURED,
            )
