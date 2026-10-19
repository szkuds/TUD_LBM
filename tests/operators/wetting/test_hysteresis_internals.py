"""Rules of the hysteresis optimiser in src/operators/wetting/_hysteresis.py.

The regime and anchor rule, the phi/d_rho knob selection for both topologies,
the pinned-side cost, the chemical-step saturation, and ``update_wetting_state``
through a synthetic trial step.
"""

from __future__ import annotations
from types import SimpleNamespace
from typing import TYPE_CHECKING
import jax.numpy as jnp
import pytest
from src.operators.wetting import _hysteresis as hyst
from src.operators.wetting._hysteresis import _cost_cll
from src.operators.wetting._hysteresis import _import_optax
from src.operators.wetting._hysteresis import _phi_is_active
from src.operators.wetting._hysteresis import _side_regime
from src.operators.wetting._hysteresis import update_wetting_state
from src.pipeline.state import WettingState

if TYPE_CHECKING:
    from src.operators.wetting._params import WettingParams


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
    moving, advancing, _ = _side_regime(jnp.array(ca), jnp.array(cll), jnp.array(10.0), 80.0, 60.0, side=side)
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
    moving, advancing, _ = _side_regime(jnp.array(70.0), jnp.array(cll), jnp.array(10.0), 50.0, 100.0, side=side)
    assert (bool(moving), bool(advancing)) == (True, expected_advancing)


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
    """Pinning resists whichever way the liquid moves, for either topology (the drift is liquid-frame)."""
    args = (jnp.array(True), jnp.array(False), jnp.array(forward_drift))
    droplet = bool(_phi_is_active(*args, jnp.array(False)))
    bubble = bool(_phi_is_active(*args, jnp.array(True)))
    assert droplet == bubble
    assert droplet != forward_drift


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


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # Front on the post surface far above its bound: full phi.
        ({"on_post": True, "held": False, "ca": 60.0, "moving": True, "advancing": True}, (True, 2.0, 0.0)),
        # Within one degree: back to the optimiser.
        ({"on_post": True, "held": False, "ca": 50.8, "moving": True, "advancing": True}, False),
        # Still on the pre surface: not saturated.
        ({"on_post": False, "held": False, "ca": 60.0, "moving": True, "advancing": True}, False),
        # Rear held on the pre side keeps the pre surface pushing (its band cells are pre-step cells).
        (
            {"on_post": False, "held": True, "ca": 99.0, "moving": False, "advancing": False, "adv": 120.0},
            (True, _PRE_PHI, _PRE_D_RHO),
        ),
        # Rear held on the post side pulls with full phi.
        (
            {"on_post": True, "held": True, "ca": 99.0, "moving": False, "advancing": False, "adv": 120.0},
            (True, 2.0, 0.0),
        ),
        # Rear held within one degree of the receding angle: optimised.
        (
            {"on_post": True, "held": True, "ca": 30.5, "moving": False, "advancing": False, "adv": 120.0},
            False,
        ),
        # Receding below its bound on the post surface: full d_rho.
        ({"on_post": True, "held": False, "ca": 20.0, "moving": True, "advancing": False}, (True, 1.0, 0.3)),
        # A bubble flips the knob.
        (
            {"on_post": True, "held": False, "ca": 60.0, "moving": True, "advancing": True, "is_bubble": True},
            (True, 1.0, 0.3),
        ),
    ],
)
def test_step_saturation(kwargs, expected):
    result = _saturation(**kwargs)
    assert (result[0] if expected is False else result) == pytest.approx(expected)


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


def test_update_wetting_state_requires_a_hysteresis_config():
    setup = SimpleNamespace(config=SimpleNamespace(hysteresis_config=None))
    wetting, rho = _wetting(), jnp.ones((4, 4, 1, 1, 1))
    with pytest.raises(TypeError, match="hysteresis_config is required"):
        update_wetting_state(wetting, rho, setup, trial_step_fn=lambda p: (rho, rho))  # ty: ignore[invalid-argument-type]
