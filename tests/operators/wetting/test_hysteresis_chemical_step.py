"""Tests for the chemical-step hysteresis window.

- _get_hysteresis_window_chemical_step: the surface's own window — post once a
  line is within ``edge_width`` of the step or past it — and the held-at-step
  flag for a line receding onto the more wetting surface
- update_wetting_state_chemical_step: registered operator lookup, guards, and
  the logged failure: a receding line just past the step stays pinned
"""

from __future__ import annotations
from types import SimpleNamespace
import jax.numpy as jnp
import pytest
from src.config.chemical_step import ChemicalStepWall
from src.operators.wetting import _hysteresis as hyst
from src.operators.wetting._hysteresis import _get_hysteresis_window_chemical_step
from src.operators.wetting._hysteresis import update_wetting_state_chemical_step

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The failing run's windows: a less wetting pre-step surface and a more wetting
# post-step one.
_CHEM_CFG = {
    "chemical_step_location": 0.5,  # step_x = 0.5 * 100 = 50
    "edge_width": 1.0,
    "ca_advancing_pre_step": 120.0,
    "ca_receding_pre_step": 100.0,
    "ca_advancing_post_step": 50.0,
    "ca_receding_post_step": 30.0,
}
_PRE = (120.0, 100.0)
_POST = (50.0, 30.0)


def _wall(nx: int = 100) -> ChemicalStepWall:
    return ChemicalStepWall(
        edge="bottom",
        step_x=0.5 * nx,
        pre_phi_left=1.0,
        pre_phi_right=1.0,
        pre_d_rho_left=0.07,
        pre_d_rho_right=0.07,
        post_phi=2.0,
        post_d_rho=0.0,
    )


def _make_setup(nx: int = 100) -> SimpleNamespace:
    config = SimpleNamespace(
        chemical_step_config=_CHEM_CFG,
        grid_shape=(nx, 50, 1),
        chemical_step_wall=_wall(nx),
    )
    return SimpleNamespace(config=config)


def _window(setup, cll: float, side: str) -> tuple[float, float]:
    ca_adv, ca_rec, _held = _get_hysteresis_window_chemical_step(setup, jnp.array(cll), side=side)
    return float(ca_adv), float(ca_rec)


def _held(setup, cll: float, side: str) -> bool:
    return bool(_get_hysteresis_window_chemical_step(setup, jnp.array(cll), side=side)[2])


# ---------------------------------------------------------------------------
# _get_hysteresis_window_chemical_step
# ---------------------------------------------------------------------------


class TestGetHysteresisWindowChemicalStep:
    """The surface's own ``(ca_adv, ca_rec)``, and whether the line is held at the step."""

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_away_from_the_step_is_that_surfaces_window(self, side):
        assert _window(_make_setup(), 20.0, side) == _PRE
        assert _window(_make_setup(), 80.0, side) == _POST

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("cll", [49.0, 49.5, 50.0, 50.8])
    def test_at_the_step_the_window_is_the_post_surfaces(self, cll, side):
        """No Gibbs-widened window: the barrier is the wall split, not the bounds."""
        assert _window(_make_setup(), cll, side) == _POST

    @pytest.mark.parametrize(
        ("cll", "side", "held"),
        [
            (49.5, "left", True),  # receding onto the wetting surface: held at the step
            (50.5, "left", True),
            (49.5, "right", False),  # advancing onto it: free to spread
            (20.0, "left", False),
            (80.0, "left", False),
        ],
    )
    def test_only_a_line_receding_onto_the_wetting_surface_is_held(self, cll, side, held):
        assert _held(_make_setup(), cll, side) is held

    def test_edge_band_is_edge_width_wide(self):
        setup = _make_setup()
        assert _window(setup, 48.9, "left") == _PRE
        assert _window(setup, 49.0, "left") == _POST

    def test_different_grid_size_scales_step_x(self):
        """step_x = location * nx, so a different nx moves the edge."""
        setup = _make_setup(nx=200)  # step_x = 100
        assert _window(setup, 80.0, "left") == _PRE
        assert _window(setup, 100.0, "left") == _POST
        assert _window(setup, 120.0, "left") == _POST

    def test_rejects_unknown_side(self):
        with pytest.raises(ValueError, match="side must be"):
            _window(_make_setup(), 20.0, "middle")


# ---------------------------------------------------------------------------
# update_wetting_state_chemical_step — registration check
# ---------------------------------------------------------------------------


def test_chemical_step_hysteresis_operator_is_registered():
    """update_wetting_state_chemical_step must be registered as 'chemical_step_hysteresis'."""
    from src.registry import get_operators

    ops = get_operators("wetting")
    assert "chemical_step_hysteresis" in ops, (
        f"Expected 'chemical_step_hysteresis' in wetting registry. Available: {sorted(ops.keys())}"
    )


# ---------------------------------------------------------------------------
# TypeError guard branches
# ---------------------------------------------------------------------------


def test_get_hysteresis_window_raises_when_chemical_step_config_none():
    setup = SimpleNamespace(config=SimpleNamespace(chemical_step_config=None))
    with pytest.raises(TypeError, match="chemical_step_config is required"):
        _window(setup, 20.0, "left")


def test_update_wetting_state_chemical_step_raises_when_multiphase_params_none():
    from src.pipeline.state import WettingState

    setup = SimpleNamespace(
        multiphase_params=None,
        config=SimpleNamespace(chemical_step_config=_CHEM_CFG, grid_shape=(100, 50, 1)),
    )
    dummy_wetting = WettingState(
        phi_left=jnp.array(0.0),
        phi_right=jnp.array(0.0),
        d_rho_left=jnp.array(0.0),
        d_rho_right=jnp.array(0.0),
        ca_left=jnp.array(0.0),
        ca_right=jnp.array(0.0),
        cll_left=jnp.array(0.0),
        cll_right=jnp.array(0.0),
    )
    rho = jnp.ones((4, 4, 1, 1, 1))
    with pytest.raises(TypeError, match="multiphase_params is required"):
        update_wetting_state_chemical_step(
            dummy_wetting,
            rho,
            setup,  # ty: ignore[invalid-argument-type]
            trial_step_fn=lambda _p: (jnp.array(0.0), jnp.array(0.0)),
        )


# ---------------------------------------------------------------------------
# The logged failure: a receding line that has just crossed onto the more
# wetting surface must stay pinned, not be driven to that surface's advancing
# bound (which spread it back across the step and flipped the window).
# ---------------------------------------------------------------------------


def test_receding_line_just_past_the_step_stays_pinned(monkeypatch):
    from src.pipeline.state import WettingState

    # [ca_left, ca_right, cll_left, cll_right]; the left line has receded from
    # its anchor at 50.3 to 50.5, just past the step at 50, with CA 99.
    measured = jnp.array([99.0, 45.0, 50.5, 90.0])
    monkeypatch.setattr(hyst, "compute_contact_angle", lambda rho, _mean, edge: (rho[0], rho[1]))
    monkeypatch.setattr(hyst, "compute_contact_line_location", lambda rho, _l, _r, _mean, edge: (rho[2], rho[3]))
    monkeypatch.setattr(hyst, "detect_bubble", lambda rho, _mean, edge: jnp.array(False))
    setup = SimpleNamespace(
        multiphase_params=SimpleNamespace(rho_l=1.0, rho_v=0.001, interface_width=5),
        wetting_edge="bottom",
        config=SimpleNamespace(
            chemical_step_config=_CHEM_CFG,
            grid_shape=(100, 50, 1),
            chemical_step_wall=_wall(),
            hysteresis_config={
                "learning_rate": 0.01,
                "learning_rate_above": 0.01,
                "max_iterations": 3,
                "max_iterations_above": 3,
                "loss_tol": 1e-4,
                "carry_inactive_params": False,
                "saturation_gap": 1.0,
            },
        ),
    )
    wetting = WettingState(
        *(jnp.array(v) for v in (1.0, 1.0, 0.0, 0.0, 99.0, 45.0, 50.3, 90.0)),
    )

    out = update_wetting_state_chemical_step(
        wetting,
        measured,
        setup,  # ty: ignore[invalid-argument-type]
        trial_step_fn=lambda _p: (jnp.array(0.0), measured),
    )

    assert float(out.cll_left) == pytest.approx(50.3)  # anchor held: above 50 but receding, so pinned
    assert float(out.cll_right) == pytest.approx(90.0)  # in its post window [30, 50]: pinned


def test_a_line_held_at_the_step_is_pinned_to_the_step_position(monkeypatch):
    """The trailing line's pin target is the step itself, not where its anchor stopped."""
    from src.pipeline.state import WettingState

    captured = {}

    def fake_impl(wetting, *_args, **kwargs):
        captured.update(kwargs)
        return wetting

    # Left line at 49.6 (held at the step at 50), right line far into the post region.
    measured = (jnp.array(45.0), jnp.array(45.0), jnp.array(49.6), jnp.array(90.0))
    monkeypatch.setattr(hyst, "_measure", lambda _rho, _setup: measured)
    monkeypatch.setattr(hyst, "_update_wetting_state_impl", fake_impl)
    wetting = WettingState(*(jnp.array(v) for v in (1.0, 1.0, 0.0, 0.0, 45.0, 45.0, 49.6, 90.0)))

    update_wetting_state_chemical_step(
        wetting,
        jnp.zeros(4),
        _make_setup(),  # ty: ignore[invalid-argument-type]
        trial_step_fn=lambda _p: (jnp.array(0.0), jnp.zeros(4)),
    )

    assert float(captured["pin_left"]) == 50.0
    assert float(captured["pin_right"]) == 90.0  # not at the step: its own anchor
