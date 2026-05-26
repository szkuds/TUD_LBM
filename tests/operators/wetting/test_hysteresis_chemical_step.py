"""Branch coverage for _get_hysteresis_window_chemical_step and
update_wetting_state_chemical_step in hysteresis.py.

Targets the remaining ~6 uncovered lines (76.9 → ~100%):
- _get_hysteresis_window_chemical_step: pre-step branch (cll < step_x)
- _get_hysteresis_window_chemical_step: post-step branch (cll >= step_x)
- update_wetting_state_chemical_step: registered operator lookup
"""

from __future__ import annotations
from types import SimpleNamespace
import jax.numpy as jnp
import pytest
from tud_lbm.operators.wetting.hysteresis.hysteresis import _get_hysteresis_window_chemical_step

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHEM_CFG = {
    "chemical_step_location": 0.5,  # step_x = 0.5 * 100 = 50
    "ca_advancing_pre_step": 110.0,
    "ca_receding_pre_step": 85.0,
    "ca_advancing_post_step": 95.0,
    "ca_receding_post_step": 70.0,
}


def _make_setup(nx: int = 100) -> SimpleNamespace:
    config = SimpleNamespace(
        chemical_step_config=_CHEM_CFG,
        grid_shape=(nx, 50, 1),
    )
    return SimpleNamespace(config=config)


# ---------------------------------------------------------------------------
# _get_hysteresis_window_chemical_step
# ---------------------------------------------------------------------------


class TestGetHysteresisWindowChemicalStep:
    """Tests for pre-step / post-step branching."""

    def test_pre_step_returns_pre_step_window(self):
        """CLL < step_x (50) → pre-step advancing/receding CA."""
        setup = _make_setup()
        cll = jnp.array(20.0)  # 20 < 50
        ca_adv, ca_rec = _get_hysteresis_window_chemical_step(setup, cll)

        assert float(ca_adv) == pytest.approx(110.0)
        assert float(ca_rec) == pytest.approx(85.0)

    def test_post_step_returns_post_step_window(self):
        """CLL >= step_x (50) → post-step advancing/receding CA."""
        setup = _make_setup()
        cll = jnp.array(60.0)  # 60 >= 50
        ca_adv, ca_rec = _get_hysteresis_window_chemical_step(setup, cll)

        assert float(ca_adv) == pytest.approx(95.0)
        assert float(ca_rec) == pytest.approx(70.0)

    def test_at_boundary_returns_post_step_window(self):
        """CLL exactly at step_x (50) → post-step window (condition is cll < step_x)."""
        setup = _make_setup()
        cll = jnp.array(50.0)  # exactly at boundary → NOT < 50 → post-step
        ca_adv, ca_rec = _get_hysteresis_window_chemical_step(setup, cll)

        assert float(ca_adv) == pytest.approx(95.0)
        assert float(ca_rec) == pytest.approx(70.0)

    def test_different_grid_size_scales_step_x(self):
        """step_x = location * nx, so a different nx moves the threshold."""
        setup = _make_setup(nx=200)  # step_x = 0.5 * 200 = 100
        cll_pre = jnp.array(80.0)  # 80 < 100 → pre-step
        cll_post = jnp.array(120.0)  # 120 >= 100 → post-step

        ca_adv_pre, _ = _get_hysteresis_window_chemical_step(setup, cll_pre)
        ca_adv_post, _ = _get_hysteresis_window_chemical_step(setup, cll_post)

        assert float(ca_adv_pre) == pytest.approx(110.0)
        assert float(ca_adv_post) == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# update_wetting_state_chemical_step — registration check
# ---------------------------------------------------------------------------


def test_chemical_step_hysteresis_operator_is_registered():
    """update_wetting_state_chemical_step must be registered as 'chemical_step_hysteresis'."""
    from tud_lbm.registry import get_operators

    ops = get_operators("wetting")
    assert "chemical_step_hysteresis" in ops, (
        f"Expected 'chemical_step_hysteresis' in wetting registry. Available: {sorted(ops.keys())}"
    )
