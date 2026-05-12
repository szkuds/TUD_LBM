"""Tests for multiphase LBM step with wetting.

Tests for :func:`operators.step.step_multiphase` with wetting configuration
and hysteresis optimisation.
"""

from pathlib import Path
import jax.numpy as jnp
from tud_lbm.config.adapter_toml import TomlAdapter
from tud_lbm.config.simulation_config import SimulationConfig
from tud_lbm.operators.step import build_step_fn
from tud_lbm.pipeline.runner import init_state
from tud_lbm.pipeline.setup import build_setup
from tud_lbm.pipeline.state import WettingState

(
    NX,
    NY,
    NZ,
) = 16, 16, 1


def _wetting_setup():
    """Build a tiny wetting SimulationSetup with hysteresis."""
    cfg = SimulationConfig(
        sim_type="multiphase",
        grid_shape=(NX, NY, NZ),
        tau=0.99,
        nt=3,
        eos="double-well",
        kappa=0.017,
        rho_l=1.0,
        rho_v=0.33,
        interface_width=4,
        bc_config={
            "bottom": "wetting",
            "top": "bounce-back",
            "left": "periodic",
            "right": "periodic",
        },
        wetting_config={
            "phi_left": 1.2,
            "phi_right": 1.2,
            "d_rho_left": 0.05,
            "d_rho_right": 0.05,
        },
        hysteresis_config={
            "ca_advancing": 120.0,
            "ca_receding": 60.0,
            "learning_rate": 0.01,
            "max_iterations": 5,
        },
    )
    return build_setup(cfg)


def _wetting_setup_no_hysteresis():
    """Build a tiny wetting SimulationSetup without hysteresis."""
    cfg = SimulationConfig(
        sim_type="multiphase",
        grid_shape=(NX, NY, NZ),
        tau=0.99,
        nt=3,
        eos="double-well",
        kappa=0.017,
        rho_l=1.0,
        rho_v=0.33,
        interface_width=4,
        bc_config={
            "bottom": "wetting",
            "top": "bounce-back",
            "left": "periodic",
            "right": "periodic",
        },
        wetting_config={
            "phi_left": 1.2,
            "phi_right": 1.2,
            "d_rho_left": 0.05,
            "d_rho_right": 0.05,
        },
    )
    return build_setup(cfg)


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════


class TestStepWetting:
    """Tests for step_multiphase and step_multiphase_wetting functions."""

    def test_step_wetting_increments_time(self):
        """One step should increment t by 1."""
        setup = _wetting_setup_no_hysteresis()
        state = init_state(setup)
        step_fn = build_step_fn("multiphase_wetting")
        new_state = step_fn(setup, state)
        assert int(new_state.t) == 1

    def test_step_wetting_preserves_shapes(self):
        """All fields should have consistent shapes after step."""
        setup = _wetting_setup_no_hysteresis()
        state = init_state(setup)
        step_fn = build_step_fn("multiphase_wetting")
        new_state = step_fn(setup, state)

        assert new_state.f.shape == state.f.shape
        assert new_state.rho.shape == state.rho.shape
        assert new_state.u.shape == state.u.shape

    def test_step_wetting_no_nan(self):
        """No NaN values should appear in the output."""
        setup = _wetting_setup_no_hysteresis()
        state = init_state(setup)
        step_fn = build_step_fn("multiphase_wetting")
        new_state = step_fn(setup, state)

        assert not jnp.isnan(new_state.f).any()
        assert not jnp.isnan(new_state.rho).any()
        assert not jnp.isnan(new_state.u).any()

    def test_step_wetting_preserves_wetting_state_structure(self):
        """Wetting state should remain a WettingState after step."""
        setup = _wetting_setup_no_hysteresis()
        state = init_state(setup)
        step_fn = build_step_fn("multiphase_wetting")
        new_state = step_fn(setup, state)

        assert new_state.wetting is not None
        assert isinstance(new_state.wetting, WettingState)
        assert hasattr(new_state.wetting, "phi_left")
        assert hasattr(new_state.wetting, "phi_right")

    def test_step_wetting_multiple_steps(self):
        """Multiple consecutive steps should work without errors."""
        setup = _wetting_setup_no_hysteresis()
        state = init_state(setup)
        step_fn = build_step_fn("multiphase_wetting")

        # Run 3 steps
        state = step_fn(setup, state)
        assert int(state.t) == 1
        state = step_fn(setup, state)
        assert int(state.t) == 2
        state = step_fn(setup, state)
        assert int(state.t) == 3

        assert not jnp.isnan(state.f).any()
        assert not jnp.isnan(state.rho).any()


class TestComplexConfig:
    """Integration test for config_complex.toml workflow."""

    def test_complex_config(self):
        """Test complex config with wetting_hysteresis."""
        adapter = TomlAdapter()
        root = next(p for p in Path(__file__).parents if (p / "pyproject.toml").exists())
        config_path = root / "examples" / "config_complex.toml"
        cfg = adapter.load(str(config_path))
        assert cfg.sim_type is not None

        setup = build_setup(cfg)
        assert callable(setup.step_fn)

        state = init_state(setup)
        assert state.f.shape is not None

        for i in range(1, 4):
            new_state = setup.step_fn(setup, state)
            assert int(new_state.t) == i
            state = new_state
