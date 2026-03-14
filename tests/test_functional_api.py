"""
Verifies that:
    - The new functional API is the only public surface.
    - ``runner.step`` exports pure functions (no ``Operators`` bundle).
    - ``runner.run`` works without legacy classes.
    - ``config.from_dict`` and ``config.DictAdapter`` work end-to-end.
    - Top-level ``src/__init__`` re-exports the correct symbols.
    - No banned legacy patterns leak into the new modules.
    - End-to-end: config → build_setup → init_state → run.
"""

from __future__ import annotations

import importlib
import inspect
import sys

import jax.numpy as jnp
import numpy as np
import pytest

# =====================================================================
# Public API surface
# =====================================================================


class TestTopLevelExports:
    """The top-level package re-exports only the new functional API."""

    def test_exports_simulation_config(self):
        import config

        assert hasattr(config, "SimulationConfig")

    def test_exports_from_dict(self):
        import config

        assert hasattr(config, "from_dict")
        assert callable(config.from_dict)

    def test_exports_dict_adapter(self):
        import config

        assert hasattr(config, "DictAdapter")


class TestRunnerExports:
    """``runner`` exports only the functional API."""

    def test_exports_run(self):
        import runner

        assert hasattr(runner, "run")
        assert callable(runner.run)

    def test_exports_init_state(self):
        import runner

        assert hasattr(runner, "init_state")

    def test_exports_step_single_phase(self):
        import runner

        assert hasattr(runner, "step_single_phase")

    def test_exports_step_multiphase(self):
        import runner

        assert hasattr(runner, "step_multiphase")

    def test_exports_get_step_fn(self):
        import runner

        assert hasattr(runner, "get_step_fn")

    def test_no_legacy_exports(self):
        import runner

        for name in (
            "Run",
            "SimulationRunner",
            "SimulationFactory",
            "StepResult",
            "Operators",
            "build_operators",
            "run_pure",
            "step_single_phase_pure",
            "step_multiphase_pure",
            "get_pure_step_fn",
        ):
            assert not hasattr(runner, name), f"Legacy export {name!r} still present"


# =====================================================================
# step function signatures (no Operators arg)
# =====================================================================


class TestStepSignatures:
    """Step functions accept (setup, state), not (setup, ops, state)."""

    def test_step_single_phase_params(self):
        from runner.step import step_single_phase

        sig = inspect.signature(step_single_phase)
        params = list(sig.parameters.keys())
        assert params == [
            "setup",
            "state",
        ], f"Expected ['setup', 'state'], got {params}"

    def test_step_multiphase_params(self):
        from runner.step import step_multiphase

        sig = inspect.signature(step_multiphase)
        params = list(sig.parameters.keys())
        assert params == [
            "setup",
            "state",
        ], f"Expected ['setup', 'state'], got {params}"


# =====================================================================
# DictAdapter
# =====================================================================


class TestDictAdapter:
    """``DictAdapter`` builds a ``SimulationConfig`` from a dict."""

    def test_basic(self):
        from config.adapter_dict import DictAdapter
        from config.simulation_config import SimulationConfig

        d = {"grid_shape": [8, 8], "tau": 0.8, "nt": 5}
        adapter = DictAdapter()
        cfg = adapter.load(d)
        assert isinstance(cfg, SimulationConfig)
        assert cfg.grid_shape == (8, 8)
        assert cfg.tau == 0.8

    def test_from_dict_convenience(self):
        from config import from_dict

        cfg = from_dict({"grid_shape": [16, 16], "tau": 0.7, "nt": 10})
        assert cfg.grid_shape == (16, 16)

    def test_validation_error(self):
        from config import from_dict

        with pytest.raises(ValueError, match="tau"):
            from_dict({"tau": 0.3})  # tau <= 0.5 is invalid


# =====================================================================
# run function signature (no Operators arg)
# =====================================================================


class TestRunSignature:
    """``run`` accepts (setup, initial_state, nt, save_interval)."""

    def test_run_params(self):
        from runner.run import run

        sig = inspect.signature(run)
        params = list(sig.parameters.keys())
        assert "setup" in params
        assert "initial_state" in params
        assert "nt" in params
        assert "io_handler" in params
        assert "save_fields" in params
        assert "skip_interval" in params
        # No 'ops' parameter
        assert "ops" not in params


# =====================================================================
# End-to-end: config → build_setup → init_state → run
# =====================================================================


class TestEndToEnd:
    """Full pipeline using only the new functional API."""

    def test_single_phase_e2e(self):
        from config import SimulationConfig
        from setup import build_setup
        from runner import run, init_state

        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=5)
        setup = build_setup(cfg)
        state = init_state(setup)
        final_state, trajectory = run(setup, state)

        assert int(final_state.t) == 5
        assert final_state.f.shape[:2] == (8, 8)
        assert trajectory.f.shape[0] == 5
        assert not jnp.isnan(final_state.f).any()

    def test_multiphase_e2e(self):
        from config import SimulationConfig
        from setup import build_setup
        from runner import run, init_state

        cfg = SimulationConfig(
            sim_type="multiphase",
            grid_shape=(16, 16),
            tau=0.99,
            nt=3,
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )
        setup = build_setup(cfg)
        state = init_state(setup)
        final_state, trajectory = run(setup, state)

        assert int(final_state.t) == 3
        assert not jnp.isnan(final_state.f).any()

    def test_run_uses_setup_nt_default(self):
        from config import SimulationConfig
        from setup import build_setup
        from runner import run, init_state

        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=4)
        setup = build_setup(cfg)
        state = init_state(setup)
        final_state, _ = run(setup, state)  # nt defaults to setup.nt
        assert int(final_state.t) == 4

    def test_save_interval(self):
        from config import SimulationConfig
        from setup import build_setup
        from runner import run, init_state

        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        setup = build_setup(cfg)
        state = init_state(setup)
        final_state, trajectory = run(setup, state, nt=10, save_interval=5)

        assert int(final_state.t) == 10
        # 10 steps, save every 5 → indices [0, 5] → 2 snapshots
        assert trajectory.f.shape[0] == 2

    def test_step_single_phase_direct(self):
        from config import SimulationConfig
        from setup import build_setup
        from runner import step_single_phase, init_state

        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=5)
        setup = build_setup(cfg)
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        assert int(new_state.t) == 1
        assert not jnp.isnan(new_state.f).any()

    def test_mass_conservation(self):
        from config import SimulationConfig
        from setup import build_setup
        from runner import run, init_state

        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        setup = build_setup(cfg)
        state = init_state(setup)
        final_state, _ = run(setup, state)

        initial_mass = float(jnp.sum(state.f))
        final_mass = float(jnp.sum(final_state.f))
        np.testing.assert_allclose(initial_mass, final_mass, rtol=1e-5)


# =====================================================================
# No banned patterns in new modules
# =====================================================================


class TestNoBannedPatterns:
    """New code does not import legacy modules."""

    def test_runner_step_no_app_setup(self):
        import runner.step

        source = inspect.getsource(runner.step)
        assert "app_setup" not in source
        assert "simulation_operators" not in source
        assert "SimulationRunner" not in source
        assert "static_argnums=(0," not in source

    def test_runner_run_no_app_setup(self):
        import runner.run

        source = inspect.getsource(runner.run)
        assert "app_setup" not in source
        assert "SimulationRunner" not in source
        assert "Operators" not in source

    def test_cli_no_app_setup(self):
        import cli.cli

        source = inspect.getsource(cli.cli)
        assert "from app_setup" not in source
        assert "from runner import Run" not in source
