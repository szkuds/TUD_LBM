"""Tests for tud_lbm/pipeline/runner.py.

Merged from:
  - tests/runner/test_step_and_run.py   : init_state, IO callbacks, streaming IO
  - tests/runner/test_pure_step_and_run.py : pure step functions, lax.scan runner
  - tests/runner/test_t_from_snapshot.py : _t_from_snapshot branch coverage
"""

from __future__ import annotations
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from tud_lbm.config.simulation_config import SimulationConfig
from tud_lbm.lattice.lattice import build_lattice
from tud_lbm.pipeline.runner import _t_from_snapshot
from tud_lbm.pipeline.runner import init_state
from tud_lbm.pipeline.setup import build_setup

# =====================================================================
# Helpers
# =====================================================================

NX, NY, NZ = 8, 8, 1


def _single_phase_setup():
    """Return a SimulationSetup for a tiny single-phase grid."""
    cfg = SimulationConfig(grid_shape=(NX, NY), tau=0.8, nt=10)
    return build_setup(cfg)


def _sp_setup():
    """Build a tiny single-phase SimulationSetup (alias)."""
    return _single_phase_setup()


def _sp_setup_with_gravity():
    """Build a tiny single-phase setup with gravity enabled."""
    cfg = SimulationConfig(
        grid_shape=(NX, NY, NZ),
        tau=0.8,
        nt=10,
        gravity_force={"force_g": 1e-6, "inclination_angle_deg": 10.0},
    )
    return build_setup(cfg)


def _mp_setup():
    """Build a tiny multiphase SimulationSetup."""
    cfg = SimulationConfig(
        sim_type="multiphase",
        grid_shape=(16, 16),
        tau=0.99,
        nt=5,
        eos="double-well",
        kappa=0.017,
        rho_l=1.0,
        rho_v=0.33,
        interface_width=4,
    )
    return build_setup(cfg)


def _mp_setup_with_gravity():
    """Build a tiny multiphase setup with gravity enabled."""
    cfg = SimulationConfig(
        sim_type="multiphase",
        grid_shape=(16, 16),
        tau=0.99,
        nt=5,
        eos="double-well",
        kappa=0.017,
        rho_l=1.0,
        rho_v=0.33,
        interface_width=4,
        gravity_force={"force_g": 1e-6, "inclination_angle_deg": 10.0},
    )
    return build_setup(cfg)


def _cfg(**kwargs) -> SimulationConfig:
    base = {"grid_shape": (8, 8), "tau": 0.8, "nt": 10}
    base.update(kwargs)
    return SimulationConfig(**base)


# =====================================================================
# init_state
# =====================================================================


class TestInitState:
    """State initialisation."""

    def test_rest_equilibrium(self):
        setup = _single_phase_setup()
        state = init_state(setup)
        assert state.f.shape == (8, 8, 1, 9, 1)
        assert state.rho.shape == (8, 8, 1, 1, 1)
        np.testing.assert_allclose(float(jnp.sum(state.rho)), 64.0, rtol=1e-5)

    def test_custom_f(self):
        setup = _single_phase_setup()
        f_custom = jnp.ones((8, 8, 1, 9, 1)) * 0.5
        state = init_state(setup, f=f_custom)
        np.testing.assert_allclose(state.f, f_custom)

    def test_resume_timestep_parsed_from_snapshot_name(self):
        cfg = SimulationConfig(
            grid_shape=(8, 8),
            tau=0.8,
            nt=10,
            init_type="init_from_file",
            init_dir="/tmp/timestep_50000.npz",
        )
        setup = build_setup(cfg)
        f_custom = jnp.ones((8, 8, 1, 9, 1)) * 0.5
        state = init_state(setup, f=f_custom)
        assert int(state.t) == 50000

    def test_resume_timestep_falls_back_to_zero_for_nonconforming_name(self):
        cfg = SimulationConfig(
            grid_shape=(8, 8),
            tau=0.8,
            nt=10,
            init_type="init_from_file",
            init_dir="/tmp/latest_snapshot.npz",
        )
        setup = build_setup(cfg)
        f_custom = jnp.ones((8, 8, 1, 9, 1)) * 0.5
        state = init_state(setup, f=f_custom)
        assert int(state.t) == 0

    def test_run_advances_from_resumed_timestep(self):
        from tud_lbm.pipeline.runner import run

        cfg = SimulationConfig(
            grid_shape=(8, 8),
            tau=0.8,
            nt=10,
            init_type="init_from_file",
            init_dir="/tmp/timestep_12.npz",
        )
        setup = build_setup(cfg)
        f_custom = jnp.ones((8, 8, 1, 9, 1)) * 0.5
        state = init_state(setup, f=f_custom)
        final_state, _ = run(setup, state, nt=3)
        assert int(final_state.t) == 15

    def test_t_from_snapshot_returns_zero_for_non_digit_suffix(self):
        cfg = SimulationConfig(
            grid_shape=(8, 8),
            tau=0.8,
            nt=10,
            init_type="init_from_file",
            init_dir="/tmp/timestep_12a.npz",
        )
        assert int(_t_from_snapshot(cfg)) == 0


# =====================================================================
# _t_from_snapshot — full branch coverage
# =====================================================================


class TestTFromSnapshot:
    """All branches of _t_from_snapshot."""

    def test_non_init_from_file_returns_zero(self):
        cfg = _cfg(init_type="standard")
        assert int(_t_from_snapshot(cfg)) == 0

    def test_init_from_file_no_init_dir_returns_zero(self):
        cfg = SimpleNamespace(init_type="init_from_file", init_dir=None)
        assert int(_t_from_snapshot(cfg)) == 0

    def test_stem_without_timestep_prefix_returns_zero(self, tmp_path):
        npz = tmp_path / "snapshot_1000.npz"
        npz.write_bytes(b"")
        cfg = _cfg(init_type="init_from_file", init_dir=str(npz))
        assert int(_t_from_snapshot(cfg)) == 0

    def test_stem_with_non_digit_suffix_returns_zero(self, tmp_path):
        npz = tmp_path / "timestep_abc.npz"
        npz.write_bytes(b"")
        cfg = _cfg(init_type="init_from_file", init_dir=str(npz))
        assert int(_t_from_snapshot(cfg)) == 0

    def test_valid_timestep_stem_returns_correct_t(self, tmp_path):
        npz = tmp_path / "timestep_500.npz"
        npz.write_bytes(b"")
        cfg = _cfg(init_type="init_from_file", init_dir=str(npz))
        assert int(_t_from_snapshot(cfg)) == 500

    def test_timestep_zero_stem(self, tmp_path):
        npz = tmp_path / "timestep_0.npz"
        npz.write_bytes(b"")
        cfg = _cfg(init_type="init_from_file", init_dir=str(npz))
        assert int(_t_from_snapshot(cfg)) == 0


# =====================================================================
# IO callbacks
# =====================================================================


class TestIOCallbacks:
    """IO callback utilities."""

    def test_state_to_numpy(self):
        from tud_lbm.pipeline.io_callbacks import _state_to_numpy

        setup = _single_phase_setup()
        state = init_state(setup)
        np_dict = _state_to_numpy(state)
        assert isinstance(np_dict, dict)
        assert "f" in np_dict
        assert isinstance(np_dict["f"], np.ndarray)


# =====================================================================
# Streaming I/O via io_handler
# =====================================================================


class TestStreamingIO:
    """run() with io_handler streams snapshots to disk."""

    def _make_io(self, tmp_path):
        """Build a SimulationIO that writes numpy files to *tmp_path*."""
        from tud_lbm.io import SimulationIO

        return SimulationIO(
            base_dir=str(tmp_path),
            output_format="numpy",
        )

    def test_trajectory_is_none_with_io_handler(self, tmp_path):
        from tud_lbm.pipeline.runner import run

        setup = _single_phase_setup()
        state = init_state(setup)
        io = self._make_io(tmp_path)

        final, trajectory = run(setup, state, nt=5, save_interval=2, io_handler=io)
        assert int(final.t) == 5
        assert trajectory is None

    def test_files_written_at_correct_steps(self, tmp_path):
        from tud_lbm.pipeline.runner import run

        setup = _single_phase_setup()
        state = init_state(setup)
        io = self._make_io(tmp_path)

        run(setup, state, nt=6, save_interval=2, io_handler=io)

        files = sorted(p.name for p in Path(io.data_dir).iterdir())
        assert len(files) >= 1
        assert all(f.endswith(".npz") for f in files)

    def test_save_fields_filters_keys(self, tmp_path):
        from tud_lbm.pipeline.runner import run

        setup = _single_phase_setup()
        state = init_state(setup)
        io = self._make_io(tmp_path)

        run(setup, state, nt=4, save_interval=2, io_handler=io, save_fields=("rho",))

        files = sorted(p.name for p in Path(io.data_dir).iterdir())
        assert len(files) >= 1

        data = np.load(str(Path(io.data_dir) / files[0]))
        assert "rho" in data.files
        assert "f" not in data.files
        assert "u" not in data.files

    def test_skip_interval_suppresses_early_saves(self, tmp_path):
        from tud_lbm.pipeline.runner import run

        setup = _single_phase_setup()
        state = init_state(setup)
        io = self._make_io(tmp_path)

        run(setup, state, nt=8, save_interval=1, io_handler=io, skip_interval=3)

        files = sorted(p.name for p in Path(io.data_dir).iterdir())
        assert len(files) == 5
        assert files[0] == "timestep_4.npz"

    def test_backward_compat_no_io_handler(self):
        from tud_lbm.pipeline.runner import run

        setup = _single_phase_setup()
        state = init_state(setup)

        final, trajectory = run(setup, state, nt=5)
        assert trajectory is not None
        assert trajectory.f.shape[0] == 5
        assert int(final.t) == 5


# =====================================================================
# source (force source term)
# =====================================================================


class TestSource:
    """``source`` computes a well-balanced forcing source term."""

    @staticmethod
    def _build_gradient_closure(lattice):
        from tud_lbm.operators.differential import build_differential_fn

        _gradient = build_differential_fn("gradient")
        pad_modes = ("wrap", "wrap", "wrap", "wrap")

        def gradient(grid):
            return _gradient(grid, lattice.w, lattice.c, pad_modes)

        return gradient

    def test_shape(self):
        from tud_lbm.operators.force._source_term import source

        lattice = build_lattice("D2Q9")
        gradient = self._build_gradient_closure(lattice)
        rho = jnp.ones((NX, NY, NZ, 1, 1))
        u = jnp.zeros((NX, NY, NZ, 1, 2))
        force = jnp.ones((NX, NY, NZ, 1, 2)) * 0.001

        src = source(rho, u, force, lattice, gradient=gradient)
        assert src.shape == (NX, NY, NZ, 9, 1)

    def test_zero_force_zero_source(self):
        from tud_lbm.operators.force._source_term import source

        lattice = build_lattice("D2Q9")
        gradient = self._build_gradient_closure(lattice)
        rho = jnp.ones((NX, NY, NZ, 1, 1))
        u = jnp.zeros((NX, NY, NZ, 1, 2))
        force = jnp.zeros((NX, NY, NZ, 1, 2))

        src = source(rho, u, force, lattice, gradient=gradient)
        np.testing.assert_allclose(np.array(src), 0.0, atol=1e-10)

    def test_jittable(self):
        from tud_lbm.operators.force._source_term import source

        lattice = build_lattice("D2Q9")
        gradient = self._build_gradient_closure(lattice)
        rho = jnp.ones((NX, NY, NZ, 1, 1))
        u = jnp.zeros((NX, NY, NZ, 1, 2))
        force = jnp.ones((NX, NY, NZ, 1, 2)) * 0.001

        jitted = jax.jit(partial(source, lattice=lattice, gradient=gradient))
        src = jitted(rho, u, force)
        assert src.shape == (NX, NY, NZ, 9, 1)

    def test_source_sums_to_zero(self):
        from tud_lbm.operators.force._source_term import source

        lattice = build_lattice("D2Q9")
        gradient = self._build_gradient_closure(lattice)
        rho = jnp.ones((NX, NY, NZ, 1, 1))
        u = jnp.zeros((NX, NY, NZ, 1, 2))
        force = jnp.ones((NX, NY, NZ, 1, 2)) * 0.01

        src = source(rho, u, force, lattice, gradient=gradient)
        src_sum = jnp.sum(src, axis=-2)
        np.testing.assert_allclose(np.array(src_sum), 0.0, atol=1e-6)


# =====================================================================
# step_single_phase (pure function API)
# =====================================================================


class TestStepSinglePhasePure:
    """``step_single_phase`` advances the state using pure functions."""

    def test_increments_t(self):
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)
        assert int(new_state.t) == 1

    def test_preserves_shape(self):
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        assert new_state.f.shape == state.f.shape
        assert new_state.rho.shape == state.rho.shape
        assert new_state.u.shape == state.u.shape

    def test_no_nan(self):
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        assert not jnp.isnan(new_state.f).any()
        assert not jnp.isnan(new_state.rho).any()

    def test_output_is_state(self):
        from tud_lbm.operators.step import build_step_fn
        from tud_lbm.pipeline.state import State

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)
        assert isinstance(new_state, State)

    def test_rest_equilibrium_unchanged(self):
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        np.testing.assert_allclose(np.array(new_state.rho), 1.0, atol=1e-5)

    def test_mass_conservation(self):
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        mass_before = float(jnp.sum(state.f))
        mass_after = float(jnp.sum(new_state.f))
        np.testing.assert_allclose(mass_before, mass_after, rtol=1e-6)

    def test_multiple_steps_stable(self):
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup()
        state = init_state(setup)

        for _ in range(5):
            state = step_single_phase(setup, state)

        assert not jnp.isnan(state.f).any()
        assert int(state.t) == 5

    def test_persists_force_ext_when_forces_active(self):
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup_with_gravity()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        assert new_state.force_ext is not None
        assert new_state.force_ext.shape == (NX, NY, NZ, 1, 2)
        assert not np.allclose(np.array(new_state.force_ext), 0.0)

    def test_force_ext_does_not_accumulate_between_steps(self):
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup_with_gravity()
        state0 = init_state(setup)

        state1 = step_single_phase(setup, state0)
        state2 = step_single_phase(setup, state1)

        np.testing.assert_allclose(np.array(state2.force_ext), np.array(state1.force_ext), rtol=1e-6, atol=1e-12)


# =====================================================================
# step_multiphase (pure function API)
# =====================================================================


class TestStepMultiphasePure:
    """``step_multiphase`` advances multiphase state using pure functions."""

    def test_increments_t(self):
        from tud_lbm.operators.step import build_step_fn

        step_multiphase = build_step_fn("multiphase")
        setup = _mp_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)
        assert int(new_state.t) == 1

    def test_preserves_shape(self):
        from tud_lbm.operators.step import build_step_fn

        step_multiphase = build_step_fn("multiphase")
        setup = _mp_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)

        assert new_state.f.shape == state.f.shape
        assert new_state.rho.shape == state.rho.shape

    def test_no_nan(self):
        from tud_lbm.operators.step import build_step_fn

        step_multiphase = build_step_fn("multiphase")
        setup = _mp_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)

        assert not jnp.isnan(new_state.f).any()

    def test_produces_force(self):
        from tud_lbm.operators.step import build_step_fn

        step_multiphase = build_step_fn("multiphase")
        setup = _mp_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)

        assert new_state.force is not None
        assert new_state.force.shape == (16, 16, 1, 1, 2)

    def test_persists_force_ext_when_forces_active(self):
        from tud_lbm.operators.step import build_step_fn

        step_multiphase = build_step_fn("multiphase")
        setup = _mp_setup_with_gravity()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)

        assert new_state.force_ext is not None
        assert new_state.force_ext.shape == (16, 16, 1, 1, 2)
        assert not np.allclose(np.array(new_state.force_ext), 0.0)

    def test_force_ext_does_not_accumulate_between_steps(self):
        from tud_lbm.operators.step import build_step_fn

        step_multiphase = build_step_fn("multiphase")
        setup = _mp_setup_with_gravity()
        state0 = init_state(setup)

        state1 = step_multiphase(setup, state0)
        state2 = step_multiphase(setup, state1)

        np.testing.assert_allclose(np.array(state2.force_ext), np.array(state1.force_ext), rtol=1e-6, atol=1e-12)


# =====================================================================
# setup.step convenience method
# =====================================================================


class TestSetupStep:
    """Step function dispatch via setup.step()."""

    def test_single_phase_via_setup(self):
        setup = _sp_setup()
        state = init_state(setup)
        new_state = setup.step_fn(setup, state)
        assert int(new_state.t) == 1

    def test_multiphase_via_setup(self):
        setup = _mp_setup()
        state = init_state(setup)
        new_state = setup.step_fn(setup, state)
        assert int(new_state.t) == 1


class TestRunPure:
    """``run_pure`` executes multiple steps via lax.scan."""

    def test_trajectory_mode(self):
        from tud_lbm.pipeline.runner import run

        setup = _sp_setup()
        state = init_state(setup)

        final_state, trajectory = run(setup, state, nt=5)

        assert int(final_state.t) == 5
        assert trajectory.f.shape[0] == 5

    def test_final_state_no_nan(self):
        from tud_lbm.pipeline.runner import run

        setup = _sp_setup()
        state = init_state(setup)

        final_state, _ = run(setup, state, nt=3)

        assert not jnp.isnan(final_state.f).any()
        assert not jnp.isnan(final_state.rho).any()

    def test_multiphase_trajectory(self):
        from tud_lbm.pipeline.runner import run

        setup = _mp_setup()
        state = init_state(setup)

        final_state, trajectory = run(setup, state, nt=3)

        assert int(final_state.t) == 3
        assert trajectory.f.shape[0] == 3

    def test_save_interval(self):
        from tud_lbm.pipeline.runner import run

        setup = _sp_setup()
        state = init_state(setup)

        final_state, trajectory = run(setup, state, nt=10, save_interval=5)

        assert int(final_state.t) == 10
        assert trajectory.f.shape[0] == 2

    def test_mass_conservation_over_trajectory(self):
        from tud_lbm.pipeline.runner import run

        setup = _sp_setup()
        state = init_state(setup)

        final_state, _trajectory = run(setup, state, nt=5)

        initial_mass = float(jnp.sum(state.f))
        final_mass = float(jnp.sum(final_state.f))
        np.testing.assert_allclose(initial_mass, final_mass, rtol=1e-5)

    def test_trajectory_t_increases(self):
        from tud_lbm.pipeline.runner import run

        setup = _sp_setup()
        state = init_state(setup)

        _, trajectory = run(setup, state, nt=5)

        ts = np.array(trajectory.t)
        np.testing.assert_array_equal(ts, np.arange(1, 6))

    def test_rest_equilibrium_stable(self):
        from tud_lbm.pipeline.runner import run

        setup = _sp_setup()
        state = init_state(setup)

        final_state, _ = run(setup, state, nt=10)

        np.testing.assert_allclose(np.array(final_state.rho), 1.0, atol=1e-5)


# =====================================================================
# Bounce-back step with pure functions
# =====================================================================


class TestStepWithBounceBack:
    """Pure-function step works with non-trivial BCs."""

    def test_bounce_back_step(self):
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        cfg = SimulationConfig(
            grid_shape=(NX, NY),
            tau=0.8,
            nt=5,
            bc_config={
                "top": "bounce-back",
                "bottom": "bounce-back",
                "left": "periodic",
                "right": "periodic",
            },
        )
        setup = build_setup(cfg)
        state = init_state(setup)

        new_state = step_single_phase(setup, state)

        assert new_state.f.shape == state.f.shape
        assert not jnp.isnan(new_state.f).any()

    def test_bounce_back_run(self):
        from tud_lbm.pipeline.runner import run

        cfg = SimulationConfig(
            grid_shape=(NX, NY, NZ),
            tau=0.8,
            nt=5,
            bc_config={
                "top": "bounce-back",
                "bottom": "bounce-back",
                "left": "periodic",
                "right": "periodic",
            },
        )
        setup = build_setup(cfg)
        state = init_state(setup)

        final_state, _trajectory = run(setup, state, nt=5)

        assert int(final_state.t) == 5
        assert not jnp.isnan(final_state.f).any()


# =====================================================================
# Backward compat: existing legacy API still works
# =====================================================================


class TestLegacyAPIUnchanged:
    """The ``step_single_phase(setup, state)`` functional API works."""

    def test_legacy_step_still_works(self):
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        cfg = SimulationConfig(grid_shape=(NX, NY), tau=0.8, nt=10)
        setup = build_setup(cfg)
        state = init_state(setup)

        new_state = step_single_phase(setup, state)
        assert int(new_state.t) == 1
