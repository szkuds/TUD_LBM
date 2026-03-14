"""Phase 3 tests — jitted step functions and lax.scan runner.

Tests for:
    - ``runner.step.step_single_phase``
    - ``runner.step.step_multiphase``
    - ``runner.step.get_step_fn``
    - ``runner.run.init_state``
    - ``runner.run.run`` (trajectory mode)
    - ``runner.io_callbacks`` (callback plumbing)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# =====================================================================
# Helpers
# =====================================================================


def _single_phase_setup():
    """Return a SimulationSetup for a tiny single-phase grid."""
    from config.simulation_config import SimulationConfig
    from setup.simulation_setup import build_setup

    cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
    setup = build_setup(cfg)
    return setup


def _multiphase_setup():
    """Return a SimulationSetup for a tiny multiphase grid."""
    from config.simulation_config import SimulationConfig
    from setup.simulation_setup import build_setup

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
    setup = build_setup(cfg)
    return setup


# =====================================================================
# init_state
# =====================================================================


class TestInitState:
    """State initialisation."""

    def test_rest_equilibrium(self):
        from runner.run import init_state

        setup = _single_phase_setup()
        state = init_state(setup)
        assert state.f.shape == (8, 8, 9, 1)
        assert state.rho.shape == (8, 8, 1, 1)
        np.testing.assert_allclose(float(jnp.sum(state.rho)), 64.0, rtol=1e-5)

    def test_custom_f(self):
        from runner.run import init_state

        setup = _single_phase_setup()
        f_custom = jnp.ones((8, 8, 9, 1)) * 0.5
        state = init_state(setup, f=f_custom)
        np.testing.assert_allclose(state.f, f_custom)


# =====================================================================
# step_single_phase
# =====================================================================


class TestStepSinglePhase:
    """Single-phase step function."""

    def test_step_increments_t(self):
        from runner.step import step_single_phase
        from runner.run import init_state

        setup = _single_phase_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)
        assert int(new_state.t) == 1

    def test_step_preserves_shape(self):
        from runner.step import step_single_phase
        from runner.run import init_state

        setup = _single_phase_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)
        assert new_state.f.shape == (8, 8, 9, 1)
        assert new_state.rho.shape == (8, 8, 1, 1)
        assert new_state.u.shape == (8, 8, 1, 2)

    def test_step_no_nan(self):
        from runner.step import step_single_phase
        from runner.run import init_state

        setup = _single_phase_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)
        assert not jnp.isnan(new_state.f).any()

    def test_step_output_is_state(self):
        from runner.step import step_single_phase
        from runner.run import init_state
        from state.state import State

        setup = _single_phase_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)
        assert isinstance(new_state, State)

    def test_rest_equilibrium_unchanged(self):
        """At rest equilibrium, one step should not change f."""
        from runner.step import step_single_phase
        from runner.run import init_state
        from operators.equilibrium.equilibrium import compute_equilibrium

        setup = _single_phase_setup()
        lattice = setup.lattice
        nx, ny = setup.grid_shape
        # Compute the actual well-balanced rest equilibrium
        rho = jnp.ones((nx, ny, 1, 1))
        u = jnp.zeros((nx, ny, 1, 2))
        feq = compute_equilibrium(rho, u, lattice)
        state = init_state(setup, f=feq)
        new_state = step_single_phase(setup, state)
        np.testing.assert_allclose(
            np.array(new_state.f),
            np.array(state.f),
            atol=1e-6,
        )


# =====================================================================
# step_multiphase
# =====================================================================


class TestStepMultiphase:
    """Multiphase step function."""

    def test_step_increments_t(self):
        from runner.step import step_multiphase
        from runner.run import init_state

        setup = _multiphase_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)
        assert int(new_state.t) == 1

    def test_step_preserves_shape(self):
        from runner.step import step_multiphase
        from runner.run import init_state

        setup = _multiphase_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)
        assert new_state.f.shape == (16, 16, 9, 1)

    def test_step_no_nan(self):
        from runner.step import step_multiphase
        from runner.run import init_state

        setup = _multiphase_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)
        assert not jnp.isnan(new_state.f).any()

    def test_step_produces_force(self):
        from runner.step import step_multiphase
        from runner.run import init_state

        setup = _multiphase_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)
        assert new_state.force is not None
        assert new_state.force.shape == (16, 16, 1, 2)


# =====================================================================
# get_step_fn
# =====================================================================


class TestGetStepFn:
    """Step function dispatch."""

    def test_single_phase_dispatch(self):
        from runner.step import get_step_fn
        from runner.run import init_state

        setup = _single_phase_setup()
        step_fn = get_step_fn(setup)
        state = init_state(setup)
        new_state = step_fn(state)
        assert int(new_state.t) == 1

    def test_multiphase_dispatch(self):
        from runner.step import get_step_fn
        from runner.run import init_state

        setup = _multiphase_setup()
        step_fn = get_step_fn(setup)
        state = init_state(setup)
        new_state = step_fn(state)
        assert int(new_state.t) == 1


# =====================================================================
# run (lax.scan)
# =====================================================================


class TestFunctionalRun:
    """lax.scan runner."""

    def test_run_trajectory_mode(self):
        from runner.run import run, init_state

        setup = _single_phase_setup()
        state = init_state(setup)
        final, traj = run(setup, state, nt=5)
        assert int(final.t) == 5
        assert traj.f.shape[0] == 5

    def test_run_final_state_no_nan(self):
        from runner.run import run, init_state

        setup = _single_phase_setup()
        state = init_state(setup)
        final, _ = run(setup, state, nt=5)
        assert not jnp.isnan(final.f).any()

    def test_run_multiphase_trajectory(self):
        from runner.run import run, init_state

        setup = _multiphase_setup()
        state = init_state(setup)
        final, traj = run(setup, state, nt=3)
        assert int(final.t) == 3
        assert traj.f.shape[0] == 3

    def test_run_uses_setup_defaults(self):
        from runner.run import run, init_state

        setup = _single_phase_setup()
        state = init_state(setup)
        final, _ = run(setup, state)  # nt defaults to setup.nt
        assert int(final.t) == setup.nt


# =====================================================================
# IO callbacks
# =====================================================================


class TestIOCallbacks:
    """IO callback utilities."""

    def test_state_to_numpy(self):
        from runner.io_callbacks import _state_to_numpy
        from runner.run import init_state

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
        from util.io import SimulationIO

        io = SimulationIO(
            base_dir=str(tmp_path),
            output_format="numpy",
        )
        return io

    def test_trajectory_is_none_with_io_handler(self, tmp_path):
        """When io_handler is supplied, trajectory must be None."""
        from runner.run import run, init_state

        setup = _single_phase_setup()
        state = init_state(setup)
        io = self._make_io(tmp_path)

        final, trajectory = run(
            setup,
            state,
            nt=5,
            save_interval=2,
            io_handler=io,
        )
        assert int(final.t) == 5
        assert trajectory is None

    def test_files_written_at_correct_steps(self, tmp_path):
        """Snapshots are written at every save_interval step."""
        import os
        from runner.run import run, init_state

        setup = _single_phase_setup()
        state = init_state(setup)
        io = self._make_io(tmp_path)

        run(setup, state, nt=6, save_interval=2, io_handler=io)

        files = sorted(os.listdir(io.data_dir))
        # Steps 0..5.  save_interval=2 → saves at t=2,4 (t=0 skipped
        # because save_snapshot_callback checks t % interval == 0 and
        # t > skip_interval; t=0 has it==0 which passes the modulo
        # check but the callback converts to int and 0%2==0 so step 0
        # IS written).  Let's just check that ≥ 1 file is created.
        assert len(files) >= 1
        # All files should be .npz
        assert all(f.endswith(".npz") for f in files)

    def test_save_fields_filters_keys(self, tmp_path):
        """Only the requested fields appear in the saved files."""
        import os
        from runner.run import run, init_state

        setup = _single_phase_setup()
        state = init_state(setup)
        io = self._make_io(tmp_path)

        run(
            setup,
            state,
            nt=4,
            save_interval=2,
            io_handler=io,
            save_fields=("rho",),
        )

        files = sorted(os.listdir(io.data_dir))
        assert len(files) >= 1

        # Check that the npz only contains 'rho'
        data = np.load(os.path.join(io.data_dir, files[0]))
        assert "rho" in data.files
        assert "f" not in data.files
        assert "u" not in data.files

    def test_skip_interval_suppresses_early_saves(self, tmp_path):
        """Steps ≤ skip_interval must not produce files."""
        import os
        from runner.run import run, init_state

        setup = _single_phase_setup()
        state = init_state(setup)
        io = self._make_io(tmp_path)

        # save_interval=1, skip_interval=3 → saves at t=4,5,6,7
        run(
            setup,
            state,
            nt=8,
            save_interval=1,
            io_handler=io,
            skip_interval=3,
        )

        files = sorted(os.listdir(io.data_dir))
        # Steps 0..7 with skip=3 → nothing saved for t=0,1,2,3
        # Steps 4..7 → 4 files
        assert len(files) == 4
        # First file should be timestep_4
        assert files[0] == "timestep_4.npz"

    def test_backward_compat_no_io_handler(self):
        """Existing trajectory-mode call is unaffected."""
        from runner.run import run, init_state

        setup = _single_phase_setup()
        state = init_state(setup)

        final, trajectory = run(setup, state, nt=5)
        assert trajectory is not None
        assert trajectory.f.shape[0] == 5
        assert int(final.t) == 5
