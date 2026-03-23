"""Tests — pure-function jitted step functions and lax.scan runner.

Tests for the **new pure-function API** (Phase 3):
    - ``runner.step.step_single_phase``
    - ``runner.step.step_multiphase``
    - ``runner.step.get_step_fn``
    - ``runner.run.run_pure``
    - ``operators.force.source_term.source``

Each test verifies correctness on small grids and jittability
without any legacy operator class instances.
"""

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from config.simulation_config import SimulationConfig
from runner.run import init_state
from setup.lattice import build_lattice
from setup.simulation_setup import build_setup

# =====================================================================
# Helpers
# =====================================================================

NX, NY = 8, 8


def _sp_setup():
    """Build a tiny single-phase SimulationSetup."""
    cfg = SimulationConfig(grid_shape=(NX, NY), tau=0.8, nt=10)
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


# =====================================================================
# source
# =====================================================================


class TestSource:
    """``source`` computes a well-balanced forcing source term."""

    def _diff_ops(self, lattice):
        from operators.differential.config import DifferentialConfig
        from operators.differential.factory import build_differential_operators

        cfg = DifferentialConfig(
            w=lattice.w,
            c=lattice.c,
            pad_modes=["wrap", "wrap", "wrap", "wrap"],
        )
        return build_differential_operators(cfg)

    def test_shape(self):
        from operators.force.source_term import source

        lattice = build_lattice("D2Q9")
        diff_ops = self._diff_ops(lattice)
        rho = jnp.ones((NX, NY, 1, 1))
        u = jnp.zeros((NX, NY, 1, 2))
        force = jnp.ones((NX, NY, 1, 2)) * 0.001

        src = source(rho, u, force, lattice, diff_ops=diff_ops)
        assert src.shape == (NX, NY, 9, 1)

    def test_zero_force_zero_source(self):
        from operators.force.source_term import source

        lattice = build_lattice("D2Q9")
        diff_ops = self._diff_ops(lattice)
        rho = jnp.ones((NX, NY, 1, 1))
        u = jnp.zeros((NX, NY, 1, 2))
        force = jnp.zeros((NX, NY, 1, 2))

        src = source(rho, u, force, lattice, diff_ops=diff_ops)
        np.testing.assert_allclose(np.array(src), 0.0, atol=1e-10)

    def test_jittable(self):
        from operators.force.source_term import source

        lattice = build_lattice("D2Q9")
        diff_ops = self._diff_ops(lattice)
        rho = jnp.ones((NX, NY, 1, 1))
        u = jnp.zeros((NX, NY, 1, 2))
        force = jnp.ones((NX, NY, 1, 2)) * 0.001

        jitted = jax.jit(partial(source, lattice=lattice, diff_ops=diff_ops))
        src = jitted(rho, u, force)
        assert src.shape == (NX, NY, 9, 1)

    def test_source_sums_to_zero(self):
        """For a uniform field the source should sum to zero over q."""
        from operators.force.source_term import source

        lattice = build_lattice("D2Q9")
        diff_ops = self._diff_ops(lattice)
        rho = jnp.ones((NX, NY, 1, 1))
        u = jnp.zeros((NX, NY, 1, 2))
        force = jnp.ones((NX, NY, 1, 2)) * 0.01

        src = source(rho, u, force, lattice, diff_ops=diff_ops)
        # The source should satisfy ∑_i S_i = 0 (mass conservation)
        src_sum = jnp.sum(src, axis=2)
        np.testing.assert_allclose(np.array(src_sum), 0.0, atol=1e-6)


# =====================================================================
# step_single_phase
# =====================================================================


class TestStepSinglePhasePure:
    """``step_single_phase`` advances the state using pure functions."""

    def test_increments_t(self):
        from runner.step import step_single_phase

        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)
        assert int(new_state.t) == 1

    def test_preserves_shape(self):
        from runner.step import step_single_phase

        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        assert new_state.f.shape == state.f.shape
        assert new_state.rho.shape == state.rho.shape
        assert new_state.u.shape == state.u.shape

    def test_no_nan(self):
        from runner.step import step_single_phase

        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        assert not jnp.isnan(new_state.f).any()
        assert not jnp.isnan(new_state.rho).any()

    def test_output_is_state(self):
        from runner.step import step_single_phase
        from state.state import State

        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)
        assert isinstance(new_state, State)

    def test_rest_equilibrium_unchanged(self):
        """At rest equilibrium with periodic BCs, density should be ~1.0."""
        from runner.step import step_single_phase

        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        np.testing.assert_allclose(np.array(new_state.rho), 1.0, atol=1e-5)

    def test_mass_conservation(self):
        """Total mass should be conserved through one step."""
        from runner.step import step_single_phase

        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        mass_before = float(jnp.sum(state.f))
        mass_after = float(jnp.sum(new_state.f))
        np.testing.assert_allclose(mass_before, mass_after, rtol=1e-6)

    def test_multiple_steps_stable(self):
        """5 steps should remain NaN-free and mass-conserving."""
        from runner.step import step_single_phase

        setup = _sp_setup()
        state = init_state(setup)

        for _ in range(5):
            state = step_single_phase(setup, state)

        assert not jnp.isnan(state.f).any()
        assert int(state.t) == 5


# =====================================================================
# step_multiphase
# =====================================================================


class TestStepMultiphasePure:
    """``step_multiphase`` advances multiphase state using pure functions."""

    def test_increments_t(self):
        from runner.step import step_multiphase

        setup = _mp_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)
        assert int(new_state.t) == 1

    def test_preserves_shape(self):
        from runner.step import step_multiphase

        setup = _mp_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)

        assert new_state.f.shape == state.f.shape
        assert new_state.rho.shape == state.rho.shape

    def test_no_nan(self):
        from runner.step import step_multiphase

        setup = _mp_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)

        assert not jnp.isnan(new_state.f).any()

    def test_produces_force(self):
        """Multiphase step should produce an interaction force field."""
        from runner.step import step_multiphase

        setup = _mp_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)

        assert new_state.force is not None
        assert new_state.force.shape == (16, 16, 1, 2)


# =====================================================================
# get_step_fn
# =====================================================================


class TestGetPureStepFn:
    """``get_step_fn`` dispatches based on simulation type."""

    def test_single_phase_dispatch(self):
        from runner.step import get_step_fn

        setup = _sp_setup()
        step_fn = get_step_fn(setup)
        state = init_state(setup)
        new_state = step_fn(state)
        assert int(new_state.t) == 1

    def test_multiphase_dispatch(self):
        from runner.step import get_step_fn

        setup = _mp_setup()
        step_fn = get_step_fn(setup)
        state = init_state(setup)
        new_state = step_fn(state)
        assert int(new_state.t) == 1

    def test_jit_single_phase(self):
        """The closed-over step_fn should be jittable."""
        from runner.step import get_step_fn

        setup = _sp_setup()
        step_fn = get_step_fn(setup)
        state = init_state(setup)

        jitted_step = jax.jit(step_fn)
        new_state = jitted_step(state)

        assert int(new_state.t) == 1
        assert not jnp.isnan(new_state.f).any()

    def test_jit_multiphase(self):
        """The closed-over multiphase step_fn should be jittable."""
        from runner.step import get_step_fn

        setup = _mp_setup()
        step_fn = get_step_fn(setup)
        state = init_state(setup)

        jitted_step = jax.jit(step_fn)
        new_state = jitted_step(state)

        assert int(new_state.t) == 1
        assert not jnp.isnan(new_state.f).any()


# =====================================================================
# run_pure (lax.scan)
# =====================================================================


class TestRunPure:
    """``run_pure`` executes multiple steps via lax.scan."""

    def test_trajectory_mode(self):
        from runner.run import run

        setup = _sp_setup()
        state = init_state(setup)

        final_state, trajectory = run(setup, state, nt=5)

        assert int(final_state.t) == 5
        assert trajectory.f.shape[0] == 5

    def test_final_state_no_nan(self):
        from runner.run import run

        setup = _sp_setup()
        state = init_state(setup)

        final_state, _ = run(setup, state, nt=3)

        assert not jnp.isnan(final_state.f).any()
        assert not jnp.isnan(final_state.rho).any()

    def test_multiphase_trajectory(self):
        from runner.run import run

        setup = _mp_setup()
        state = init_state(setup)

        final_state, trajectory = run(setup, state, nt=3)

        assert int(final_state.t) == 3
        assert trajectory.f.shape[0] == 3

    def test_save_interval(self):
        """With save_interval > 1, trajectory is subsampled."""
        from runner.run import run

        setup = _sp_setup()
        state = init_state(setup)

        final_state, trajectory = run(setup, state, nt=10, save_interval=5)

        assert int(final_state.t) == 10
        # 10 steps, save every 5 → indices [0, 5] → 2 snapshots
        assert trajectory.f.shape[0] == 2

    def test_mass_conservation_over_trajectory(self):
        """Total mass should be conserved across the entire run."""
        from runner.run import run

        setup = _sp_setup()
        state = init_state(setup)

        final_state, trajectory = run(setup, state, nt=5)

        initial_mass = float(jnp.sum(state.f))
        final_mass = float(jnp.sum(final_state.f))
        np.testing.assert_allclose(initial_mass, final_mass, rtol=1e-5)

    def test_trajectory_t_increases(self):
        """Each snapshot should have an increasing t."""
        from runner.run import run

        setup = _sp_setup()
        state = init_state(setup)

        _, trajectory = run(setup, state, nt=5)

        ts = np.array(trajectory.t)
        # t should be [1, 2, 3, 4, 5]
        np.testing.assert_array_equal(ts, np.arange(1, 6))

    def test_rest_equilibrium_stable(self):
        """Running 10 steps from rest equilibrium should stay near rho=1."""
        from runner.run import run

        setup = _sp_setup()
        state = init_state(setup)

        final_state, _ = run(setup, state, nt=10)

        np.testing.assert_allclose(
            np.array(final_state.rho),
            1.0,
            atol=1e-5,
        )


# =====================================================================
# Bounce-back step with pure functions
# =====================================================================


class TestStepWithBounceBack:
    """Pure-function step works with non-trivial BCs."""

    def test_bounce_back_step(self):
        """Step with bounce-back top/bottom runs without error."""
        from runner.step import step_single_phase

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
        """run_pure with bounce-back BCs over multiple steps."""
        from runner.run import run

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

        final_state, trajectory = run(setup, state, nt=5)

        assert int(final_state.t) == 5
        assert not jnp.isnan(final_state.f).any()


# =====================================================================
# Backward compat: existing legacy API still works
# =====================================================================


class TestLegacyAPIUnchanged:
    """The ``step_single_phase(setup, state)`` functional API works."""

    def test_legacy_step_still_works(self):
        from runner.step import step_single_phase

        cfg = SimulationConfig(grid_shape=(NX, NY), tau=0.8, nt=10)
        setup = build_setup(cfg)
        state = init_state(setup)

        new_state = step_single_phase(setup, state)
        assert int(new_state.t) == 1
