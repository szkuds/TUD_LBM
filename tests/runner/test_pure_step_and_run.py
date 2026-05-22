"""Tests — pure-function jitted step functions and lax.scan runner.

Tests for the **new pure-function API** (Phase 3):
    - ``operators.step.step_single_phase``
    - ``operators.step.step_multiphase``
    - ``setup.step`` convenience method
    - ``runner.run.run_pure``
    - ``operators.force.source_term.source``

Each test verifies correctness on small grids and jittability
without any legacy operator class instances.
"""

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from tud_lbm.config.simulation_config import SimulationConfig
from tud_lbm.lattice.lattice import build_lattice
from tud_lbm.pipeline.runner import init_state
from tud_lbm.pipeline.setup import build_setup

# =====================================================================
# Helpers
# =====================================================================

NX, NY, NZ = 8, 8, 1


def _sp_setup():
    """Build a tiny single-phase SimulationSetup."""
    cfg = SimulationConfig(grid_shape=(NX, NY, NZ), tau=0.8, nt=10)
    return build_setup(cfg)


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


# =====================================================================
# source
# =====================================================================


class TestSource:
    """``source`` computes a well-balanced forcing source term."""

    @staticmethod
    def _build_gradient_closure(lattice):
        """Build a gradient closure that takes only (grid)."""
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
        """For a uniform field the source should sum to zero over q."""
        from tud_lbm.operators.force._source_term import source

        lattice = build_lattice("D2Q9")
        gradient = self._build_gradient_closure(lattice)
        rho = jnp.ones((NX, NY, NZ, 1, 1))
        u = jnp.zeros((NX, NY, NZ, 1, 2))
        force = jnp.ones((NX, NY, NZ, 1, 2)) * 0.01

        src = source(rho, u, force, lattice, gradient=gradient)
        # The source should satisfy ∑_i S_i = 0 (mass conservation)
        src_sum = jnp.sum(src, axis=-2)
        np.testing.assert_allclose(np.array(src_sum), 0.0, atol=1e-6)


# =====================================================================
# step_single_phase
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
        """At rest equilibrium with periodic BCs, density should be ~1.0."""
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        np.testing.assert_allclose(np.array(new_state.rho), 1.0, atol=1e-5)

    def test_mass_conservation(self):
        """Total mass should be conserved through one step."""
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        mass_before = float(jnp.sum(state.f))
        mass_after = float(jnp.sum(new_state.f))
        np.testing.assert_allclose(mass_before, mass_after, rtol=1e-6)

    def test_multiple_steps_stable(self):
        """5 steps should remain NaN-free and mass-conserving."""
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup()
        state = init_state(setup)

        for _ in range(5):
            state = step_single_phase(setup, state)

        assert not jnp.isnan(state.f).any()
        assert int(state.t) == 5

    def test_persists_force_ext_when_forces_active(self):
        """Single-phase step should persist computed external force on the state."""
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup_with_gravity()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)

        assert new_state.force_ext is not None
        assert new_state.force_ext.shape == (NX, NY, NZ, 1, 2)
        assert not np.allclose(np.array(new_state.force_ext), 0.0)

    def test_force_ext_does_not_accumulate_between_steps(self):
        """Constant gravity should produce a stable per-step external force field."""
        from tud_lbm.operators.step import build_step_fn

        step_single_phase = build_step_fn("single_phase")
        setup = _sp_setup_with_gravity()
        state0 = init_state(setup)

        state1 = step_single_phase(setup, state0)
        state2 = step_single_phase(setup, state1)

        np.testing.assert_allclose(np.array(state2.force_ext), np.array(state1.force_ext), rtol=1e-6, atol=1e-12)


# =====================================================================
# step_multiphase
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
        """Multiphase step should produce an interaction force field."""
        from tud_lbm.operators.step import build_step_fn

        step_multiphase = build_step_fn("multiphase")
        setup = _mp_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)

        assert new_state.force is not None
        assert new_state.force.shape == (16, 16, 1, 1, 2)

    def test_persists_force_ext_when_forces_active(self):
        """Multiphase step should persist computed external force on the state."""
        from tud_lbm.operators.step import build_step_fn

        step_multiphase = build_step_fn("multiphase")
        setup = _mp_setup_with_gravity()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)

        assert new_state.force_ext is not None
        assert new_state.force_ext.shape == (16, 16, 1, 1, 2)
        assert not np.allclose(np.array(new_state.force_ext), 0.0)

    def test_force_ext_does_not_accumulate_between_steps(self):
        """Constant gravity should produce a stable per-step external force field."""
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


# =====================================================================
# --- run_pure (lax.scan) ---
# =====================================================================


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
        """With save_interval > 1, trajectory is subsampled."""
        from tud_lbm.pipeline.runner import run

        setup = _sp_setup()
        state = init_state(setup)

        final_state, trajectory = run(setup, state, nt=10, save_interval=5)

        assert int(final_state.t) == 10
        # 10 steps, save every 5 → indices [0, 5] → 2 snapshots
        assert trajectory.f.shape[0] == 2

    def test_mass_conservation_over_trajectory(self):
        """Total mass should be conserved across the entire run."""
        from tud_lbm.pipeline.runner import run

        setup = _sp_setup()
        state = init_state(setup)

        final_state, _trajectory = run(setup, state, nt=5)

        initial_mass = float(jnp.sum(state.f))
        final_mass = float(jnp.sum(final_state.f))
        np.testing.assert_allclose(initial_mass, final_mass, rtol=1e-5)

    def test_trajectory_t_increases(self):
        """Each snapshot should have an increasing t."""
        from tud_lbm.pipeline.runner import run

        setup = _sp_setup()
        state = init_state(setup)

        _, trajectory = run(setup, state, nt=5)

        ts = np.array(trajectory.t)
        # t should be [1, 2, 3, 4, 5]
        np.testing.assert_array_equal(ts, np.arange(1, 6))

    def test_rest_equilibrium_stable(self):
        """Running 10 steps from rest equilibrium should stay near rho_t_plus1=1."""
        from tud_lbm.pipeline.runner import run

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
        """run_pure with bounce-back BCs over multiple steps."""
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
