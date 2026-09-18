"""Tests for step operators and _common.py shared pipeline.

Covers:
- All TypeError guard branches in _apply_common_step and _multiphase_pipeline
- step_single_phase: macroscopic_fn guard + no-force + with-force integration
- step_multiphase: gradient/laplacian guards + integration
- step_multiphase_wetting: gradient/laplacian guards
"""

from __future__ import annotations
from types import SimpleNamespace
import jax.numpy as jnp
import pytest
from src.config.simulation_config import SimulationConfig
from src.lattice.lattice import build_lattice
from src.pipeline.runner import init_state
from src.pipeline.setup import build_setup
from src.pipeline.state.state import State

NX, NY, NZ = 8, 8, 1


# ── Shared fixtures ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


def _minimal_state(lattice):
    f = jnp.ones((NX, NY, NZ, lattice.q, 1)) / lattice.q
    return State(
        f=f,
        rho=jnp.ones((NX, NY, NZ, 1, 1)),
        u=jnp.zeros((NX, NY, NZ, 1, lattice.d)),
        t=jnp.array(0),
    )


def _sp_setup():
    cfg = SimulationConfig(grid_shape=(NX, NY), tau=0.8, nt=5)
    return build_setup(cfg)


def _sp_setup_gravity():
    cfg = SimulationConfig(
        grid_shape=(NX, NY),
        tau=0.8,
        nt=5,
        gravity_force={"force_g": 1e-6, "inclination_angle_deg": 0.0},
    )
    return build_setup(cfg)


def _mp_setup():
    cfg = SimulationConfig(
        sim_type="multiphase",
        grid_shape=(NX, NY),
        tau=0.99,
        nt=3,
        eos="double-well",
        kappa=0.017,
        rho_l=1.0,
        rho_v=0.33,
        interface_width=4,
    )
    return build_setup(cfg)


# ── _apply_common_step: TypeError guard branches ─────────────────────


class TestApplyCommonStepGuards:
    """Each None-check in _apply_common_step raises TypeError with the right message."""

    def _setup_with(self, lattice, **overrides):
        feq = jnp.ones((NX, NY, NZ, lattice.q, 1)) / lattice.q
        base = {
            "equilibrium_fn": lambda _rho, _u, _lat: feq,
            "collision_fn": lambda f, _feq, _tau, _src=None: f,
            "streaming_fn": lambda f, _lat: f,
            "bc_fn": lambda f_s, _f_c, _masks: f_s,
            "forces": None,
            "tau": 1.0,
            "bc_masks": None,
            "gradient_density": lambda g: g,
            "lattice": lattice,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_raises_when_equilibrium_fn_none(self, lattice):
        from src.operators.step._common import _apply_common_step

        setup = self._setup_with(lattice, equilibrium_fn=None)
        state = _minimal_state(lattice)
        with pytest.raises(TypeError, match="equilibrium_fn is required"):
            _apply_common_step(setup, state, state.rho, state.u, force_tot=None)

    def test_raises_when_collision_fn_none(self, lattice):
        from src.operators.step._common import _apply_common_step

        setup = self._setup_with(lattice, collision_fn=None)
        state = _minimal_state(lattice)
        with pytest.raises(TypeError, match="collision_fn is required"):
            _apply_common_step(setup, state, state.rho, state.u, force_tot=None)

    def test_raises_when_streaming_fn_none(self, lattice):
        from src.operators.step._common import _apply_common_step

        setup = self._setup_with(lattice, streaming_fn=None)
        state = _minimal_state(lattice)
        with pytest.raises(TypeError, match="streaming_fn is required"):
            _apply_common_step(setup, state, state.rho, state.u, force_tot=None)

    def test_raises_when_bc_fn_none(self, lattice):
        from src.operators.step._common import _apply_common_step

        setup = self._setup_with(lattice, bc_fn=None)
        state = _minimal_state(lattice)
        with pytest.raises(TypeError, match="bc_fn is required"):
            _apply_common_step(setup, state, state.rho, state.u, force_tot=None)

    def test_raises_when_forces_none_but_force_tot_present(self, lattice):
        from src.operators.step._common import _apply_common_step

        setup = self._setup_with(lattice, forces=None)
        state = _minimal_state(lattice)
        dummy_force = jnp.zeros((NX, NY, NZ, 1, lattice.d))
        with pytest.raises(TypeError, match="forces is required"):
            _apply_common_step(setup, state, state.rho, state.u, force_tot=dummy_force)


# ── _multiphase_pipeline: TypeError guard branches ───────────────────


class TestMultiphasePipelineGuards:
    """Each None-check in _multiphase_pipeline raises TypeError with the right message."""

    def test_raises_when_macroscopic_fn_none(self, lattice):
        from src.operators.step._common import _multiphase_pipeline

        setup = SimpleNamespace(macroscopic_fn=None)
        f = jnp.ones((NX, NY, NZ, lattice.q, 1)) / lattice.q
        with pytest.raises(TypeError, match="macroscopic_fn is required"):
            _multiphase_pipeline(setup, f, None, lambda g: g, lambda g: g)  # ty: ignore[invalid-argument-type]

    def test_raises_when_multiphase_params_none(self, lattice):
        from src.operators.step._common import _multiphase_pipeline

        setup = SimpleNamespace(macroscopic_fn=lambda *_a, **_k: None, multiphase_params=None)
        f = jnp.ones((NX, NY, NZ, lattice.q, 1)) / lattice.q
        with pytest.raises(TypeError, match="multiphase_params is required"):
            _multiphase_pipeline(setup, f, None, lambda g: g, lambda g: g)  # ty: ignore[invalid-argument-type]

    def test_raises_when_gradient_standard_none(self, lattice):
        from src.operators.step._common import _multiphase_pipeline

        setup = SimpleNamespace(
            macroscopic_fn=lambda *_a, **_k: None,
            multiphase_params=SimpleNamespace(eos="double-well"),
            gradient_standard=None,
        )
        f = jnp.ones((NX, NY, NZ, lattice.q, 1)) / lattice.q
        with pytest.raises(TypeError, match="gradient_standard is required"):
            _multiphase_pipeline(setup, f, None, lambda g: g, lambda g: g)  # ty: ignore[invalid-argument-type]


# ── step_single_phase ─────────────────────────────────────────────────


class TestStepSinglePhase:
    """step_single_phase guard branch and integration paths."""

    def test_raises_when_macroscopic_fn_none(self, lattice):
        from src.operators.step._single_phase import step_single_phase

        setup = SimpleNamespace(macroscopic_fn=None, forces=None)
        state = _minimal_state(lattice)
        with pytest.raises(TypeError, match="macroscopic_fn is required"):
            step_single_phase(setup, state)  # ty: ignore[invalid-argument-type]

    def test_step_increments_time(self):
        from src.operators.step._single_phase import step_single_phase

        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)
        assert int(new_state.t) == 1

    def test_step_preserves_shapes(self):
        from src.operators.step._single_phase import step_single_phase

        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)
        assert new_state.f.shape == state.f.shape
        assert new_state.rho.shape == state.rho.shape

    def test_step_no_nan(self):
        from src.operators.step._single_phase import step_single_phase

        setup = _sp_setup()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)
        assert not jnp.any(jnp.isnan(new_state.f))

    def test_step_with_external_force_covers_force_branch(self):
        """Exercises the force_ext is not None branch in step_single_phase."""
        from src.operators.step._single_phase import step_single_phase

        setup = _sp_setup_gravity()
        state = init_state(setup)
        new_state = step_single_phase(setup, state)
        assert int(new_state.t) == 1
        assert not jnp.any(jnp.isnan(new_state.f))
        assert new_state.force_ext is not None


# ── step_multiphase ───────────────────────────────────────────────────


class TestStepMultiphase:
    """step_multiphase guard branches and integration."""

    def test_raises_when_gradient_density_none(self, lattice):
        from src.operators.step._multiphase import step_multiphase

        setup = SimpleNamespace(gradient_density=None, laplacian_density=lambda g: g, forces=None)
        state = _minimal_state(lattice)
        with pytest.raises(TypeError, match="gradient_density is required"):
            step_multiphase(setup, state)  # ty: ignore[invalid-argument-type]

    def test_raises_when_laplacian_density_none(self, lattice):
        from src.operators.step._multiphase import step_multiphase

        setup = SimpleNamespace(gradient_density=lambda g: g, laplacian_density=None, forces=None)
        state = _minimal_state(lattice)
        with pytest.raises(TypeError, match="laplacian_density is required"):
            step_multiphase(setup, state)  # ty: ignore[invalid-argument-type]

    def test_step_increments_time(self):
        from src.operators.step._multiphase import step_multiphase

        setup = _mp_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)
        assert int(new_state.t) == 1

    def test_step_preserves_shapes(self):
        from src.operators.step._multiphase import step_multiphase

        setup = _mp_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)
        assert new_state.f.shape == state.f.shape
        assert new_state.rho.shape == state.rho.shape

    def test_step_no_nan(self):
        from src.operators.step._multiphase import step_multiphase

        setup = _mp_setup()
        state = init_state(setup)
        new_state = step_multiphase(setup, state)
        assert not jnp.any(jnp.isnan(new_state.f))
        assert not jnp.any(jnp.isnan(new_state.rho))


# ── step_multiphase_wetting guards ────────────────────────────────────


class TestStepMultiphaseWettingGuards:
    """Guard branches in step_multiphase_wetting."""

    def test_raises_when_gradient_density_none(self, lattice):
        from src.operators.step._multiphase_wetting import step_multiphase_wetting

        setup = SimpleNamespace(gradient_density=None, laplacian_density=lambda g: g, forces=None)
        state = _minimal_state(lattice)
        with pytest.raises(TypeError, match="gradient_density is required"):
            step_multiphase_wetting(setup, state)  # ty: ignore[invalid-argument-type]

    def test_raises_when_laplacian_density_none(self, lattice):
        from src.operators.step._multiphase_wetting import step_multiphase_wetting

        setup = SimpleNamespace(gradient_density=lambda g: g, laplacian_density=None, forces=None)
        state = _minimal_state(lattice)
        with pytest.raises(TypeError, match="laplacian_density is required"):
            step_multiphase_wetting(setup, state)  # ty: ignore[invalid-argument-type]
