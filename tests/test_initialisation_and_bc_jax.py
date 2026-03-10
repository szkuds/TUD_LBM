"""JAX-aware smoke tests for initialisation and boundary condition operators.

Constructs a small 8×8 grid, resolves operators from the registry,
wraps them in jax.jit, and verifies basic correctness.
"""

import pytest
import jax
import jax.numpy as jnp

# Ensure all operators are registered
import simulation_operators  # noqa: F401
import simulation_type  # noqa: F401
import update_timestep  # noqa: F401

from app_setup.registry import get_operators
from app_setup.simulation_setup import SimulationSetup


@pytest.fixture
def small_config():
    """Minimal 8×8 single-phase config for smoke testing."""
    return SimulationSetup(
        grid_shape=(8, 8),
        lattice_type="D2Q9",
        tau=0.6,
        nt=10,
    )


class TestStandardInitialisationJAX:
    """Smoke tests for the standard initialiser on a small grid."""

    def test_standard_init_rho_is_one(self, small_config):
        """After standard init, rho should equal 1 everywhere."""
        init_ops = get_operators("initialise")
        std_cls = init_ops["standard"].cls
        initialiser = std_cls.from_config(small_config)

        f = initialiser()

        # rho = sum over q dimension
        rho = jnp.sum(f, axis=2, keepdims=True)
        assert f.shape[0] == 8
        assert f.shape[1] == 8
        assert jnp.allclose(rho, 1.0, atol=1e-6)

    def test_standard_init_returns_jax_array(self, small_config):
        """Standard init should return a JAX DeviceArray."""
        init_ops = get_operators("initialise")
        std_cls = init_ops["standard"].cls
        initialiser = std_cls.from_config(small_config)

        f = initialiser()
        assert isinstance(f, jnp.ndarray)
        # D2Q9 → q=9, shape should be (8, 8, 9, 1)
        assert f.shape == (8, 8, 9, 1)


class TestPeriodicBoundaryConditionJAX:
    """Smoke tests for the periodic BC on a small grid.

    Note: The BC classes apply ``@partial(jit, static_argnums=(0,))`` inside
    ``__call__``, so we call them directly rather than wrapping in ``jax.jit``.
    """

    def test_periodic_bc_is_noop(self, small_config):
        """Periodic BC should return f_streamed unchanged."""
        bc_ops = get_operators("boundary_condition")
        periodic_cls = bc_ops["periodic"].cls
        bc = periodic_cls.from_config(small_config)

        key = jax.random.PRNGKey(0)
        f_streamed = jax.random.uniform(key, (8, 8, 9, 1))
        f_collision = jax.random.uniform(key, (8, 8, 9, 1))

        result = bc.__call__.__wrapped__(bc, f_streamed, f_collision)
        assert jnp.allclose(result, f_streamed)

    def test_periodic_bc_shape(self, small_config):
        """Periodic BC should preserve shape."""
        bc_ops = get_operators("boundary_condition")
        periodic_cls = bc_ops["periodic"].cls
        bc = periodic_cls.from_config(small_config)

        key = jax.random.PRNGKey(42)
        f_streamed = jax.random.uniform(key, (8, 8, 9, 1))
        f_collision = jax.random.uniform(key, (8, 8, 9, 1))

        result = bc.__call__.__wrapped__(bc, f_streamed, f_collision)
        assert result.shape == f_streamed.shape


class TestDefaultPeriodicConfig:
    """Verify that bc_config defaults to periodic when not provided."""

    def test_single_phase_defaults_to_periodic(self):
        config = SimulationSetup(grid_shape=(8, 8), tau=0.6, nt=10)
        assert config.bc_config is not None
        for edge in ("top", "bottom", "left", "right"):
            assert config.bc_config[edge] == "periodic"

    def test_explicit_bc_config_preserved(self):
        bc = {"top": "symmetry", "bottom": "bounce-back", "left": "periodic", "right": "periodic"}
        config = SimulationSetup(grid_shape=(8, 8), tau=0.6, nt=10, bc_config=bc)
        assert config.bc_config["top"] == "symmetry"
        assert config.bc_config["bottom"] == "bounce-back"


class TestBaseSimulationHelpers:
    """Test _make_initialiser and _make_boundary_condition from BaseSimulation."""

    def test_make_initialiser_standard(self, small_config):
        from simulation_type.base import BaseSimulation

        initialiser = BaseSimulation._make_initialiser("standard", small_config)
        f = initialiser()
        rho = jnp.sum(f, axis=2, keepdims=True)
        assert jnp.allclose(rho, 1.0, atol=1e-6)

    def test_make_initialiser_unknown_raises(self, small_config):
        from simulation_type.base import BaseSimulation

        with pytest.raises(KeyError, match="Unknown initialisation type"):
            BaseSimulation._make_initialiser("nonexistent_init_xyz", small_config)

    def test_make_boundary_condition(self, small_config):
        from simulation_type.base import BaseSimulation

        bc = BaseSimulation._make_boundary_condition(small_config)
        assert bc is not None
        assert callable(bc)


