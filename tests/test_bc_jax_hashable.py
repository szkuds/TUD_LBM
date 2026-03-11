"""Tests to verify that BoundaryCondition and its subclasses are hashable for JAX JIT.

The ``@partial(jit, static_argnums=(0,))`` pattern used on ``__call__`` in every
BC operator requires that ``self`` (argument 0) can be hashed by JAX.
``@dataclass`` auto-generates ``__eq__`` which sets ``__hash__ = None``, so
we explicitly restore identity-based hashing on ``BoundaryConditionBase``.

These tests verify:
1. BC instances are hashable (hash() does not raise).
2. The composite BoundaryCondition can be called inside JIT without error.
3. A full multiphase Update step with non-periodic BCs (bounce-back, symmetry,
   wetting) can run through JIT without the ``unhashable type`` error.
"""

import os
import sys

import jax.numpy as jnp
import pytest

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Trigger operator registration
import simulation_operators  # noqa: F401
import simulation_type  # noqa: F401
import update_timestep  # noqa: F401

from app_setup.simulation_setup import SimulationSetup
from simulation_operators.boundary_condition import (
    BoundaryCondition,
    BoundaryConditionBase,
    PeriodicBoundaryCondition,
)
from simulation_operators.boundary_condition.bounce_back import BounceBackBoundaryCondition
from simulation_operators.boundary_condition.symmetry import SymmetryBoundaryCondition


# ── Hashability ──────────────────────────────────────────────────────

class TestBCHashability:
    """All BC instances must be hashable (required by JAX static_argnums)."""

    def test_base_is_hashable(self):
        bc = BoundaryConditionBase()
        assert isinstance(hash(bc), int)

    def test_periodic_is_hashable(self):
        config = SimulationSetup(grid_shape=(16, 16), tau=0.6)
        bc = PeriodicBoundaryCondition.from_config(config)
        assert isinstance(hash(bc), int)

    def test_composite_bc_is_hashable(self):
        config = SimulationSetup(
            grid_shape=(16, 16), tau=0.6,
            bc_config={
                "top": "periodic", "bottom": "bounce-back",
                "left": "periodic", "right": "periodic",
            },
        )
        bc = BoundaryCondition(config)
        assert isinstance(hash(bc), int)

    def test_bounce_back_is_hashable(self):
        config = SimulationSetup(
            grid_shape=(16, 16), tau=0.6,
            bc_config={
                "top": "bounce-back", "bottom": "bounce-back",
                "left": "periodic", "right": "periodic",
            },
        )
        bc = BounceBackBoundaryCondition.from_config(config)
        assert isinstance(hash(bc), int)

    def test_symmetry_is_hashable(self):
        config = SimulationSetup(
            grid_shape=(16, 16), tau=0.6,
            bc_config={
                "top": "symmetry", "bottom": "symmetry",
                "left": "periodic", "right": "periodic",
            },
        )
        bc = SymmetryBoundaryCondition.from_config(config)
        assert isinstance(hash(bc), int)

    def test_identity_equality(self):
        """Two different instances should not be equal; same instance should."""
        a = BoundaryConditionBase()
        b = BoundaryConditionBase()
        assert a == a
        assert a != b


# ── JIT call ─────────────────────────────────────────────────────────

class TestBCJITCall:
    """BoundaryCondition.__call__ must work inside JAX JIT without hashing errors."""

    def test_periodic_bc_jit(self):
        config = SimulationSetup(grid_shape=(8, 8), tau=0.6)
        bc = BoundaryCondition(config)
        f = jnp.ones((8, 8, 9, 1))
        result = bc(f, f)  # first call triggers JIT tracing
        assert result.shape == (8, 8, 9, 1)

    def test_bounce_back_bc_jit(self):
        config = SimulationSetup(
            grid_shape=(8, 8), tau=0.6,
            bc_config={
                "top": "bounce-back", "bottom": "bounce-back",
                "left": "periodic", "right": "periodic",
            },
        )
        bc = BoundaryCondition(config)
        f = jnp.ones((8, 8, 9, 1))
        result = bc(f, f)
        assert result.shape == (8, 8, 9, 1)

    def test_symmetry_bc_jit(self):
        config = SimulationSetup(
            grid_shape=(8, 8), tau=0.6,
            bc_config={
                "top": "symmetry", "bottom": "bounce-back",
                "left": "periodic", "right": "periodic",
            },
        )
        bc = BoundaryCondition(config)
        f = jnp.ones((8, 8, 9, 1))
        result = bc(f, f)
        assert result.shape == (8, 8, 9, 1)


# ── Full Update step with BCs ───────────────────────────────────────

class TestUpdateWithBCsJIT:
    """A full Update timestep with non-periodic BCs must JIT without error."""

    def test_single_phase_bounce_back_update(self):
        config = SimulationSetup(
            grid_shape=(16, 16), tau=0.6,
            bc_config={
                "top": "bounce-back", "bottom": "bounce-back",
                "left": "periodic", "right": "periodic",
            },
        )
        from update_timestep import Update
        update = Update(config)
        f = jnp.ones((16, 16, 9, 1)) / 9.0
        result = update(f)
        assert result.shape == (16, 16, 9, 1)
        assert not jnp.isnan(result).any()

    def test_multiphase_wetting_update(self):
        config = SimulationSetup(
            sim_type="multiphase",
            grid_shape=(32, 16), tau=0.99,
            eos="double-well", kappa=0.017,
            rho_l=1.0, rho_v=0.33, interface_width=4,
            bc_config={
                "top": "symmetry", "bottom": "wetting",
                "left": "periodic", "right": "periodic",
                "wetting_params": {
                    "phi_left": 1.0, "phi_right": 1.0,
                    "d_rho_left": 0.0, "d_rho_right": 0.0,
                },
            },
        )
        from update_timestep import UpdateMultiphase
        update = UpdateMultiphase(config)
        f = jnp.ones((32, 16, 9, 1)) / 9.0
        result = update(f)
        assert result.shape == (32, 16, 9, 1)
        assert not jnp.isnan(result).any()

