"""Conformance tests: existing operators match their protocols.

This test module ensures that all operators in the registry are
compatible with the formal protocol contracts.  If an operator is
registered, it must satisfy its protocol's structural requirements.
"""

import sys
import os

import numpy as np
import pytest
import jax.numpy as jnp

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from operators.protocols import (
    CollisionOperator,
    StreamingOperator,
    EquilibriumOperator,
    MacroscopicOperator,
    BoundaryOperator,
    InitialiserOperator,
)
from setup.lattice import build_lattice
from setup.simulation_setup import build_bc_masks
from registry import get_operators


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def lattice_d2q9():
    """Provide a D2Q9 lattice."""
    return build_lattice("D2Q9")


@pytest.fixture
def grid_shape():
    """Small grid for testing."""
    return (8, 8)


@pytest.fixture
def test_state(lattice_d2q9, grid_shape):
    """Create a simple test state."""
    nx, ny = grid_shape
    q = lattice_d2q9.q
    f = jnp.ones((nx, ny, q, 1)) / q
    rho = jnp.ones((nx, ny, 1, 1))
    u = jnp.zeros((nx, ny, 1, 2))
    return f, rho, u


# ── Collision Operator Tests ─────────────────────────────────────────────────


class TestCollisionProtocol:
    """Verify collision operators conform to CollisionOperator."""

    def test_bgk_collision_conformance(self, lattice_d2q9, grid_shape):
        """BGK collision operator should accept the protocol signature."""
        from operators.collision import build_collision_fn

        bgk_fn = build_collision_fn("bgk")

        nx, ny = grid_shape
        q = lattice_d2q9.q
        f = jnp.ones((nx, ny, q, 1)) / q
        feq = jnp.ones((nx, ny, q, 1)) / q
        tau = 0.8

        # Should not raise
        result = bgk_fn(f, feq, tau)
        assert result.shape == f.shape

    def test_mrt_collision_conformance(self, lattice_d2q9, grid_shape):
        """MRT collision operator should accept the protocol signature."""
        from operators.collision import build_collision_fn

        mrt_fn = build_collision_fn("mrt")

        nx, ny = grid_shape
        q = lattice_d2q9.q
        f = jnp.ones((nx, ny, q, 1)) / q
        feq = jnp.ones((nx, ny, q, 1)) / q
        tau = 0.8
        # MRT requires k_diag as an array
        k_diag = jnp.array([1.0] * q)

        # Should not raise
        result = mrt_fn(f, feq, tau, k_diag=k_diag)
        assert result.shape == f.shape


# ── Streaming Operator Tests ─────────────────────────────────────────────────


class TestStreamingProtocol:
    """Verify streaming operators conform to StreamingOperator."""

    def test_standard_streaming_conformance(self, lattice_d2q9, grid_shape):
        """Standard streaming should match protocol."""
        from operators.streaming import build_streaming_fn

        stream = build_streaming_fn("standard")

        nx, ny = grid_shape
        q = lattice_d2q9.q
        f = jnp.ones((nx, ny, q, 1)) / q

        result = stream(f, lattice_d2q9)
        assert result.shape == f.shape
        # Streaming should preserve mass
        assert jnp.allclose(jnp.sum(result), jnp.sum(f))


# ── Equilibrium Operator Tests ───────────────────────────────────────────────


class TestEquilibriumProtocol:
    """Verify equilibrium operators conform to EquilibriumOperator."""

    def test_wb_equilibrium_conformance(self, lattice_d2q9, grid_shape, test_state):
        """Well-balanced equilibrium should match protocol."""
        from operators.equilibrium import build_equilibrium_fn

        equilibrium = build_equilibrium_fn("wb")

        _, rho, u = test_state
        feq = equilibrium(rho, u, lattice_d2q9)

        nx, ny = grid_shape
        q = lattice_d2q9.q
        assert feq.shape == (nx, ny, q, 1)
        # Equilibrium should conserve mass
        assert jnp.allclose(jnp.sum(feq), jnp.sum(rho))


# ── Macroscopic Operator Tests ───────────────────────────────────────────────


class TestMacroscopicProtocol:
    """Verify macroscopic operators conform to MacroscopicOperator."""

    def test_standard_macroscopic_conformance(
        self, lattice_d2q9, grid_shape, test_state
    ):
        """Standard macroscopic should match protocol."""
        from operators.macroscopic import build_macroscopic_fn

        macroscopic = build_macroscopic_fn("standard")

        f, rho_expected, u_expected = test_state

        rho, u = macroscopic(f, lattice_d2q9)
        assert rho.shape == rho_expected.shape
        assert u.shape == u_expected.shape

    def test_macroscopic_with_force_conformance(
        self, lattice_d2q9, grid_shape, test_state
    ):
        """Macroscopic with force should return 3-tuple."""
        from operators.macroscopic import build_macroscopic_fn

        macroscopic = build_macroscopic_fn("standard")

        f, rho_expected, u_expected = test_state
        force = jnp.zeros((grid_shape[0], grid_shape[1], 1, 2))

        rho, u, force_out = macroscopic(f, lattice_d2q9, force=force)
        assert rho.shape == rho_expected.shape
        assert u.shape == u_expected.shape
        assert force_out.shape == force.shape


# ── Boundary Operator Tests ──────────────────────────────────────────────────


class TestBoundaryProtocol:
    """Verify boundary operators conform to BoundaryOperator."""

    def test_periodic_bc_conformance(self, lattice_d2q9, grid_shape, test_state):
        """Periodic boundary should match protocol."""
        from operators.boundary.composite import build_composite_bc

        f_stream, f_col = test_state[0], test_state[0]
        bc_config = {
            "top": "periodic",
            "bottom": "periodic",
            "left": "periodic",
            "right": "periodic",
        }
        bc_masks = build_bc_masks(grid_shape, bc_config)

        bc_fn = build_composite_bc(bc_config, lattice_d2q9)
        result = bc_fn(f_stream, f_col, bc_masks)
        assert result.shape == f_stream.shape

    def test_bounce_back_bc_conformance(self, lattice_d2q9, grid_shape, test_state):
        """Bounce-back boundary should match protocol."""
        from operators.boundary.composite import build_composite_bc

        f_stream, f_col = test_state[0], test_state[0]
        bc_config = {
            "top": "bounce-back",
            "bottom": "bounce-back",
            "left": "periodic",
            "right": "periodic",
        }
        bc_masks = build_bc_masks(grid_shape, bc_config)

        bc_fn = build_composite_bc(bc_config, lattice_d2q9)
        result = bc_fn(f_stream, f_col, bc_masks)
        assert result.shape == f_stream.shape


# ── Initialiser Operator Tests ───────────────────────────────────────────────


class TestInitialiserProtocol:
    """Verify initialiser operators conform to InitialiserOperator."""

    def test_standard_initialiser_conformance(self, lattice_d2q9, grid_shape):
        """Standard initialiser should match protocol."""
        from operators.initialise import build_initialise_fn

        init_fn = build_initialise_fn("standard")
        nx, ny = grid_shape
        q = lattice_d2q9.q

        f = init_fn(nx, ny, lattice_d2q9)
        assert f.shape == (nx, ny, q, 1)
        # Should conserve total density (mass conservation)
        total_density = jnp.sum(f)
        expected_density = nx * ny  # density = 1 at each point
        assert jnp.allclose(total_density, expected_density, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
