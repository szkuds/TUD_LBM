"""Tests for the velocity-inlet boundary condition."""

import jax.numpy as jnp
import numpy as np
import pytest
from tud_lbm.lattice.lattice import build_lattice
from tud_lbm.operators.boundary._velocity_inlet import apply_velocity_inlet
from tud_lbm.operators.macroscopic._single_phase import compute_macroscopic

NX, NY, NZ = 8, 10, 1


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


def test_inlet_sets_parabolic_profile(lattice) -> None:
    f_streamed = jnp.full((NX, NY, NZ, lattice.q, 1), 0.5)
    f_collision = f_streamed

    result = apply_velocity_inlet(f_streamed, f_collision, lattice, "left", u0=0.05)

    moments = compute_macroscopic(result[0:1], lattice)
    rho, u = moments[0], moments[1]
    u_x = np.asarray(u[0, :, 0, 0, 0])
    y = np.arange(NY)
    expected = 0.05 * 4.0 * (y / (NY - 1)) * (1.0 - y / (NY - 1))
    np.testing.assert_allclose(u_x, expected, atol=1e-6)
    np.testing.assert_allclose(np.asarray(u[0, :, 0, 0, 1]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(rho[0, :, 0, 0, 0]), 1.0, atol=1e-6)


def test_inlet_noop_on_other_edges(lattice) -> None:
    f_streamed = jnp.full((NX, NY, NZ, lattice.q, 1), 0.5)
    result = apply_velocity_inlet(f_streamed, f_streamed, lattice, "right", u0=0.05)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(f_streamed))


def test_inlet_does_not_touch_other_columns(lattice) -> None:
    f_streamed = jnp.full((NX, NY, NZ, lattice.q, 1), 0.5)
    result = apply_velocity_inlet(f_streamed, f_streamed, lattice, "left", u0=0.05)
    np.testing.assert_array_equal(np.asarray(result[1:]), np.asarray(f_streamed[1:]))
