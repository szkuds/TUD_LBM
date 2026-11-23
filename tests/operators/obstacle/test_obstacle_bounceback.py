"""Tests for the interior-obstacle bounce-back, the first link of the boundary chain."""

import jax.numpy as jnp
import numpy as np
import pytest
from src.config import SimulationConfig
from src.config.boundary import boundary_edges
from src.config.boundary import edges_of
from src.lattice.lattice import build_lattice
from src.operators.boundary import build_bc


def _edges_of(section):
    return boundary_edges(SimulationConfig(boundary_conditions=section))


NX, NY, NZ = 8, 8, 1
_PERIODIC = _edges_of(dict.fromkeys(edges_of(2), "periodic"))


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


def _populations(lattice):
    # Distinct, asymmetric per-direction values so reversal is checkable.
    f_collision = jnp.broadcast_to(
        jnp.asarray(np.arange(lattice.q, dtype=np.float64)).reshape(1, 1, 1, lattice.q, 1),
        (NX, NY, NZ, lattice.q, 1),
    )
    return jnp.full((NX, NY, NZ, lattice.q, 1), -1.0), f_collision


def test_without_a_mask_the_chain_has_no_obstacle(lattice) -> None:
    f_streamed, f_collision = _populations(lattice)

    result = build_bc(_PERIODIC, lattice)(f_streamed, f_collision)

    np.testing.assert_array_equal(np.asarray(result), np.asarray(f_streamed))


def test_bounceback_reverses_populations_at_masked_cells(lattice) -> None:
    mask = jnp.zeros((NX, NY, NZ, 1, 1), dtype=bool).at[3, 3, 0, 0, 0].set(True)
    f_streamed, f_collision = _populations(lattice)

    result = build_bc(_PERIODIC, lattice, mask)(f_streamed, f_collision)

    opp = np.array(lattice.opp_indices)
    expected_masked = f_collision[3, 3, 0, opp, 0]
    np.testing.assert_array_equal(np.asarray(result[3, 3, 0, :, 0]), np.asarray(expected_masked))

    # Unmasked cells are untouched.
    np.testing.assert_array_equal(np.asarray(result[0, 0, 0, :, 0]), np.asarray(f_streamed[0, 0, 0, :, 0]))
