"""Tests for the interior-obstacle bounce-back closure."""

import jax.numpy as jnp
import numpy as np
import pytest
from tud_lbm.lattice.lattice import build_lattice
from tud_lbm.operators.obstacle import build_obstacle_fn

NX, NY, NZ = 8, 8, 1


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


def test_build_obstacle_fn_none_mask_returns_none(lattice) -> None:
    assert build_obstacle_fn(None, lattice) is None


def test_bounceback_reverses_populations_at_masked_cells(lattice) -> None:
    mask = jnp.zeros((NX, NY, NZ, 1, 1), dtype=bool).at[3, 3, 0, 0, 0].set(True)
    obstacle_fn = build_obstacle_fn(mask, lattice)
    assert obstacle_fn is not None

    # Distinct, asymmetric per-direction values so reversal is checkable.
    f_collision = jnp.broadcast_to(
        jnp.asarray(np.arange(lattice.q, dtype=np.float64)).reshape(1, 1, 1, lattice.q, 1),
        (NX, NY, NZ, lattice.q, 1),
    )
    f_streamed = jnp.full((NX, NY, NZ, lattice.q, 1), -1.0)

    result = obstacle_fn(f_streamed, f_collision)

    opp = np.array(lattice.opp_indices)
    expected_masked = f_collision[3, 3, 0, opp, 0]
    np.testing.assert_array_equal(np.asarray(result[3, 3, 0, :, 0]), np.asarray(expected_masked))

    # Unmasked cells are untouched.
    np.testing.assert_array_equal(np.asarray(result[0, 0, 0, :, 0]), np.asarray(f_streamed[0, 0, 0, :, 0]))
