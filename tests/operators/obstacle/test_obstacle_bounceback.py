"""Tests for the interior-obstacle bounce-back closure."""

import jax.numpy as jnp
import numpy as np
import pytest
from src.config.simulation_config import SimulationConfig
from src.lattice.lattice import build_lattice
from src.operators.obstacle import build_obstacle_fn
from src.pipeline.runner import init_state
from src.pipeline.runner import run
from src.pipeline.setup import build_setup

NX, NY, NZ = 8, 8, 1


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


def test_build_obstacle_fn_none_mask_returns_none(lattice) -> None:
    assert build_obstacle_fn(None, lattice) is None


def test_bounceback_reverses_pre_collision_populations_at_masked_cells(lattice) -> None:
    mask = jnp.zeros((NX, NY, NZ, 1, 1), dtype=bool).at[3, 3, 0, 0, 0].set(True)
    obstacle_fn = build_obstacle_fn(mask, lattice)
    assert obstacle_fn is not None

    # Distinct, asymmetric per-direction values so reversal is checkable.
    f_pre = jnp.broadcast_to(
        jnp.asarray(np.arange(lattice.q, dtype=np.float64)).reshape(1, 1, 1, lattice.q, 1),
        (NX, NY, NZ, lattice.q, 1),
    )
    f_col = jnp.full((NX, NY, NZ, lattice.q, 1), -1.0)

    result = obstacle_fn(f_col, f_pre)

    opp = np.array(lattice.opp_indices)
    expected_masked = f_pre[3, 3, 0, opp, 0]
    np.testing.assert_array_equal(np.asarray(result[3, 3, 0, :, 0]), np.asarray(expected_masked))

    # Unmasked cells keep their collision.
    np.testing.assert_array_equal(np.asarray(result[0, 0, 0, :, 0]), np.asarray(f_col[0, 0, 0, :, 0]))


def test_obstacle_conserves_mass_in_a_periodic_flow() -> None:
    """A cylinder in a fully periodic, force-free box neither creates nor destroys mass."""
    cfg = SimulationConfig(
        sim_type="single_phase",
        grid_shape=(40, 30, 1),
        tau=0.6,
        nt=200,
        init_type="standard",
        obstacle_config={"center_x": 20, "center_y": 15, "radius": 5},
    )
    setup = build_setup(cfg)
    state = init_state(setup)
    # Give the fluid a uniform push so populations actually stream into the solid.
    assert setup.equilibrium_fn is not None
    u0 = jnp.zeros_like(state.u).at[..., 0].set(0.05)
    f0 = setup.equilibrium_fn(state.rho, u0, setup.lattice)
    state = state._replace(f=f0)

    final_state, _ = run(setup, state, nt=cfg.nt, save_interval=cfg.nt)

    np.testing.assert_allclose(float(jnp.sum(final_state.f)), float(jnp.sum(f0)), rtol=1e-12)
