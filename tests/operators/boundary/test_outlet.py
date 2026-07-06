"""Tests for the zero-gradient outlet boundary condition."""

import jax.numpy as jnp
import numpy as np
import pytest
from tud_lbm.lattice.lattice import build_lattice
from tud_lbm.operators.boundary._outlet import apply_outlet

NX, NY, NZ = 8, 6, 1


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


def test_outlet_copies_second_to_last_column(lattice) -> None:
    f_streamed = jnp.zeros((NX, NY, NZ, lattice.q, 1))
    f_streamed = f_streamed.at[-2, :, 0, :, 0].set(2.0)
    f_streamed = f_streamed.at[-1, :, 0, :, 0].set(9.0)

    result = apply_outlet(f_streamed, f_streamed, lattice, "right")

    np.testing.assert_array_equal(np.asarray(result[-1]), np.asarray(f_streamed[-2]))
    np.testing.assert_array_equal(np.asarray(result[:-1]), np.asarray(f_streamed[:-1]))


def test_outlet_noop_on_other_edges(lattice) -> None:
    f_streamed = jnp.full((NX, NY, NZ, lattice.q, 1), 0.5)
    result = apply_outlet(f_streamed, f_streamed, lattice, "left")
    np.testing.assert_array_equal(np.asarray(result), np.asarray(f_streamed))


def test_outlet_finite_after_real_step() -> None:
    from tud_lbm.config.simulation_config import SimulationConfig
    from tud_lbm.pipeline.runner import init_state
    from tud_lbm.pipeline.runner import run
    from tud_lbm.pipeline.setup import build_setup

    cfg = SimulationConfig(
        sim_type="single_phase",
        grid_shape=(20, 10, 1),
        tau=0.8,
        nt=5,
        bc_config={
            "top": "bounce-back",
            "bottom": "bounce-back",
            "left": "velocity-inlet",
            "right": "outlet",
            "left_velocity_inlet": {"u0": 0.02},
        },
    )
    setup = build_setup(cfg)
    state = init_state(setup)
    final_state, _ = run(setup, state, nt=cfg.nt)

    assert bool(jnp.all(jnp.isfinite(final_state.f)))
    assert bool(jnp.all(jnp.isfinite(final_state.rho)))
    assert bool(jnp.all(jnp.isfinite(final_state.u)))
