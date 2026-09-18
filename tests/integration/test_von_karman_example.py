"""Integration smoke test for the Von Karman vortex street example.

Mirrors the config_von_karman.toml setup (channel flow past a circular
obstacle, velocity-inlet/outlet edges) on a tiny grid and short run —
asserts the full pipeline completes without NaN/Inf and stays bounded.
"""

import jax.numpy as jnp
import pytest
from src.config.simulation_config import SimulationConfig
from src.pipeline.runner import init_state
from src.pipeline.runner import run
from src.pipeline.setup import build_setup


@pytest.mark.slow
def test_von_karman_pipeline_smoke() -> None:
    cfg = SimulationConfig(
        sim_type="single_phase",
        grid_shape=(80, 30, 1),
        tau=0.524,
        nt=50,
        bc_config={
            "top": "bounce-back",
            "bottom": "bounce-back",
            "left": "velocity-inlet",
            "right": "outlet",
            "left_velocity_inlet": {"u0": 0.04},
        },
        obstacle_config={"center_x": 20, "center_y": 15, "radius": 5},
    )

    setup = build_setup(cfg)
    state = init_state(setup)
    final_state, _ = run(setup, state, nt=cfg.nt)

    assert bool(jnp.all(jnp.isfinite(final_state.f)))
    assert bool(jnp.all(jnp.isfinite(final_state.rho)))
    assert bool(jnp.all(jnp.isfinite(final_state.u)))
    assert float(jnp.min(final_state.rho)) > 0.0
    assert float(jnp.max(final_state.rho)) < 5.0
