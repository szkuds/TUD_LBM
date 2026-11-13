"""``wb_hydro_inc`` through the whole step: the same fluid as ``wb``, with the pressure in the equilibrium.

The two equilibria solve one momentum equation, so a relaxing droplet must agree
between them to discretisation error, conserve mass, and stay finite through the
hysteresis optimiser's trial steps.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.config.simulation_config import SimulationConfig
from src.pipeline.runner import init_state
from src.pipeline.setup import build_setup
from src.pipeline.step import observe
from src.pipeline.step import step

_MULTIPHASE = {"eos": "double-well", "kappa": 0.02, "rho_l": 1.0, "rho_v": 0.1, "interface_width": 3}


def _advance(config: SimulationConfig, steps: int) -> tuple[np.ndarray, float, float]:
    """``(rho, mass before, mass after)`` of *config* after *steps*."""
    setup = build_setup(config)
    state = init_state(setup)
    final = jax.lax.scan(lambda s, _: (step(setup, s), None), state, None, length=steps)[0]
    rho = np.asarray(observe(setup, final).rho)[:, :, 0, 0, 0]
    return rho, float(jnp.sum(state.f)), float(jnp.sum(final.f))


def _droplet(equilibrium: str) -> SimulationConfig:
    return SimulationConfig(
        multiphase=dict(_MULTIPHASE),
        grid_shape=(32, 32),
        tau=0.9,
        nt=400,
        equilibrium=equilibrium,
        init_type="multiphase_bubbles",
        initialisation={"centres": [[16 / 31, 16 / 31]], "radii": [0.25], "dispersed": "liquid"},
    )


def test_a_relaxing_droplet_matches_the_wb_run():
    rho_wb, _, _ = _advance(_droplet("wb"), 400)
    rho, before, after = _advance(_droplet("wb_hydro_inc"), 400)

    assert after == pytest.approx(before, rel=1e-12)
    assert rho.max() == pytest.approx(rho_wb.max(), abs=5e-3)
    assert rho.min() == pytest.approx(rho_wb.min(), abs=5e-3)
    np.testing.assert_allclose(rho, rho_wb, atol=2e-2)
    # Not the same arithmetic: the pressure really is in the populations.
    assert float(np.abs(rho - rho_wb).max()) > 0.0


def test_it_runs_through_the_hysteresis_trial_steps():
    config = SimulationConfig(
        multiphase=dict(_MULTIPHASE),
        grid_shape=(64, 24),
        tau=0.9,
        nt=60,
        equilibrium="wb_hydro_inc",
        init_type="multiphase_bubbles",
        initialisation={"centres": [[25 / 63, 0.0]], "radii": [0.3], "dispersed": "liquid"},
        boundary_conditions={"bottom": "wetting", "top": "symmetry", "left": "bounce-back", "right": "bounce-back"},
        wetting={"phi_left": 1.0, "phi_right": 1.0, "d_rho_left": 0.0, "d_rho_right": 0.0},
        hysteresis={"ca_advancing": 100.0, "ca_receding": 80.0, "max_iterations": 3, "learning_rate": 0.005},
    )

    rho, before, after = _advance(config, 60)

    assert np.all(np.isfinite(rho))
    assert after == pytest.approx(before, rel=1e-12)
