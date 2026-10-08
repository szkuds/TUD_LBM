"""Integration tests: single-phase core LBM against closed-form solutions.

The first rungs of a ladder that isolates the core pipeline (equilibrium ->
collision -> streaming -> bounce-back) from multiphase, wetting and hysteresis:

* **Pipe flow, MRT** — the Poiseuille profile of ``test_poiseuille.py`` under
  the MRT collision and the relaxation rates of the production bubble runs.
* **Inclined channel** — gravity at 60 deg in a periodic-x channel. The
  along-channel component must drive the Poiseuille profile of ``g*sin(60)``;
  the wall-normal one must be carried by a hydrostatic density gradient with
  no flow.
* **Closed box ("just air")** — bounce-back on all four sides with inclined
  gravity. The steady state is hydrostatic,
  ``rho ∝ exp(g_vec . x / cs2)``, at rest, with mass conserved.

A single-phase run takes the ``standard_equilibrium`` / ``guo`` pair, whose
second moment carries the pressure ``cs2*rho`` that balances a force normal to
a wall. With the pressureless ``wb`` equilibrium the last two rungs fail: the
fluid piles up against the wall indefinitely.
"""

import numpy as np
import pytest
from src.config.simulation_config import SimulationConfig
from src.pipeline.runner import init_state
from src.pipeline.runner import run
from src.pipeline.setup import build_setup

TAU = 0.99
K_DIAG = (0.0, 0.5, 0.6, 0.0, 1.2, 0.0, 1.2, 1.0 / TAU, 1.0 / TAU)
CS2 = 1.0 / 3.0

CHANNEL_BCS = {
    "left": "periodic",
    "right": "periodic",
    "top": "bounce-back",
    "bottom": "bounce-back",
    "front": "periodic",
    "back": "periodic",
}
BOX_BCS = {**CHANNEL_BCS, "left": "bounce-back", "right": "bounce-back"}


def _run(scheme: str, grid: tuple[int, int, int], bcs: dict, force_g: float, angle_deg: float, nt: int):
    mrt = scheme == "mrt"
    config = SimulationConfig(
        sim_type="single_phase",
        grid_shape=grid,
        lattice_type="D2Q9",
        tau=TAU,
        nt=nt,
        gravity_force={"force_g": force_g, "inclination_angle_deg": angle_deg},
        bc_config=bcs,
        save_interval=nt,
        collision_scheme="mrt" if mrt else "bgk",
        k_diag=K_DIAG if mrt else None,
    )
    setup = build_setup(config)
    state = init_state(setup)
    final_state, _ = run(setup, state, nt=nt)
    rho = np.asarray(final_state.rho)[:, :, 0, 0, 0]
    u = np.asarray(final_state.u)[:, :, 0, 0, :]
    return rho, u, float(np.asarray(state.rho).sum())


def _poiseuille(ny: int, force: float) -> np.ndarray:
    """Half-way bounce-back Poiseuille profile, as in ``test_poiseuille.py``."""
    nu = CS2 * (TAU - 0.5)
    y = np.arange(ny)
    return force / (2.0 * nu) * (y + 0.5) * (ny - 0.5 - y)


def _l2_error(simulated: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(simulated - expected) / np.linalg.norm(expected))


@pytest.mark.integration
def test_mrt_poiseuille_matches_the_parabolic_profile():
    force_g = 1e-6
    _, u, _ = _run("mrt", (5, 32, 1), CHANNEL_BCS, force_g, 90.0, 5000)

    assert _l2_error(u[..., 0].mean(axis=0), _poiseuille(32, force_g)) < 0.02


@pytest.mark.integration
@pytest.mark.parametrize("scheme", ["bgk", "mrt"])
def test_inclined_channel_splits_gravity_into_flow_and_hydrostatics(scheme):
    force_g, angle = 1e-6, np.deg2rad(60.0)
    rho, u, _ = _run(scheme, (5, 32, 1), CHANNEL_BCS, force_g, 60.0, 5000)

    assert _l2_error(u[..., 0].mean(axis=0), _poiseuille(32, force_g * np.sin(angle))) < 0.02
    assert np.abs(u[..., 1]).max() < 1e-3 * force_g * 5000
    slope = np.polyfit(np.arange(32), np.log(rho.mean(axis=0)), 1)[0]
    assert slope == pytest.approx(-force_g * np.cos(angle) / CS2, rel=0.02)


@pytest.mark.integration
@pytest.mark.parametrize("scheme", ["bgk", "mrt"])
def test_closed_box_reaches_hydrostatic_rest(scheme):
    force_g, angle = 1e-5, np.deg2rad(60.0)
    rho, u, initial_mass = _run(scheme, (32, 32, 1), BOX_BCS, force_g, 60.0, 20000)

    x, y = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    design = np.c_[x.ravel(), y.ravel(), np.ones(x.size)]
    grad_x, grad_y, _ = np.linalg.lstsq(design, np.log(rho).ravel(), rcond=None)[0]

    assert grad_x == pytest.approx(force_g * np.sin(angle) / CS2, rel=0.01)
    assert grad_y == pytest.approx(-force_g * np.cos(angle) / CS2, rel=0.01)
    assert np.abs(u).max() < 1e-3 * force_g
    assert rho.sum() == pytest.approx(initial_mass, rel=1e-10)
