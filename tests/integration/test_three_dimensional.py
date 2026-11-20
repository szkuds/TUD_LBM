"""3D bulk runs against their analytic results: Poiseuille flow and the Laplace law."""

from __future__ import annotations
import jax
import numpy as np
import pytest
from src.config import SimulationConfig
from src.pipeline.runner import init_state
from src.pipeline.setup import build_setup
from src.pipeline.step import observe
from src.pipeline.step import step


def _advance(setup, state, steps: int):
    """*steps* steps keeping only the last state: ``run`` would hold all of them in memory."""
    return jax.lax.scan(lambda current, _: (step(setup, current), None), state, None, length=steps)[0]


def test_channel_flow_matches_the_parabola_with_walls_in_y() -> None:
    force_g, tau, ny, steps = 1e-5, 0.8, 21, 4000
    config = SimulationConfig(
        lattice_type="D3Q19",
        grid_shape=(8, ny, 8),
        tau=tau,
        nt=steps,
        save_interval=steps,
        forces={"gravity_force": {"force_g": force_g, "inclination_angle_deg": 90.0}},
        boundary_conditions={"bottom": "bounce-back", "top": "bounce-back"},
    )
    setup = build_setup(config)

    final = _advance(setup, init_state(setup), steps)

    u = np.asarray(observe(setup, final).u)[..., 0, :]
    nu = (tau - 0.5) / 3.0
    y = np.arange(ny)
    parabola = force_g / (2.0 * nu) * (y + 0.5) * (ny - 0.5 - y)
    assert np.max(np.abs(u[4, :, 4, 0] - parabola)) / parabola.max() < 0.005
    # The flow is uniform along z and x, and has no other component.
    assert np.ptp(u[..., 0], axis=2).max() < 1e-12
    assert np.abs(u[..., 2]).max() < 1e-12


@pytest.mark.slow
def test_a_droplet_obeys_the_laplace_law() -> None:
    """``dP = 2 sigma / R`` in 3D, with the planar double-well ``sigma``.

    The planar ``sigma`` holds only for ``R >> W``: measured at ``R ~ 7.5``, the
    jump is 0.76 of ``2 sigma / R`` with ``W = 4`` and 0.94 with ``W = 3``, and
    0.96 at ``R ~ 11.5``, ``W = 4``. Hence the thin interface here.
    """
    kappa, rho_l, rho_v, width, n, steps = 0.02, 1.0, 0.1, 3, 32, 4000
    config = SimulationConfig(
        lattice_type="D3Q19",
        grid_shape=(n, n, n),
        tau=1.0,
        nt=steps,
        save_interval=steps,
        multiphase={"eos": "double-well", "kappa": kappa, "rho_l": rho_l, "rho_v": rho_v, "interface_width": width},
        init_type="multiphase_bubbles",
        initialisation={"centres": [[0.5, 0.5, 0.5]], "radii": [0.25], "dispersed": "liquid"},
    )
    setup = build_setup(config)
    start = init_state(setup)

    final = _advance(setup, start, steps)

    observed = observe(setup, final)
    rho = np.asarray(observed.rho)[..., 0, 0]
    pressure = np.asarray(observed.pressure)[..., 0, 0]
    radius = ((rho > (rho_l + rho_v) / 2.0).sum() * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    sigma = (2.0 / 3.0) * (kappa / width) * (rho_l - rho_v) ** 2
    jump = pressure[n // 2, n // 2, n // 2] - pressure[1, 1, 1]
    assert jump == pytest.approx(2.0 * sigma / radius, rel=0.1)
    assert rho.sum() == pytest.approx(float(np.asarray(start.f).sum()), rel=1e-12)
