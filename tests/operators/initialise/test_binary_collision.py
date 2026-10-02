"""``binary_collision``: two droplets approaching along ``x`` (Zhang, Guo & Wang 2022, Eqs. 36-37)."""

from __future__ import annotations
import numpy as np
import pytest
from src.lattice.lattice import build_lattice

NX, NY = 121, 61
RHO_L, RHO_V = 6.7645, 0.8388
RADIUS, SPEED = 15.0, 0.02


def _fields(impact_parameter: float):
    from src.operators.initialise import build_initialise_fn

    lattice = build_lattice("D2Q9")
    f = np.asarray(
        build_initialise_fn("binary_collision")(
            (NX, NY, 1),
            lattice,
            rho_l=RHO_L,
            rho_v=RHO_V,
            interface_width=4,
            radius=RADIUS,
            speed=SPEED,
            impact_parameter=impact_parameter,
            gap=10.0,
        )
    )
    rho = f.sum(axis=-2)[:, :, 0, 0]
    momentum = (f * np.asarray(lattice.c)).sum(axis=-2)[:, :, 0, :]
    return rho, momentum


def test_droplets_move_towards_each_other_with_zero_net_momentum():
    rho, momentum = _fields(0.0)
    u_x = momentum[..., 0] / rho
    left = (int((NX - 1) / 2 - RADIUS - 5), (NY - 1) // 2)
    right = (int((NX - 1) / 2 + RADIUS + 5), (NY - 1) // 2)

    assert rho[left] == pytest.approx(RHO_L, rel=1e-3)
    assert rho[right] == pytest.approx(RHO_L, rel=1e-3)
    assert rho[0, 0] == pytest.approx(RHO_V, rel=1e-3)
    assert u_x[left] == pytest.approx(SPEED, rel=1e-3)
    assert u_x[right] == pytest.approx(-SPEED, rel=1e-3)
    assert abs(u_x[0, 0]) < 1e-6
    np.testing.assert_allclose(momentum.sum(axis=(0, 1)), 0.0, atol=1e-10)


def test_impact_parameter_offsets_the_centres_by_two_r_b():
    rho, _ = _fields(0.5)
    liquid = rho > (RHO_L + RHO_V) / 2
    x, y = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    left = liquid & (x < NX / 2)
    right = liquid & (x > NX / 2)

    offset = y[left].mean() - y[right].mean()

    assert offset == pytest.approx(2 * RADIUS * 0.5, abs=0.2)
