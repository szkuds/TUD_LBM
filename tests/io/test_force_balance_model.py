"""The reduced droplet model: closed form against the integrator, and its limits."""

from __future__ import annotations
import math
import numpy as np
import pytest
from src.simulation_io.analysis.force_balance.model import Regime
from src.simulation_io.analysis.force_balance.model import coasting_length
from src.simulation_io.analysis.force_balance.model import crossing_time
from src.simulation_io.analysis.force_balance.model import frictionless_crossing_capillary
from src.simulation_io.analysis.force_balance.model import frictionless_stop_distance
from src.simulation_io.analysis.force_balance.model import integrate_trajectory
from src.simulation_io.analysis.force_balance.model import piecewise_trajectory
from src.simulation_io.analysis.force_balance.model import regime_state
from src.simulation_io.analysis.force_balance.model import relaxation_time
from src.simulation_io.analysis.force_balance.model import steady_capillary
from src.simulation_io.analysis.force_balance.runs import Windows

_H1 = math.cos(math.radians(100)) - math.cos(math.radians(120))
_H2 = math.cos(math.radians(100)) - math.cos(math.radians(70))
_H3 = math.cos(math.radians(50)) - math.cos(math.radians(70))


def _step_functions(bounds: tuple[float, float], hs: tuple[float, ...], betas: tuple[float, ...]):
    def h_of_x(x: float) -> float:
        return hs[int(np.searchsorted(bounds, x, side="right"))]

    def beta_of_x(x: float) -> float:
        return betas[int(np.searchsorted(bounds, x, side="right"))]

    return h_of_x, beta_of_x


def test_window_hysteresis_values():
    """The SI's reference windows: 100/120 hydrophobic, 50/70 hydrophilic."""
    windows = Windows(receding_phob=100.0, advancing_phob=120.0, receding_phil=50.0, advancing_phil=70.0)
    assert windows.h2 == pytest.approx(-0.516, abs=5e-4)
    assert windows.h3 == pytest.approx(0.301, abs=5e-4)
    assert windows.h1 == pytest.approx(_H1)


@pytest.mark.parametrize(
    ("bo", "betas"),
    [(0.83, (4.8, 6.4, 13.3)), (0.5, (4.8, 6.4, 13.3)), (0.0, (2.0, 2.0, 2.0))],
)
def test_closed_form_matches_integrator(bo, betas):
    oh, bounds = 0.40, (-0.6, 0.6)
    hs = (_H1, _H2, _H3)
    t = np.linspace(0.0, 25.0, 501)
    x0 = -1.0 if bo > 0 else 0.0
    x_cf, ca_cf = piecewise_trajectory(
        t, x0, 0.0, oh, bo, bounds, [Regime(h, b) for h, b in zip(hs, betas, strict=True)]
    )
    h_of_x, beta_of_x = _step_functions(bounds, hs, betas)
    x_num, ca_num = integrate_trajectory(t, x0, 0.0, oh, bo, h_of_x, beta_of_x, breakpoints=bounds)

    assert np.max(np.abs(ca_cf - ca_num)) < 1e-4
    assert np.max(np.abs(x_cf - x_num)) < 1e-4


def test_steady_state_limit():
    oh, bo, h, beta = 0.4, 0.83, _H1, 4.8
    t = np.linspace(0.0, 40.0 * relaxation_time(oh, beta), 200)
    _, ca = integrate_trajectory(t, 0.0, 0.0, oh, bo, lambda _x: h, lambda _x: beta)

    assert ca[-1] == pytest.approx(steady_capillary(bo, h, beta), rel=1e-6)
    assert coasting_length(bo, h, beta, oh) == pytest.approx(steady_capillary(bo, h, beta) / (oh**2 * beta))


def test_pinned_when_gravity_below_hysteresis():
    t = np.linspace(0.0, 10.0, 11)
    x, ca = piecewise_trajectory(t, -1.0, 0.0, 0.4, 0.9 * _H1, (-0.6, 0.6), [Regime(_H1, 5.0)] * 3)

    assert np.all(ca == 0.0)
    assert np.all(x == -1.0)


def test_frictionless_stop_distance():
    """Released centred on the step: X_f - X0 = (w/2)(1 - H2/H3) = 2.71 (w/2)."""
    width = 1.2
    assert frictionless_stop_distance(_H2, _H3, width) / (0.5 * width) == pytest.approx(2.71, abs=0.01)

    oh = 0.4
    t = np.linspace(0.0, 60.0, 6001)
    bounds = (-0.5 * width, 0.5 * width)
    x, ca = piecewise_trajectory(t, 0.0, 0.0, oh, 0.0, bounds, [Regime(_H1, 0.0), Regime(_H2, 0.0), Regime(_H3, 0.0)])

    assert x[-1] == pytest.approx(frictionless_stop_distance(_H2, _H3, width), rel=1e-9)
    regime2 = Regime(_H2, 0.0)
    t_cross = crossing_time(0.5 * width, 0.0, oh, 0.0, regime2)
    ca_cross, _ = regime_state(np.array(t_cross), 0.0, oh, 0.0, regime2)
    assert float(ca_cross) == pytest.approx(frictionless_crossing_capillary(oh, _H2, width), rel=1e-9)
    assert np.max(ca) == pytest.approx(float(ca_cross), rel=1e-3)

    h_of_x, beta_of_x = _step_functions(bounds, (_H1, _H2, _H3), (0.0, 0.0, 0.0))
    x_num, _ = integrate_trajectory(t, 0.0, 0.0, oh, 0.0, h_of_x, beta_of_x, breakpoints=bounds)
    assert x_num[-1] == pytest.approx(x[-1], abs=1e-6)
