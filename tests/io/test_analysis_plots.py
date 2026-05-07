"""Unit tests for analysis plotting operators."""

from __future__ import annotations
import numpy as np
from tud_lbm.io.plotting.analysis import AvgDensityPlot
from tud_lbm.io.plotting.analysis import ContactAngleLeftPlot
from tud_lbm.io.plotting.analysis import ContactLineSpeedLeftPlot
from tud_lbm.io.plotting.analysis import DensityRatioPlot
from tud_lbm.io.plotting.analysis import MaxVelocityPlot


def _make_snapshot(path, *, step: int, rho_val: float, ux: float, ca_left: float, cll_left: float):
    rho = np.ones((3, 3, 1, 1, 1)) * rho_val
    u = np.zeros((3, 3, 1, 1, 2))
    u[:, :, 0, 0, 0] = ux
    np.savez(
        path / f"timestep_{step}.npz",
        rho=rho,
        u=u,
        ca_left=np.array(ca_left),
        ca_right=np.array(90.0),
        cll_left=np.array(cll_left),
        cll_right=np.array(0.0),
    )


def test_max_velocity_plot(tmp_path):
    _make_snapshot(tmp_path, step=10, rho_val=2.0, ux=0.2, ca_left=80.0, cll_left=1.0)
    _make_snapshot(tmp_path, step=20, rho_val=3.0, ux=0.4, ca_left=81.0, cll_left=2.0)

    plot = MaxVelocityPlot()
    result = plot.compute(sorted(tmp_path.glob("*.npz")))

    assert result["iters"].tolist() == [10, 20]
    assert np.allclose(result["values"], [0.2, 0.4])


def test_density_ratio_plot(tmp_path):
    rho = np.ones((3, 3, 1, 1, 1))
    rho[0, 0, 0, 0, 0] = 4.0
    u = np.zeros((3, 3, 1, 1, 2))
    np.savez(tmp_path / "timestep_5.npz", rho=rho, u=u)

    plot = DensityRatioPlot()
    result = plot.compute(sorted(tmp_path.glob("*.npz")))

    assert result["iters"].tolist() == [5]
    assert np.allclose(result["values"], [4.0])


def test_avg_density_plot(tmp_path):
    _make_snapshot(tmp_path, step=1, rho_val=2.0, ux=0.0, ca_left=80.0, cll_left=0.0)
    _make_snapshot(tmp_path, step=2, rho_val=4.0, ux=0.0, ca_left=82.0, cll_left=0.0)

    plot = AvgDensityPlot()
    result = plot.compute(sorted(tmp_path.glob("*.npz")))

    assert result["iters"].tolist() == [1, 2]
    assert np.allclose(result["values"], [2.0, 4.0])


def test_contact_angle_left_plot(tmp_path):
    _make_snapshot(tmp_path, step=7, rho_val=1.0, ux=0.0, ca_left=77.5, cll_left=1.0)

    plot = ContactAngleLeftPlot()
    result = plot.compute(sorted(tmp_path.glob("*.npz")))

    assert result["iters"].tolist() == [7]
    assert np.allclose(result["values"], [77.5])


def test_contact_line_speed_left_plot(tmp_path):
    _make_snapshot(tmp_path, step=10, rho_val=1.0, ux=0.0, ca_left=80.0, cll_left=2.0)
    _make_snapshot(tmp_path, step=15, rho_val=1.0, ux=0.0, ca_left=80.0, cll_left=3.0)
    _make_snapshot(tmp_path, step=25, rho_val=1.0, ux=0.0, ca_left=80.0, cll_left=5.0)

    plot = ContactLineSpeedLeftPlot()
    result = plot.compute(sorted(tmp_path.glob("*.npz")))

    assert result["iters"].tolist() == [10, 15, 25]
    assert np.allclose(result["values"], [0.0, 0.2, 0.2])
