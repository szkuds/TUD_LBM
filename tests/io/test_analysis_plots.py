"""Unit tests for analysis plotting operators."""

from __future__ import annotations
import numpy as np
from src.simulation_io.analysis.droplet_metrics import droplet_series_for_run
from src.simulation_io.plotting.contact_angle_plot import ContactAngleLeftPlot
from src.simulation_io.plotting.contact_line_speed_plot import ContactLineSpeedLeftPlot
from src.simulation_io.plotting.scalar_history_plot import AvgDensityPlot
from src.simulation_io.plotting.scalar_history_plot import DensityRatioPlot
from src.simulation_io.plotting.scalar_history_plot import MaxVelocityPlot
from tests.support.run_dirs import UNIFORM_ITERATIONS
from tests.support.run_dirs import build_run_dir
from tests.support.run_dirs import wetting_config


def _snapshots(run_dir):
    """Snapshot files of a run directory, in timestep order."""
    return sorted((run_dir / "data").glob("timestep_*.npz"))


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
    config = wetting_config()
    run_dir = build_run_dir(tmp_path, config=config)

    result = ContactAngleLeftPlot(config=config).compute(_snapshots(run_dir))

    assert result["iters"].tolist() == list(UNIFORM_ITERATIONS)
    assert len(result["values"]) == len(UNIFORM_ITERATIONS)


def test_contact_angle_left_plot_renders_without_stored_angles(tmp_path):
    """The panel derives angles from rho, so runs that never saved them render.

    Previously these files were skipped outright and the panel came out blank.
    """
    config = wetting_config()
    run_dir = build_run_dir(tmp_path, config=config, with_contact_metrics=False)

    result = ContactAngleLeftPlot(config=config).compute(_snapshots(run_dir))

    assert result["iters"].tolist() == list(UNIFORM_ITERATIONS)
    assert np.all(np.isfinite(result["values"]))


def test_contact_line_speed_left_plot(tmp_path):
    config = wetting_config()
    run_dir = build_run_dir(tmp_path, config=config)

    result = ContactLineSpeedLeftPlot(config=config).compute(_snapshots(run_dir))

    assert result["iters"].tolist() == list(UNIFORM_ITERATIONS)
    # No earlier sample exists for the first snapshot.
    assert result["values"][0] == 0.0


def test_contact_line_speed_matches_shared_series(tmp_path):
    """The operator is a view onto the series, not a second differentiation."""
    config = wetting_config()
    run_dir = build_run_dir(tmp_path, config=config)

    result = ContactLineSpeedLeftPlot(config=config).compute(_snapshots(run_dir))
    series = droplet_series_for_run(run_dir, config)

    assert series is not None
    assert np.allclose(result["values"], series.v_left)


def test_analysis_operators_return_empty_without_config(tmp_path):
    """Config-free construction yields empty panels rather than raising.

    FigureBuilder always supplies a config; this is the defensive path.
    """
    run_dir = build_run_dir(tmp_path)

    result = ContactAngleLeftPlot().compute(_snapshots(run_dir))

    assert len(result["iters"]) == 0
    assert len(result["values"]) == 0
