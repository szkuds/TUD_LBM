"""Tests for the plotting operator package and figure builder."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pytest
from tud_lbm.config import SimulationConfig
from tud_lbm.io.plotting.figure_builder import FigureBuilder
from tud_lbm.registry import get_operator_names

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def plotting_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    return run_dir


@pytest.fixture
def simple_config():
    return SimulationConfig(
        plot_fields=["density", "velocity"],
        simulation_name="test",
    )


def test_layout_1():
    assert FigureBuilder._layout(1) == (1, 1)


def test_layout_4():
    assert FigureBuilder._layout(4) == (2, 2)


def test_layout_5():
    ncols, nrows = FigureBuilder._layout(5)
    assert ncols * nrows >= 5


def test_build_calls_savefig(plotting_run_dir, simple_config):
    builder = FigureBuilder(simple_config, plotting_run_dir)
    data = {
        "rho": np.ones((16, 16, 1, 1, 1)),
        "u": np.zeros((16, 16, 1, 1, 2)),
    }
    path = builder.build(data, timestep=100)
    assert path is not None
    assert path.exists()


def test_build_skips_unavailable_operators(plotting_run_dir):
    config = SimulationConfig(plot_fields=["density", "force"])
    builder = FigureBuilder(config, plotting_run_dir)
    data = {"rho": np.ones((8, 8, 1, 1, 1))}
    path = builder.build(data, timestep=5)
    assert path is not None
    assert path.exists()


def test_unknown_plotfield_warns(plotting_run_dir):
    config = SimulationConfig(plot_fields=["nonexistent"])
    with pytest.warns(UserWarning, match="No plot operator registered"):
        FigureBuilder(config, plotting_run_dir)


def test_density_operator_registered():
    assert "density" in get_operator_names("plotting")


def test_velocity_operator_registered():
    names = get_operator_names("plotting")
    assert "velocity" in names


def test_analysis_operators_registered():
    names = get_operator_names("analysis")
    assert "max_velocity" in names
    assert "density_ratio" in names
    assert "avg_density" in names
    assert "simulation_csv" in names


def test_build_all_creates_one_figure_per_snapshot(plotting_run_dir, simple_config):
    np.savez(
        plotting_run_dir / "data" / "timestep_10.npz",
        rho=np.ones((6, 6, 1, 1, 1)),
        u=np.zeros((6, 6, 1, 1, 2)),
    )
    np.savez(
        plotting_run_dir / "data" / "timestep_20.npz",
        rho=np.ones((6, 6, 1, 1, 1)),
        u=np.zeros((6, 6, 1, 1, 2)),
    )

    builder = FigureBuilder(simple_config, plotting_run_dir)
    saved = builder.build_all()
    assert len(saved) == 2
    assert all(path.exists() for path in saved)


def test_build_analysis_writes_analysis_plots(plotting_run_dir):
    np.savez(
        plotting_run_dir / "data" / "timestep_10.npz",
        rho=np.ones((6, 6, 1, 1, 1)),
        u=np.zeros((6, 6, 1, 1, 2)),
        ca_left=np.array(80.0),
        ca_right=np.array(100.0),
        cll_left=np.array(2.0),
        cll_right=np.array(5.0),
    )
    np.savez(
        plotting_run_dir / "data" / "timestep_20.npz",
        rho=np.ones((6, 6, 1, 1, 1)) * 2,
        u=np.ones((6, 6, 1, 1, 2)) * 0.1,
        ca_left=np.array(81.0),
        ca_right=np.array(99.0),
        cll_left=np.array(2.5),
        cll_right=np.array(5.4),
    )

    config = SimulationConfig(plot_fields=["max_velocity", "avg_density", "contact_angles_pair"])
    builder = FigureBuilder(config, plotting_run_dir)
    written = builder.build_analysis()
    assert len(written) == 3
    assert all(path.exists() for path in written)


def test_build_all_accepts_run_directory(plotting_run_dir: Path):
    np.savez(
        plotting_run_dir / "data" / "timestep_1.npz",
        rho=np.ones((6, 6, 1, 1, 1)),
        u=np.zeros((6, 6, 1, 1, 2)),
    )
    config = SimulationConfig(simulation_name="demo", plot_fields=["density", "velocity"])
    builder = FigureBuilder(config=config, run_dir=str(plotting_run_dir))

    saved = builder.build_all()

    assert len(saved) == 1
    assert saved[0].exists()

    plots = list((plotting_run_dir / "plots" / "snapshots").glob("*.png"))
    assert len(plots) == 1
