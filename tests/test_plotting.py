"""Tests for the plotting operator package and figure builder."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import get_operator_names
from util.plotting import visualise
from util.plotting.figure_builder import FigureBuilder


@pytest.fixture
def plotting_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    return run_dir


@pytest.fixture
def simple_config():
    return {"plot_fields": ["density", "velocity"], "simulation_name": "test"}


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
        "rho": np.ones((16, 16, 1, 1)),
        "u": np.zeros((16, 16, 1, 2)),
    }
    path = builder.build(data, timestep=100)
    assert path is not None
    assert path.exists()


def test_build_skips_unavailable_operators(plotting_run_dir):
    builder = FigureBuilder({"plot_fields": ["density", "force"]}, plotting_run_dir)
    data = {"rho": np.ones((8, 8, 1, 1))}
    path = builder.build(data, timestep=5)
    assert path is not None
    assert path.exists()


def test_unknown_plotfield_warns(plotting_run_dir):
    with pytest.warns(UserWarning, match="No plot operator registered"):
        FigureBuilder({"plot_fields": ["nonexistent"]}, plotting_run_dir)


def test_density_operator_registered():
    assert "density" in get_operator_names("plotting")


def test_velocity_operator_registered():
    names = get_operator_names("plotting")
    assert "velocity" in names
    assert "analysis" in names


def test_build_all_creates_one_figure_per_snapshot(plotting_run_dir, simple_config):
    np.savez(
        plotting_run_dir / "data" / "timestep_10.npz",
        rho=np.ones((6, 6, 1, 1)),
        u=np.zeros((6, 6, 1, 2)),
    )
    np.savez(
        plotting_run_dir / "data" / "timestep_20.npz",
        rho=np.ones((6, 6, 1, 1)),
        u=np.zeros((6, 6, 1, 2)),
    )

    builder = FigureBuilder(simple_config, plotting_run_dir)
    saved = builder.build_all()
    assert len(saved) == 2
    assert all(path.exists() for path in saved)


def test_visualise_accepts_run_directory(plotting_run_dir):
    np.savez(
        plotting_run_dir / "data" / "timestep_1.npz",
        rho=np.ones((6, 6, 1, 1)),
        u=np.zeros((6, 6, 1, 2)),
    )
    (plotting_run_dir / "config.json").write_text(
        '{"simulation_name": "demo", "plot_fields": ["density", "velocity"]}',
    )

    visualise(str(plotting_run_dir))

    plots = list((plotting_run_dir / "plots").glob("*.png"))
    assert len(plots) == 1
