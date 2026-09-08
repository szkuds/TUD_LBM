"""Tests for the Bo/Oh length-scale diagnostic figure."""

from __future__ import annotations
import numpy as np
import pytest
from src.config import SimulationConfig
from src.simulation_io.analysis.physical_parameters import write_length_scale_figure
from src.simulation_io.analysis.physical_parameters.length_scale_figure import _analytic_region
from src.simulation_io.analysis.physical_parameters.length_scale_figure import _counted_panel
from src.simulation_io.analysis.physical_parameters.length_scale_figure import _snapshot_panels

_FIGURE_REL = ("plots", "analysis", "length_scale.png")


def _mp_config(**kwargs) -> SimulationConfig:
    base = {
        "sim_type": "multiphase",
        "grid_shape": (40, 20),
        "eos": "double-well",
        "kappa": 0.02,
        "rho_l": 1.0,
        "rho_v": 0.5,
        "interface_width": 2,
        "gravity_force": {"force_g": 1e-6},
        "initialisation": {"centres": [[0.5, 0.1]], "radii": [0.4]},
    }
    base.update(kwargs)
    return SimulationConfig(**base)  # ty: ignore[invalid-argument-type]


def _droplet_field(nx: int = 40, ny: int = 20, low: float = 0.55, high: float = 0.91) -> np.ndarray:
    rho = np.full((nx, ny, 1, 1, 1), low)
    rho[10:30, 3:8, 0, 0, 0] = high
    return rho


def _write_snapshots(run_dir, steps=(0, 100)) -> None:
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        np.savez(data_dir / f"timestep_{step}.npz", rho=_droplet_field())


def _init_from_file_config(tmp_path, **kwargs) -> SimulationConfig:
    npz_path = tmp_path / "init_state.npz"
    np.savez(npz_path, rho=_droplet_field())
    return _mp_config(init_type="init_from_file", init_dir=str(npz_path), initialisation={}, **kwargs)


def test_writes_a_figure_for_the_analytic_branch_without_snapshots(tmp_path):
    path = write_length_scale_figure(_mp_config(), tmp_path)

    assert path is not None
    assert path == tmp_path.joinpath(*_FIGURE_REL)
    assert path.exists()
    assert path.stat().st_size > 0


def test_writes_a_figure_from_snapshots(tmp_path):
    _write_snapshots(tmp_path)

    path = write_length_scale_figure(_init_from_file_config(tmp_path), tmp_path)

    assert path is not None
    assert path.exists()


def test_output_directory_is_independent_of_the_snapshot_directory(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_snapshots(run_dir)
    out_dir = tmp_path / "elsewhere"

    path = write_length_scale_figure(_mp_config(), out_dir, run_dir=run_dir)

    assert path is not None
    assert path == out_dir.joinpath(*_FIGURE_REL)
    assert path.exists()


def test_renders_without_gravity(tmp_path):
    """No gravity means no Bo to caption; the figure must still be written."""
    path = write_length_scale_figure(_mp_config(gravity_force=None), tmp_path)

    assert path is not None
    assert path.exists()


def test_returns_none_when_no_region_can_be_resolved(tmp_path):
    config = _mp_config(initialisation={"centres": [], "radii": []})

    assert write_length_scale_figure(config, tmp_path) is None


def test_counted_panel_reports_the_area_the_overview_uses(tmp_path):
    """Panel one carries the literal number behind L_eff, not a re-measurement."""
    config = _init_from_file_config(tmp_path)

    panel = _counted_panel(config)

    assert panel is not None
    assert panel.area == pytest.approx(20.0 * 5.0)
    assert panel.rho_mean == pytest.approx(0.73)  # measured, not the config's 0.75
    assert panel.drho == pytest.approx(0.36)
    assert panel.mask is not None
    assert np.count_nonzero(panel.mask) == panel.area


def test_counted_panel_falls_back_to_the_analytic_circle():
    panel = _counted_panel(_mp_config())

    assert panel is not None
    assert panel.rho is None
    assert panel.analytic is not None
    assert panel.rho_mean is None
    assert "analytic" in panel.title


def test_snapshot_panels_measure_each_snapshot_separately(tmp_path):
    _write_snapshots(tmp_path)

    panels = _snapshot_panels(tmp_path)

    assert len(panels) == 2
    assert [panel.timestep for panel in panels] == [0, 100]
    for panel in panels:
        assert panel.rho_mean == pytest.approx(0.73)
        assert panel.area == pytest.approx(100.0)


def test_snapshot_panels_show_a_single_snapshot_once(tmp_path):
    _write_snapshots(tmp_path, steps=(0,))

    assert len(_snapshot_panels(tmp_path)) == 1


def test_snapshot_panels_are_empty_without_a_data_directory(tmp_path):
    assert _snapshot_panels(tmp_path) == []


def test_analytic_region_marks_the_wall_the_area_clips_against():
    """R = 0.4*20 = 8, centre 2 lu above the bottom wall -> that wall is nearest."""
    region = _analytic_region(_mp_config())

    assert region is not None
    assert region.radius == pytest.approx(8.0)
    assert region.wall_axis == "y"
    assert region.wall_position == pytest.approx(0.0)


def test_analytic_region_returns_none_without_init_geometry():
    assert _analytic_region(_mp_config(initialisation={})) is None
