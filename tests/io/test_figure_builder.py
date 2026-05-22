"""Tests for FigureBuilder and plotting subsystem."""

from __future__ import annotations
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from tud_lbm.config import SimulationConfig
from tud_lbm.io.plotting import FigureBuilder


class TestFigureBuilderGuardFor3D:
    """Test the 3D simulation guard in FigureBuilder.__init__."""

    def test_2d_simulation_creates_operators(self):
        """FigureBuilder should initialize normally for 2D simulations (nz=1)."""
        config = SimulationConfig(
            grid_shape=(64, 64, 1),
            tau=0.8,
            nt=100,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = FigureBuilder(config, run_dir=tmpdir)
            # For 2D (nz=1), operators should be initialized (or empty if no plotting registered)
            # The key is that _operators is a list (not None) and no warning is raised
            assert isinstance(builder._operators, list)

    def test_3d_simulation_nz_2_raises_warning(self):
        """FigureBuilder should warn and leave _operators empty for 3D simulations (nz=2)."""
        config = SimulationConfig(
            grid_shape=(64, 64, 2),
            tau=0.8,
            nt=100,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                builder = FigureBuilder(config, run_dir=tmpdir)

                # Check that a warning was raised
                assert len(w) >= 1
                assert issubclass(w[-1].category, UserWarning)
                assert "Plotting is not supported for 3D simulations" in str(w[-1].message)
                assert "(nz=2)" in str(w[-1].message)

            # Operators should be empty (guard returned early)
            assert builder._operators == []

    def test_3d_simulation_nz_4_raises_warning(self):
        """FigureBuilder should warn and leave _operators empty for larger 3D simulations (nz=4)."""
        config = SimulationConfig(
            grid_shape=(64, 64, 4),
            tau=0.8,
            nt=100,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                builder = FigureBuilder(config, run_dir=tmpdir)

                # Check that a warning was raised
                assert len(w) >= 1
                assert issubclass(w[-1].category, UserWarning)
                assert "Plotting is not supported for 3D simulations" in str(w[-1].message)
                assert "(nz=4)" in str(w[-1].message)

            # Operators should be empty
            assert builder._operators == []

    def test_3d_simulation_build_returns_none(self):
        """When operators are empty due to 3D guard, build() should warn and return None."""
        config = SimulationConfig(
            grid_shape=(64, 64, 2),
            tau=0.8,
            nt=100,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                builder = FigureBuilder(config, run_dir=tmpdir)

            # Simulate calling build() with dummy data
            data = {}
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = builder.build(data, timestep=0)

                # build() should warn about no operators
                assert len(w) >= 1
                assert "no operators have data" in str(w[-1].message).lower()
                assert result is None

    def test_3d_simulation_build_all_returns_empty_list(self):
        """When operators are empty due to 3D guard, build_all() should return []."""
        config = SimulationConfig(
            grid_shape=(64, 64, 2),
            tau=0.8,
            nt=100,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                builder = FigureBuilder(config, run_dir=tmpdir)

            # build_all() should return empty list (data dir doesn't exist, but that's OK)
            result = builder.build_all()
            assert result == []

    def test_plot_dir_not_created_for_3d(self):
        """For 3D simulations, plot_dir should not be created (guard returns early)."""
        config = SimulationConfig(
            grid_shape=(64, 64, 2),
            tau=0.8,
            nt=100,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                FigureBuilder(config, run_dir=tmpdir)

            # plot directory should not have been created
            plot_dir = Path(tmpdir) / "plots"
            assert not plot_dir.exists()

    def test_plot_dir_created_for_2d(self):
        """For 2D simulations, plot_dir should be created."""
        config = SimulationConfig(
            grid_shape=(64, 64, 1),
            tau=0.8,
            nt=100,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            FigureBuilder(config, run_dir=tmpdir)

            # plot directory should have been created
            plot_dir = Path(tmpdir) / "plots"
            assert plot_dir.exists()


def test_build_all_calls_build_analysis_once(tmp_path):
    run_dir = tmp_path / "run"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    np.savez(data_dir / "timestep_1.npz", rho=np.ones((4, 4, 1, 1, 1)), u=np.zeros((4, 4, 1, 1, 2)))
    np.savez(data_dir / "timestep_2.npz", rho=np.ones((4, 4, 1, 1, 1)), u=np.zeros((4, 4, 1, 1, 2)))

    config = SimulationConfig(plot_fields=["density", "max_velocity"])
    builder = FigureBuilder(config, run_dir=run_dir)
    builder.build_analysis = MagicMock(return_value=[])

    builder.build_all()
    builder.build_analysis.assert_called_once()


def test_build_field_only_does_not_create_analysis_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    config = SimulationConfig(plot_fields=["density"])
    builder = FigureBuilder(config, run_dir=run_dir)

    data = {"rho": np.ones((8, 8, 1, 1, 1))}
    out = builder.build(data, timestep=0)

    assert out is not None
    assert out.exists()
    assert not (run_dir / "plots" / "analysis").exists()


def test_build_csv_runs_when_simulation_csv_selected(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    config = SimulationConfig(plot_fields=["simulation_csv"])
    builder = FigureBuilder(config, run_dir=run_dir)

    called = {"n": 0}

    def _fake_export(run_dir_arg, config_arg):
        called["n"] += 1
        assert Path(run_dir_arg) == run_dir
        assert config_arg == config
        return run_dir / "simulation_data.csv"

    monkeypatch.setattr("tud_lbm.io.plotting.analysis.build_simulation_csv", _fake_export)

    out = builder.build_csv()
    assert called["n"] == 1
    assert out == run_dir / "simulation_data.csv"


def test_build_csv_skips_when_simulation_csv_not_selected(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    config = SimulationConfig(plot_fields=["density"])
    builder = FigureBuilder(config, run_dir=run_dir)

    def _fail_if_called(*_args, **_kwargs):
        msg = "build_simulation_csv should not be called"
        raise AssertionError(msg)

    monkeypatch.setattr("tud_lbm.io.plotting.analysis.build_simulation_csv", _fail_if_called)

    out = builder.build_csv()
    assert out is None


def test_layout_falls_back_to_squareish_grid_for_five_panels():
    assert FigureBuilder.layout(5) == (3, 2)


def test_build_analysis_renders_error_panel_when_operator_fails(tmp_path):
    run_dir = tmp_path / "run"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    np.savez(data_dir / "timestep_1.npz", rho=np.ones((4, 4, 1, 1, 1)), u=np.zeros((4, 4, 1, 1, 2)))

    config = SimulationConfig(plot_fields=["density"])
    builder = FigureBuilder(config, run_dir=run_dir)

    class _FailingAnalysisOperator:
        name = "failing_analysis"

        def compute(self, _files):
            msg = "boom"
            raise RuntimeError(msg)

        def render(self, _ax, _precomputed):
            return None

    builder._analysis_operators = [_FailingAnalysisOperator()]

    saved = builder.build_analysis()

    assert len(saved) == 1
    assert saved[0].name == "failing_analysis.png"
    assert saved[0].exists()
