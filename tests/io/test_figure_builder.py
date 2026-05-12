"""Tests for FigureBuilder and plotting subsystem."""

from __future__ import annotations
import tempfile
import warnings
from pathlib import Path
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
