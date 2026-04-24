"""Tests for plotting operators data shape handling."""

from __future__ import annotations
import tempfile
import numpy as np
from tud_lbm.config import SimulationConfig
from tud_lbm.io.plotting import FigureBuilder


class TestPlottingOperatorsShapeHandling:
    """Test that plotting operators correctly handle 2D data slicing."""

    def test_density_operator_2d_shape(self):
        """Density operator should produce correct 2D array from data."""
        import matplotlib.pyplot as plt
        from tud_lbm.io.plotting.density import DensityPlotOperator

        config = SimulationConfig(
            grid_shape=(100, 100, 1),
            tau=0.8,
            nt=100,
        )

        # Create dummy data: (nx, ny, nz, q, d) for rho where q=9 (D2Q9), d=1 (scalar)
        rho_data = np.ones((100, 100, 1, 9, 1))
        data = {"rho": rho_data}

        op = DensityPlotOperator(config)
        assert op.is_available(data)

        fig, ax = plt.subplots()
        try:
            op(ax, data, timestep=0)
            # If we get here without error, the shape was correct
            assert True
        finally:
            plt.close(fig)

    def test_velocity_operator_2d_shape(self):
        """Velocity operator should produce correct 2D array from data."""
        import matplotlib.pyplot as plt
        from tud_lbm.io.plotting.velocity import VelocityPlotOperator

        config = SimulationConfig(
            grid_shape=(100, 100, 1),
            tau=0.8,
            nt=100,
        )

        # Create dummy data: (nx, ny, nz, q, d) for u where q=9 (D2Q9), d=2 (2D velocity)
        u_data = np.ones((100, 100, 1, 9, 2))
        data = {"u": u_data}

        op = VelocityPlotOperator(config)
        assert op.is_available(data)

        fig, ax = plt.subplots()
        try:
            op(ax, data, timestep=0)
            # If we get here without error, the shape was correct
            assert True
        finally:
            plt.close(fig)

    def test_figure_builder_with_density_data(self):
        """FigureBuilder.build() should work with 2D density data."""
        config = SimulationConfig(
            grid_shape=(100, 100, 1),
            tau=0.8,
            nt=100,
            plot_fields=["density"],
        )

        # Create dummy data with correct shapes: (nx, ny, nz, q, d)
        rho_data = np.random.Generator(100, 100, 1, 9, 1)
        data = {"rho": rho_data}

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = FigureBuilder(config, run_dir=tmpdir)
            # Should not raise shape error
            result = builder.build(data, timestep=0)
            assert result is not None
            assert result.exists()
