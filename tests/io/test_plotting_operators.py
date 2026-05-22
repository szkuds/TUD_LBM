"""Tests for plotting operators data shape handling."""

from __future__ import annotations
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from tud_lbm.config import SimulationConfig
from tud_lbm.io.plotting import FigureBuilder
from tud_lbm.io.plotting.force import ExternalForcePlotOperator
from tud_lbm.io.plotting.force import ForcePlotOperator


class TestPlottingOperatorsShapeHandling:
    """Test that plotting operators correctly handle 2D data slicing."""

    def test_density_operator_2d_shape(self):
        """Density operator should produce correct 2D array from data."""
        from tud_lbm.io.plotting.density import DensityPlotOperator

        config = SimulationConfig(
            grid_shape=(100, 100, 1),
            tau=0.8,
            nt=100,
        )

        rho_data = np.ones((100, 100, 1, 9, 1))
        data = {"rho": rho_data}

        op = DensityPlotOperator(config)
        assert op.is_available(data)

        fig, ax = plt.subplots()
        try:
            op(ax, data, timestep=0)
        finally:
            plt.close(fig)

    def test_velocity_operator_2d_shape(self):
        """Velocity operator should produce correct 2D array from data."""
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
        rho_data = np.ones((100, 100, 1, 9, 1))
        data = {"rho": rho_data}

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = FigureBuilder(config, run_dir=tmpdir)
            # Should not raise shape error
            result = builder.build(data, timestep=0)
            assert result is not None
            assert result.exists()


def test_force_plot_operator_availability_and_render():
    """Force operator should be available for force data and render labels/title."""
    config = SimulationConfig(grid_shape=(16, 16, 1), tau=0.8, nt=2)
    op = ForcePlotOperator(config)

    force = np.zeros((16, 16, 1, 1, 2), dtype=float)
    force[:, :, 0, 0, 0] = 1.0
    data = {"force": force}

    assert op.is_available(data)
    assert not op.is_available({})

    fig, ax = plt.subplots()
    try:
        op(ax, data, timestep=3)
        assert ax.get_title() == "Total force  t=3"
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
        assert len(ax.images) == 1
    finally:
        plt.close(fig)


def test_external_force_plot_operator_availability_and_render():
    """External-force operator should use the dedicated field and render."""
    config = SimulationConfig(grid_shape=(16, 16, 1), tau=0.8, nt=2)
    op = ExternalForcePlotOperator(config)

    force_ext = np.zeros((16, 16, 1, 1, 2), dtype=float)
    force_ext[:, :, 0, 0, 1] = 2.0
    data = {"force_ext": force_ext}

    assert op.is_available(data)
    assert not op.is_available({"force": force_ext})

    fig, ax = plt.subplots()
    try:
        op(ax, data, timestep=4)
        assert ax.get_title() == "External force  t=4"
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
        assert len(ax.images) == 1
    finally:
        plt.close(fig)
