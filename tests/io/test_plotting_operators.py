"""Tests for plotting operators data shape handling."""

from __future__ import annotations
import dataclasses
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from src.config import SimulationConfig
from src.simulation_io.plotting import FigureBuilder
from src.simulation_io.plotting.force import ExternalForcePlotOperator
from src.simulation_io.plotting.force import ForcePlotOperator
from src.simulation_io.plotting.pressure import BulkPressurePlotOperator
from src.simulation_io.plotting.pressure import TotalPressurePlotOperator

_KAPPA = 0.01
_RHO_L = 1.0
_RHO_V = 0.1
_WIDTH = 4


def _multiphase_config(**overrides: object) -> SimulationConfig:
    """A minimal double-well multiphase config for the pressure operators."""
    base = SimulationConfig(
        sim_type="multiphase",
        grid_shape=(16, 16, 1),
        tau=0.8,
        nt=2,
        eos="double-well",
        kappa=_KAPPA,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=_WIDTH,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


class TestPlottingOperatorsShapeHandling:
    """Test that plotting operators correctly handle 2D data slicing."""

    def test_density_operator_2d_shape(self):
        """Density operator should produce correct 2D array from data."""
        from src.simulation_io.plotting.density import DensityPlotOperator

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
        from src.simulation_io.plotting.velocity import VelocityPlotOperator

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


def test_bulk_pressure_operator_matches_eos_reference():
    """Bulk pressure panel should render p_0(rho) straight from the EOS function."""
    from src.operators.macroscopic.eos._double_well import _pressure_double_well

    config = _multiphase_config()
    op = BulkPressurePlotOperator(config)

    rho = np.full((16, 16, 1, 1, 1), 0.7)
    data = {"rho": rho}
    assert op.is_available(data)

    beta = 8.0 * _KAPPA / (float(_WIDTH) ** 2 * (_RHO_L - _RHO_V) ** 2)
    expected = np.asarray(_pressure_double_well(rho[:, :, 0, 0, 0], beta, _RHO_L, _RHO_V)).T
    np.testing.assert_allclose(op._pressure_2d(data), expected)

    fig, ax = plt.subplots()
    try:
        op(ax, data, timestep=5)
        assert ax.get_title() == "Bulk pressure  t=5"
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
        assert len(ax.images) == 1
    finally:
        plt.close(fig)


def test_total_pressure_reduces_to_bulk_for_uniform_density():
    """With no density gradient the kappa terms vanish, so total == bulk."""
    config = _multiphase_config()
    rho = np.full((16, 16, 1, 1, 1), 0.7)
    data = {"rho": rho}

    bulk = BulkPressurePlotOperator(config)._pressure_2d(data)
    total = TotalPressurePlotOperator(config)._pressure_2d(data)

    assert total.shape == bulk.shape == (16, 16)
    np.testing.assert_allclose(total, bulk, atol=1e-12)


def test_total_pressure_flattens_the_interface_swing():
    """The kappa terms should cancel most of the p_0 excursion across an interface."""
    config = _multiphase_config(
        grid_shape=(64, 64, 1),
        bc_config={"top": "periodic", "bottom": "periodic", "left": "periodic", "right": "periodic"},
    )

    x, y = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    radius = np.sqrt((x - 32.0) ** 2 + (y - 32.0) ** 2)
    profile = 0.5 * (_RHO_L + _RHO_V) - 0.5 * (_RHO_L - _RHO_V) * np.tanh(2.0 * (radius - 16.0) / _WIDTH)
    data = {"rho": profile[:, :, None, None, None]}

    bulk = BulkPressurePlotOperator(config)._pressure_2d(data)
    total = TotalPressurePlotOperator(config)._pressure_2d(data)

    assert np.ptp(total) < np.ptp(bulk)


def test_pressure_operators_unavailable_without_supported_eos():
    """Single-phase runs and unsupported EOS must drop the panel, not error in it."""
    data = {"rho": np.full((16, 16, 1, 1, 1), 0.7)}

    single_phase = SimulationConfig(grid_shape=(16, 16, 1), tau=0.8, nt=2)
    assert not BulkPressurePlotOperator(single_phase).is_available(data)
    assert not TotalPressurePlotOperator(single_phase).is_available(data)

    multiphase = _multiphase_config()
    assert not BulkPressurePlotOperator(multiphase).is_available({})


def test_pressure_operators_are_opt_in():
    """Pressure panels stay out of the default figure until named in plot_fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        default_names = {op.name for op in FigureBuilder(_multiphase_config(), run_dir=tmpdir).field_operators}
        assert "density" in default_names
        assert {"pressure", "pressure_total"}.isdisjoint(default_names)

        explicit = _multiphase_config(plot_fields=["density", "pressure", "pressure_total"])
        explicit_names = {op.name for op in FigureBuilder(explicit, run_dir=tmpdir).field_operators}
        assert explicit_names == {"density", "pressure", "pressure_total"}
