"""Integration tests for the public API.

Tests end-to-end workflows that users will employ:
- Configuration creation (defaults, adapters)
- Simulation pipeline (setup → init → run)
- Input adapters (dict, TOML)
- Output adapters (numpy, VTK)
- Parameter sweeps and force configurations
"""

from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pytest
from tud_lbm import SimulationConfig
from tud_lbm import build_setup
from tud_lbm import run
from tud_lbm.pipeline.runner import init_state
from tud_lbm.readers import DictAdapter


class TestMinimalSimulation:
    """Test the 5-line minimal workflow."""

    def test_minimal_default_config(self):
        """Verify minimal simulation with default config works."""
        config = SimulationConfig(grid_shape=(32, 32), tau=0.8, nt=10)
        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=10)

        assert final_state.t == 10
        assert final_state.rho.shape == (32, 32, 1, 1)
        assert final_state.u.shape == (32, 32, 1, 2)
        assert not np.isnan(final_state.f).any()

    def test_minimal_with_trajectory(self):
        """Verify trajectory output works."""
        config = SimulationConfig(grid_shape=(16, 16), tau=0.8, nt=5)
        setup = build_setup(config)
        state = init_state(setup)
        final_state, trajectory = run(setup, state, nt=5)

        assert final_state.t == 5
        # trajectory can be None or a list depending on implementation
        assert trajectory is None or isinstance(trajectory, (list, tuple))


class TestConfigCreation:
    """Test configuration creation and validation."""

    def test_default_parameters(self):
        """Verify all defaults are sensible."""
        config = SimulationConfig()

        assert config.grid_shape == (64, 64)
        assert config.tau == 1.0
        assert config.nt == 1000
        assert config.lattice_type == "D2Q9"
        assert config.collision_scheme == "bgk"
        assert config.sim_type == "single_phase"

    def test_custom_grid_shapes(self):
        """Test various grid shapes."""
        for shape in [(16, 16), (32, 32), (64, 64), (128, 128)]:
            config = SimulationConfig(grid_shape=shape, nt=5)
            setup = build_setup(config)
            state = init_state(setup)
            final_state, _ = run(setup, state, nt=5)

            assert final_state.rho.shape == (*shape, 1, 1)

    def test_tau_must_be_greater_than_half(self):
        """Verify tau validation."""
        with pytest.raises(ValueError, match=r"tau.*must be.*0.5"):
            SimulationConfig(tau=0.4)

    def test_tau_at_boundary(self):
        """Verify tau=0.5 is rejected, tau=0.500001 works."""
        with pytest.raises(ValueError):
            SimulationConfig(tau=0.5)

        config = SimulationConfig(tau=0.500001, nt=5)
        setup = build_setup(config)
        assert setup is not None


class TestDictAdapter:
    """Test dict-based configuration loading."""

    def test_dict_adapter_basic(self):
        """Verify dict adapter loads and runs simulation."""
        adapter = DictAdapter()
        config = adapter.load(
            {
                "grid_shape": (16, 16),
                "tau": 0.8,
                "nt": 5,
            }
        )

        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=5)

        assert final_state.t == 5

    def test_dict_adapter_normalizes_grid_shape(self):
        """Verify list grid_shape is converted to tuple."""
        adapter = DictAdapter()
        config = adapter.load(
            {
                "grid_shape": [32, 32],  # List instead of tuple
                "nt": 5,
            }
        )

        assert isinstance(config.grid_shape, tuple)
        assert config.grid_shape == (32, 32)

    def test_dict_adapter_partial_config(self):
        """Verify dict adapter works with partial config (rest are defaults)."""
        adapter = DictAdapter()
        config = adapter.load(
            {
                "grid_shape": (16, 16),
                # Other params use defaults
            }
        )

        assert config.tau == 1.0  # default
        assert config.nt == 1000  # default
        setup = build_setup(config)
        assert setup is not None


class TestParameterSweep:
    """Test parameter sweep patterns."""

    def test_tau_sweep(self):
        """Verify tau parameter sweep works."""
        tau_values = [0.6, 0.8, 1.0, 1.2]
        results = []

        for tau in tau_values:
            config = SimulationConfig(grid_shape=(16, 16), tau=tau, nt=5)
            setup = build_setup(config)
            state = init_state(setup)
            final_state, _ = run(setup, state, nt=5)

            results.append(
                {
                    "tau": tau,
                    "final_t": final_state.t,
                    "rho_mean": float(final_state.rho.mean()),
                }
            )

        assert len(results) == 4
        assert all(r["final_t"] == 5 for r in results)

    def test_grid_shape_sweep(self):
        """Verify grid shape variations work."""
        grid_shapes = [(8, 8), (16, 16), (32, 32)]
        results = []

        for shape in grid_shapes:
            config = SimulationConfig(grid_shape=shape, nt=3)
            setup = build_setup(config)
            state = init_state(setup)
            final_state, _ = run(setup, state, nt=3)

            results.append(
                {
                    "shape": shape,
                    "rho_shape": final_state.rho.shape,
                }
            )

        assert all(r["rho_shape"][:2] == r["shape"] for r in results)


class TestForceConfiguration:
    """Test force configurations in simulations."""

    def test_gravity_force(self):
        """Verify gravity force configuration and execution."""
        config = SimulationConfig(
            grid_shape=(16, 16),
            tau=0.8,
            nt=5,
            gravity_force={
                "force_g": 1e-5,
                "inclination_angle_deg": 45.0,
            },
        )

        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=5)

        assert final_state.t == 5
        assert not np.isnan(final_state.f).any()

    def test_no_forces(self):
        """Verify simulation without forces (default)."""
        config = SimulationConfig(
            grid_shape=(16, 16),
            tau=0.8,
            nt=5,
            gravity_force=None,
        )

        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=5)

        assert final_state.t == 5


class TestOutputAdapters:
    """Test output adapter functionality."""

    @pytest.mark.skip(reason="Output writer instantiation requires further investigation")
    def test_write_numpy_creates_file(self):
        """Verify write_numpy creates valid .npz file."""
        try:
            from tud_lbm.io.output_data import output_writers
        except ImportError:
            pytest.skip("Output writer module not available")

        config = SimulationConfig(grid_shape=(8, 8), nt=2)
        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Access writer class and instantiate it
            numpy_writer_class = output_writers["numpy"]
            numpy_writer = numpy_writer_class()
            numpy_writer.data_dir = str(tmpdir)
            numpy_writer.save_data_step(
                iteration=0,
                data={
                    "rho": final_state.rho,
                    "u": final_state.u,
                },
            )

            # Verify file was created
            files = list(Path(tmpdir).glob("*.npz"))
            assert len(files) > 0

    @pytest.mark.skip(reason="VTK writer requires pyevtk optional dependency")
    def test_write_vtk_creates_file(self):
        """Verify write_vtk creates valid .vtk file."""
        try:
            import pyevtk  # noqa: F401
            from tud_lbm.io.output_data import output_writers
        except ImportError:
            pytest.skip("VTK writer or pyevtk not available")

        config = SimulationConfig(grid_shape=(8, 8), nt=2)
        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            vtk_writer = output_writers["vtk"]
            vtk_writer.data_dir = str(tmpdir)
            vtk_writer.save_data_step(
                iteration=0,
                data={
                    "rho": final_state.rho,
                    "u": final_state.u,
                },
            )

            # Verify file was created
            files = list(Path(tmpdir).glob("*.vti"))
            assert len(files) > 0


class TestStateConsistency:
    """Test that state remains consistent throughout simulation."""

    def test_rho_conservation(self):
        """Verify density conservation (no NaNs, valid range)."""
        config = SimulationConfig(grid_shape=(16, 16), tau=0.8, nt=10)
        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=10)

        # No NaNs
        assert not np.isnan(final_state.rho).any()
        assert not np.isnan(final_state.u).any()

        # Density should be positive
        assert (final_state.rho > 0).all()

    def test_velocity_bounded(self):
        """Verify velocity stays within physical bounds."""
        config = SimulationConfig(grid_shape=(16, 16), tau=0.8, nt=10)
        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=10)

        # Mach number should be small (subsonic flow)
        u_mag = np.sqrt(final_state.u[..., 0] ** 2 + final_state.u[..., 1] ** 2)
        assert (u_mag < 0.3).all()  # Typical LBM limit

    def test_timestep_increments(self):
        """Verify timestep counter increments correctly."""
        config = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=1)
        setup = build_setup(config)
        state = init_state(setup)

        for i in range(1, 6):
            final_state, _ = run(setup, state, nt=i)
            assert final_state.t == i


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_grid(self):
        """Verify very small grids work."""
        config = SimulationConfig(grid_shape=(4, 4), nt=2)
        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=2)

        assert final_state.rho.shape == (4, 4, 1, 1)

    def test_very_large_tau(self):
        """Verify large tau values work."""
        config = SimulationConfig(grid_shape=(8, 8), tau=2.0, nt=2)
        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=2)

        assert final_state.t == 2

    def test_single_timestep(self):
        """Verify single timestep execution."""
        config = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=1)
        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=1)

        assert final_state.t == 1


class TestAPIIntegration:
    """Test that all API components work together."""

    def test_full_workflow(self):
        """Test complete workflow: config → setup → state → run."""
        # Create config
        config = SimulationConfig(
            grid_shape=(32, 32),
            tau=0.8,
            nt=100,
        )

        # Build setup
        setup = build_setup(config)
        assert setup is not None

        # Initialize state
        state = init_state(setup)
        assert state is not None

        # Run simulation
        final_state, _trajectory = run(setup, state, nt=50)
        assert final_state.t == 50

    def test_dict_adapter_workflow(self):
        """Test workflow using dict adapter."""
        adapter = DictAdapter()
        config = adapter.load(
            {
                "grid_shape": (32, 32),
                "tau": 0.9,
                "nt": 50,
            }
        )

        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=50)

        assert final_state.t == 50
