"""Tests for the config adapter *save* (write) side.

Covers:
- ``TomlAdapter.save()`` creates valid TOML files
- Round-trip: save → load preserves field values
- ``ConfigAdapter`` ABC enforces ``save()`` on subclasses
- ``SimulationIO`` respects ``config_file_type``
- ``get_adapter()`` rejects unsupported extensions
"""

import sys
from pathlib import Path
import pytest

# Ensure src/ is on the path so imports work from the tests/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config.adapter_base import ConfigAdapter
from config.adapter_base import get_adapter
from config.adapter_toml import TomlAdapter
from config.simulation_config import SimulationConfig


class TestTomlAdapterSave:
    """Tests for TomlAdapter.save()."""

    def test_save_creates_file(self, tmp_path):
        """save() should produce a file on disk."""
        cfg = SimulationConfig(grid_shape=(8, 8))
        adapter = TomlAdapter()
        dest = tmp_path / "config.toml"
        adapter.save(cfg, str(dest))
        assert dest.exists()

    def test_save_roundtrip_simple(self, tmp_path):
        """A saved config should be loadable and match the original values."""
        cfg = SimulationConfig(grid_shape=(16, 16), tau=0.8, nt=500)
        adapter = TomlAdapter()
        dest = tmp_path / "config.toml"
        adapter.save(cfg, str(dest))

        loaded = TomlAdapter().load(str(dest))
        assert loaded.grid_shape == (16, 16)
        assert loaded.tau == 0.8
        assert loaded.nt == 500

    def test_save_roundtrip_preserves_sim_type(self, tmp_path):
        """sim_type should survive a save/load round-trip."""
        cfg = SimulationConfig(grid_shape=(10, 10), sim_type="single_phase")
        adapter = TomlAdapter()
        dest = tmp_path / "config.toml"
        adapter.save(cfg, str(dest))

        loaded = TomlAdapter().load(str(dest))
        assert loaded.sim_type == "single_phase"

    def test_save_roundtrip_multiphase(self, tmp_path):
        """Multiphase fields should round-trip correctly."""
        cfg = SimulationConfig(
            grid_shape=(64, 64),
            sim_type="multiphase",
            tau=0.99,
            nt=2000,
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )
        adapter = TomlAdapter()
        dest = tmp_path / "config.toml"
        adapter.save(cfg, str(dest))

        loaded = TomlAdapter().load(str(dest))
        assert loaded.sim_type == "multiphase"
        assert loaded.eos == "double-well"
        assert loaded.kappa == 0.017
        assert loaded.rho_l == 1.0
        assert loaded.rho_v == 0.33
        assert loaded.interface_width == 4

    def test_save_roundtrip_collision_scheme(self, tmp_path):
        """collision_scheme should be preserved."""
        cfg = SimulationConfig(grid_shape=(8, 8), collision_scheme="bgk")
        adapter = TomlAdapter()
        dest = tmp_path / "config.toml"
        adapter.save(cfg, str(dest))

        loaded = TomlAdapter().load(str(dest))
        assert loaded.collision_scheme == "bgk"

    def test_save_roundtrip_output_fields(self, tmp_path):
        """Output-section fields (results_dir, save_fields) should round-trip."""
        cfg = SimulationConfig(
            grid_shape=(8, 8),
            results_dir="/tmp/my_results",
            save_fields=["rho", "u"],
        )
        adapter = TomlAdapter()
        dest = tmp_path / "config.toml"
        adapter.save(cfg, str(dest))

        loaded = TomlAdapter().load(str(dest))
        assert loaded.save_fields == ["rho", "u"]

    def test_save_creates_parent_dirs(self, tmp_path):
        """save() should create parent directories if they don't exist."""
        cfg = SimulationConfig(grid_shape=(8, 8))
        adapter = TomlAdapter()
        dest = tmp_path / "deep" / "nested" / "config.toml"
        adapter.save(cfg, str(dest))
        assert dest.exists()


# ══════════════════════════════════════════════════════════════════════
# ConfigAdapter ABC enforcement
# ══════════════════════════════════════════════════════════════════════


class TestConfigAdapterABCSave:
    """The ABC must force subclasses to implement save()."""

    def test_subclass_must_implement_save(self):
        """A subclass that only implements load() should fail to instantiate."""

        class IncompleteAdapter(ConfigAdapter):
            def load(self, path): ...

            # save() not implemented

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_subclass_must_implement_load(self):
        """A subclass that only implements save() should fail to instantiate."""

        class IncompleteAdapter(ConfigAdapter):
            def save(self, config, path): ...

            # load() not implemented

        with pytest.raises(TypeError):
            IncompleteAdapter()


# ══════════════════════════════════════════════════════════════════════
# get_adapter() dispatch
# ══════════════════════════════════════════════════════════════════════


class TestGetAdapterDispatch:
    """get_adapter() should raise for unsupported extensions."""

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            get_adapter("config.xyz")

    def test_toml_extension_returns_toml_adapter(self):
        adapter = get_adapter("config.toml")
        assert isinstance(adapter, TomlAdapter)


# ══════════════════════════════════════════════════════════════════════
# SimulationIO integration
# ══════════════════════════════════════════════════════════════════════


class TestSimulationIOConfigFileType:
    """SimulationIO should write the config file in the requested format."""

    def test_default_filetype_saves_toml(self, tmp_path):
        from util.io import SimulationIO

        cfg = SimulationConfig(grid_shape=(8, 8))
        io = SimulationIO(
            base_dir=str(tmp_path),
            config=cfg.to_dict(),
            config_file_type=".toml",
            output_format="numpy",
        )
        # The run_dir is timestamped; find the config.toml inside it
        toml_files = list(Path(io.run_dir).glob("config.toml"))
        assert len(toml_files) == 1

    def test_toml_config_is_loadable(self, tmp_path):
        """The config.toml saved by SimulationIO should be loadable."""
        from util.io import SimulationIO

        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.7, nt=200)
        io = SimulationIO(
            base_dir=str(tmp_path),
            config=cfg.to_dict(),
            config_file_type=".toml",
            output_format="numpy",
        )
        config_path = Path(io.run_dir) / "config.toml"
        loaded = TomlAdapter().load(str(config_path))
        assert loaded.grid_shape == (8, 8)
        assert loaded.tau == 0.7
        assert loaded.nt == 200
