"""Test that tud_lbm package structure is complete.

This test verifies the Phase 1 checkpoint of the folder restructuring refactoring,
ensuring that the new tud_lbm/ package hierarchy exists and has proper exports.
"""

import pytest
import importlib.util


class TestTudLbmStructure:
    """Verify tud_lbm package modules exist and have expected exports."""

    def test_lattice_module_exists(self):
        """Test that lattice module can be imported."""
        spec = importlib.util.find_spec("tud_lbm.lattice")
        assert spec is not None, "tud_lbm.lattice module not found"

    def test_config_module_exists(self):
        """Test that config module can be imported."""
        spec = importlib.util.find_spec("tud_lbm.config")
        assert spec is not None, "tud_lbm.config module not found"

    def test_operators_module_exists(self):
        """Test that operators module can be imported."""
        spec = importlib.util.find_spec("tud_lbm.operators")
        assert spec is not None, "tud_lbm.operators module not found"

    def test_pipeline_module_exists(self):
        """Test that pipeline module can be imported."""
        spec = importlib.util.find_spec("tud_lbm.pipeline")
        assert spec is not None, "tud_lbm.pipeline module not found"

    def test_pipeline_state_module_exists(self):
        """Test that pipeline.state module exists on disk."""
        import os
        state_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "tud_lbm",
            "pipeline",
            "state",
            "__init__.py"
        )
        assert os.path.exists(state_path), f"State module __init__.py not found at {state_path}"

    def test_registry_module_exists(self):
        """Test that registry module can be imported."""
        spec = importlib.util.find_spec("tud_lbm.registry")
        assert spec is not None, "tud_lbm.registry module not found"

    def test_io_module_exists(self):
        """Test that io module can be imported."""
        spec = importlib.util.find_spec("tud_lbm.io")
        assert spec is not None, "tud_lbm.io module not found"

    def test_cli_module_exists(self):
        """Test that cli module can be imported."""
        spec = importlib.util.find_spec("tud_lbm.cli")
        assert spec is not None, "tud_lbm.cli module not found"

    def test_readers_module_exists(self):
        """Test that readers module can be imported."""
        spec = importlib.util.find_spec("tud_lbm.readers")
        assert spec is not None, "tud_lbm.readers module not found"

    def test_operator_subcategories_exist(self):
        """Test that all operator subcategories exist."""
        subcategories = [
            "collision", "streaming", "equilibrium", "macroscopic",
            "boundary", "differential", "force", "wetting", "initialise"
        ]
        for sub in subcategories:
            spec = importlib.util.find_spec(f"tud_lbm.operators.{sub}")
            assert spec is not None, f"tud_lbm.operators.{sub} module not found"

    def test_tud_lbm_package_has_init(self):
        """Test that main tud_lbm __init__.py exists."""
        spec = importlib.util.find_spec("tud_lbm")
        assert spec is not None, "tud_lbm package not found"
        assert spec.origin is not None, "tud_lbm has no __init__.py"
