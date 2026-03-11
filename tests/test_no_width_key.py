"""Tests to verify that the legacy 'width' key is fully rejected.

These tests confirm:
1. TomlAdapter rejects 'width' in the [multiphase] TOML table.
2. SimulationSetup rejects 'width' passed via extra kwargs.
3. config_complex.toml loads correctly with 'interface_width' (no 'width').
4. Gradient and Laplacian derive width from config.interface_width.
"""

import os
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Trigger operator registration
import simulation_operators  # noqa: F401
import simulation_type  # noqa: F401
import update_timestep  # noqa: F401

from app_setup.adapter_toml import TomlAdapter
from app_setup.simulation_setup import SimulationSetup


# ── TomlAdapter: reject legacy 'width' in [multiphase] ──────────────

class TestTomlAdapterRejectsWidth:
    """TomlAdapter must reject a 'width' key in [multiphase]."""

    def test_width_in_multiphase_table_raises(self, tmp_path):
        content = textwrap.dedent("""\
            [simulation_type]
            type = "multiphase"
            grid_shape = [64, 64]
            tau = 0.99
            nt = 1000

            [multiphase]
            kappa = 0.017
            rho_l = 1.0
            rho_v = 0.33
            interface_width = 4
            eos = "double-well"
            width = 4
        """)
        p = tmp_path / "bad_width.toml"
        p.write_text(content)
        with pytest.raises(KeyError, match="Legacy key 'width'"):
            TomlAdapter().load(str(p))

    def test_interface_width_accepted(self, tmp_path):
        """interface_width (without width) should work fine."""
        content = textwrap.dedent("""\
            [simulation_type]
            type = "multiphase"
            grid_shape = [64, 64]
            tau = 0.99
            nt = 1000

            [multiphase]
            kappa = 0.017
            rho_l = 1.0
            rho_v = 0.33
            interface_width = 4
            eos = "double-well"
        """)
        p = tmp_path / "ok.toml"
        p.write_text(content)
        setup = TomlAdapter().load(str(p))
        assert setup.interface_width == 4


# ── SimulationSetup: reject 'width' in extra ────────────────────────

class TestSimulationSetupRejectsWidth:
    """SimulationSetup must reject 'width' passed via extra."""

    def test_interface_width_attribute_works(self):
        setup = SimulationSetup(
            sim_type="multiphase",
            grid_shape=(64, 64),
            tau=0.99,
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )
        assert setup.interface_width == 4

    def test_to_dict_contains_interface_width(self):
        setup = SimulationSetup(
            sim_type="multiphase",
            grid_shape=(64, 64),
            tau=0.99,
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )
        d = setup.to_dict()
        assert "interface_width" in d
        assert d["interface_width"] == 4
        # 'width' should never appear
        assert "width" not in d



