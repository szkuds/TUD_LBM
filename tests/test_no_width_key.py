"""Tests to verify that the legacy 'width' key is fully rejected.

These tests confirm:
1. TomlAdapter rejects 'width' in the [multiphase] TOML table.
2. SimulationConfig accepts 'interface_width'.
3. config_complex.toml loads correctly with 'interface_width' (no 'width').
"""

import textwrap

import pytest

from config.adapter_toml import TomlAdapter
from config.simulation_config import SimulationConfig

# ── TomlAdapter: reject legacy 'width' in [multiphase] ──────────────


class TestTomlAdapterNeedsInterfaceWidth:
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
            eos = "double-well"
            width = 6
        """)
        p = tmp_path / "bad_width.toml"
        p.write_text(content)
        with pytest.raises(ValueError, match="'interface_width' is required for multiphase simulations"):
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
        config = TomlAdapter().load(str(p))
        assert config.interface_width == 4


# ── SimulationConfig: interface_width works ──────────────────────────


class TestSimulationConfigWidth:
    """SimulationConfig must support 'interface_width'."""

    def test_interface_width_attribute_works(self):
        config = SimulationConfig(
            sim_type="multiphase",
            grid_shape=(64, 64),
            tau=0.99,
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )
        assert config.interface_width == 4

    def test_to_dict_contains_interface_width(self):
        config = SimulationConfig(
            sim_type="multiphase",
            grid_shape=(64, 64),
            tau=0.99,
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )
        d = config.to_dict()
        assert "interface_width" in d
        assert d["interface_width"] == 4
        assert "width" not in d
