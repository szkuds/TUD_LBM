"""Focused tests for ConfigAdapter shared utilities."""

from __future__ import annotations
from src.config.adapter_base import ConfigAdapter
from src.config.simulation_config import SimulationConfig


def test_build_sections_skips_multiphase_bucket_for_single_phase_and_merges_extra():
    cfg = SimulationConfig(
        sim_type="single_phase",
        grid_shape=(8, 8),
        eos="double-well",
        kappa=0.1,
        extra={"tag": ("a", "b")},
    )

    sections = ConfigAdapter.build_sections(cfg)

    assert "multiphase" not in sections
    assert sections["simulation_type"]["type"] == "single_phase"
    assert sections["simulation_type"]["tag"] == ["a", "b"]


def test_build_sections_flattens_force_dict_into_force_section():
    cfg = SimulationConfig(
        grid_shape=(8, 8),
        gravity_force={"force_g": 5e-7, "inclination_angle_deg": 45},
    )

    sections = ConfigAdapter.build_sections(cfg)

    assert sections["gravity_force"]["force_g"] == 5e-7
    assert sections["gravity_force"]["inclination_angle_deg"] == 45
