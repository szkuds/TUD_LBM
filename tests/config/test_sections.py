"""The mapping between the nested config layout and the config fields."""

from __future__ import annotations
from src.config.sections import build_sections
from src.config.simulation_config import SimulationConfig


def test_build_sections_skips_multiphase_bucket_for_single_phase_and_merges_extra():
    cfg = SimulationConfig(grid_shape=(8, 8), extra={"tag": ("a", "b")})

    sections = build_sections(cfg)

    assert "multiphase" not in sections
    assert "type" not in sections["simulation_type"]
    assert sections["simulation_type"]["tag"] == ["a", "b"]


def test_build_sections_flattens_force_dict_into_force_section():
    cfg = SimulationConfig(forces={"gravity_force": {"force_g": 5e-7, "inclination_angle_deg": 45}}, grid_shape=(8, 8))

    sections = build_sections(cfg)

    assert sections["gravity_force"]["force_g"] == 5e-7
    assert sections["gravity_force"]["inclination_angle_deg"] == 45
