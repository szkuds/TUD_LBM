"""The mapping between the nested config layout and the config fields."""

from __future__ import annotations
import pytest
from src.config.sections import build_sections
from src.config.sections import merge_sections
from src.config.sections import serialise_safe
from src.config.simulation_config import SimulationConfig
from src.config.simulation_config import get_whole_section_fields

_SECTION_FIELDS = get_whole_section_fields()


@pytest.mark.parametrize("section", _SECTION_FIELDS)
def test_merge_sections_stores_a_section_table_on_the_field_of_its_name(section):
    raw = {"simulation_type": {"grid_shape": [8, 8]}, section: {"key": 1}}

    assert merge_sections(raw)[section] == {"key": 1}


def test_section_fields_cover_the_sections_of_the_layout():
    assert {"multiphase", "boundary_conditions", "wetting", "hysteresis", "chemical_step"} <= set(_SECTION_FIELDS)


@pytest.mark.parametrize("table", ["extra", "forces"])
def test_an_assembled_field_is_not_a_whole_section(table):
    assert table not in _SECTION_FIELDS


@pytest.mark.parametrize("table", ["extra", "forces"])
def test_merge_sections_does_not_read_an_assembled_field_as_a_table(table):
    raw = {"simulation_type": {"grid_shape": [8, 8]}, table: {"key": 1}}

    assert table not in merge_sections(raw)


def test_sections_round_trip_through_build_and_merge():
    cfg = SimulationConfig(
        grid_shape=(16, 16),
        multiphase={"eos": "double-well", "kappa": 0.02, "rho_l": 1.0, "rho_v": 0.1, "interface_width": 4},
        boundary_conditions={"top": "bounce-back", "bottom": "bounce-back"},
        initialisation={"centres": [[0.5, 0.5]]},
    )

    flat = merge_sections(build_sections(cfg))

    for section in ("multiphase", "boundary_conditions", "initialisation"):
        assert flat[section] == serialise_safe(getattr(cfg, section))


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
