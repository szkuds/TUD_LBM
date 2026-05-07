"""Tests for CLI helper functions and edge cases."""

from __future__ import annotations
import pytest

try:
    from tud_lbm.cli.cli import _apply_overrides
    from tud_lbm.cli.cli import _normalize_override_path
    from tud_lbm.cli.cli import _parse_override_argument
    from tud_lbm.cli.cli import _set_nested_override
except ImportError:
    pytest.skip("click or rich dependency not installed", allow_module_level=True)

# =========================================================================
# _parse_override_argument Tests
# =========================================================================


class TestParseOverrideArgument:
    """Tests for parsing --override expressions."""

    def test_parse_simple_number(self):
        path, value = _parse_override_argument("tau=0.7")
        assert path == "tau"
        assert value == 0.7

    def test_parse_integer(self):
        path, value = _parse_override_argument("nt=500")
        assert path == "nt"
        assert value == 500

    def test_parse_scientific_notation(self):
        path, value = _parse_override_argument("force_g=1e-6")
        assert path == "force_g"
        assert value == 1e-6

    def test_parse_quoted_string(self):
        path, value = _parse_override_argument('simulation_name="test_run"')
        assert path == "simulation_name"
        assert value == "test_run"

    def test_parse_boolean_true(self):
        path, value = _parse_override_argument("some_flag=true")
        assert path == "some_flag"
        assert value is True

    def test_parse_boolean_false(self):
        path, value = _parse_override_argument("some_flag=false")
        assert path == "some_flag"
        assert value is False

    def test_parse_array_of_numbers(self):
        path, value = _parse_override_argument("tau=[0.6, 0.7, 0.8]")
        assert path == "tau"
        assert value == [0.6, 0.7, 0.8]

    def test_parse_array_of_strings(self):
        path, value = _parse_override_argument('fields=["rho_t_plus1", "u", "f"]')
        assert path == "fields"
        assert value == ["rho_t_plus1", "u", "f"]

    def test_parse_dotted_path(self):
        path, value = _parse_override_argument("gravity_force.force_g=5e-7")
        assert path == "gravity_force.force_g"
        assert value == 5e-7

    def test_parse_simulation_type_prefix(self):
        path, value = _parse_override_argument('simulation_type.simulation_name="exp1"')
        assert path == "simulation_type.simulation_name"
        assert value == "exp1"

    def test_reject_empty_expression(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _parse_override_argument("")

    def test_reject_whitespace_only_expression(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _parse_override_argument("   ")

    def test_reject_missing_path(self):
        with pytest.raises(ValueError, match="path cannot be empty"):
            _parse_override_argument("=0.7")

    def test_reject_missing_value(self):
        with pytest.raises(ValueError, match="value cannot be empty"):
            _parse_override_argument("tau=")

    def test_reject_invalid_toml_syntax(self):
        with pytest.raises(ValueError, match="invalid override value"):
            _parse_override_argument("tau=not_a_valid_toml_value")

    def test_legacy_format_with_parentheses(self):
        path, value = _parse_override_argument("(tau, 0.7)")
        assert path == "tau"
        assert value == 0.7

    def test_legacy_format_with_override_prefix(self):
        path, value = _parse_override_argument("override(tau, 0.7)")
        assert path == "tau"
        assert value == 0.7


# =========================================================================
# _normalize_override_path Tests
# =========================================================================


class TestNormalizeOverridePath:
    """Tests for normalizing TOML section paths to field names."""

    def test_simple_field_no_normalization(self):
        result = _normalize_override_path("tau")
        assert result == ["tau"]

    def test_nested_field_no_normalization(self):
        result = _normalize_override_path("gravity_force.force_g")
        assert result == ["gravity_force", "force_g"]

    def test_simulation_type_prefix_removed(self):
        result = _normalize_override_path("simulation_type.tau")
        assert result == ["tau"]

    def test_simulation_type_with_nested_field(self):
        result = _normalize_override_path("simulation_type.simulation_name")
        assert result == ["simulation_name"]

    def test_boundary_conditions_mapped_to_bc_config(self):
        result = _normalize_override_path("boundary_conditions.top")
        assert result == ["bc_config", "top"]

    def test_wetting_mapped_to_wetting_config(self):
        result = _normalize_override_path("wetting_t.contact_angle")
        assert result == ["wetting_config", "contact_angle"]

    def test_hysteresis_mapped_to_hysteresis_config(self):
        result = _normalize_override_path("hysteresis.angle_max")
        assert result == ["hysteresis_config", "angle_max"]

    def test_gravity_force_not_mapped(self):
        result = _normalize_override_path("gravity_force.force_g")
        assert result == ["gravity_force", "force_g"]

    def test_electric_force_not_mapped(self):
        result = _normalize_override_path("electric_force.permittivity")
        assert result == ["electric_force", "permittivity"]

    def test_multiphase_prefix_removed(self):
        result = _normalize_override_path("multiphase.kappa")
        assert result == ["kappa"]

    def test_output_prefix_removed(self):
        result = _normalize_override_path("output.results_dir")
        assert result == ["results_dir"]

    def test_reject_empty_path(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _normalize_override_path("")

    def test_reject_dots_only(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _normalize_override_path("...")

    def test_reject_prefix_without_field(self):
        with pytest.raises(ValueError, match="does not reference a field"):
            _normalize_override_path("simulation_type")

    def test_reject_prefix_with_dots_only(self):
        with pytest.raises(ValueError, match="does not reference a field"):
            _normalize_override_path("simulation_type...")

    def test_strip_whitespace_in_segments(self):
        result = _normalize_override_path("gravity_force . force_g")
        assert result == ["gravity_force", "force_g"]


# =========================================================================
# _set_nested_override Tests
# =========================================================================


class TestSetNestedOverride:
    """Tests for setting nested dict values via dotted paths."""

    def test_set_top_level_scalar(self):
        raw = {}
        _set_nested_override(raw, "tau", 0.7)
        assert raw == {"tau": 0.7}

    def test_set_top_level_string(self):
        raw = {}
        _set_nested_override(raw, "simulation_name", "exp1")
        assert raw == {"simulation_name": "exp1"}

    def test_set_one_level_nested(self):
        raw = {}
        _set_nested_override(raw, "gravity_force.force_g", 5e-7)
        assert raw == {"gravity_force": {"force_g": 5e-7}}

    def test_set_two_levels_nested(self):
        raw = {}
        _set_nested_override(raw, "some.nested.field", 42)
        assert raw == {"some": {"nested": {"field": 42}}}

    def test_override_existing_scalar(self):
        raw = {"tau": 0.6}
        _set_nested_override(raw, "tau", 0.8)
        assert raw == {"tau": 0.8}

    def test_override_in_existing_nested_dict(self):
        raw = {"gravity_force": {"force_g": 5e-7}}
        _set_nested_override(raw, "gravity_force.inclination_angle_deg", 50)
        assert raw == {"gravity_force": {"force_g": 5e-7, "inclination_angle_deg": 50}}

    def test_reject_override_on_non_dict_value(self):
        raw = {"tau": 0.6}
        with pytest.raises(TypeError, match="is not a table"):
            _set_nested_override(raw, "tau.something", 42)

    def test_set_with_normalized_path(self):
        raw = {}
        _set_nested_override(raw, "simulation_type.tau", 0.7)
        assert raw == {"tau": 0.7}

    def test_set_nested_with_normalized_path(self):
        raw = {}
        _set_nested_override(raw, "boundary_conditions.top", "periodic")
        assert raw == {"bc_config": {"top": "periodic"}}

    def test_set_array_value(self):
        raw = {}
        _set_nested_override(raw, "tau", [0.6, 0.7, 0.8])
        assert raw == {"tau": [0.6, 0.7, 0.8]}


# =========================================================================
# _apply_overrides Tests
# =========================================================================


class TestApplyOverrides:
    """Tests for applying multiple overrides in order."""

    def test_apply_no_overrides(self):
        raw = {"tau": 0.6}
        _apply_overrides(raw, ())
        assert raw == {"tau": 0.6}

    def test_apply_single_override(self, capsys):
        raw = {"tau": 0.6}
        _apply_overrides(raw, ("tau=0.8",))
        assert raw == {"tau": 0.8}
        captured = capsys.readouterr()
        assert "Applying CLI overrides" in captured.out
        # Rich adds ANSI formatting, so check for the key parts
        assert "tau" in captured.out
        assert "0.8" in captured.out

    def test_apply_multiple_overrides_in_order(self):
        raw = {"tau": 0.6, "nt": 1000}
        _apply_overrides(raw, ("tau=0.8", "nt=2000"))
        assert raw == {"tau": 0.8, "nt": 2000}

    def test_apply_overrides_last_wins(self):
        raw = {"tau": 0.6}
        _apply_overrides(raw, ("tau=0.7", "tau=0.9"))
        assert raw == {"tau": 0.9}

    def test_apply_overrides_to_nested_dict(self):
        raw = {"gravity_force": {"force_g": 5e-7}}
        _apply_overrides(raw, ("gravity_force.inclination_angle_deg=50",))
        assert raw == {"gravity_force": {"force_g": 5e-7, "inclination_angle_deg": 50}}

    def test_apply_overrides_with_alias_normalization(self):
        raw = {}
        _apply_overrides(raw, ("simulation_type.tau=0.8",))
        assert raw == {"tau": 0.8}

    def test_apply_mixed_scalar_and_array_overrides(self):
        raw = {"tau": 0.6}
        _apply_overrides(raw, ("nt=500", 'fields=["rho_t_plus1", "u"]'))
        assert raw == {"tau": 0.6, "nt": 500, "fields": ["rho_t_plus1", "u"]}

    def test_apply_overrides_reject_invalid_value(self):
        raw = {}
        with pytest.raises(ValueError, match="invalid override value"):
            _apply_overrides(raw, ("tau=not_toml",))

    def test_apply_overrides_reject_invalid_path(self):
        raw = {"some_scalar": 123}
        with pytest.raises(TypeError, match="is not a table"):
            _apply_overrides(raw, ("some_scalar.nested=456",))


# =========================================================================
# CLI Integration Edge Cases
# =========================================================================


class TestCLIEdgeCases:
    """Integration tests for CLI edge cases."""

    def test_override_with_special_characters_in_string(self):
        path, value = _parse_override_argument('name="test_sim-2024"')
        assert path == "name"
        assert value == "test_sim-2024"

    def test_override_with_unicode_string(self):
        path, value = _parse_override_argument('title="Simulation α"')
        assert path == "title"
        assert value == "Simulation α"

    def test_override_with_negative_number(self):
        path, value = _parse_override_argument("offset=-0.5")
        assert path == "offset"
        assert value == -0.5

    def test_override_with_large_number(self):
        path, value = _parse_override_argument("large=1.23e100")
        assert path == "large"
        assert value == 1.23e100

    def test_override_with_empty_array(self):
        path, value = _parse_override_argument("fields=[]")
        assert path == "fields"
        assert value == []

    def test_override_creates_deeply_nested_structure(self):
        raw = {}
        _set_nested_override(raw, "a.b.c.d.e", "deep")
        assert raw == {"a": {"b": {"c": {"d": {"e": "deep"}}}}}

    def test_override_multiple_deep_nested_fields(self):
        raw = {}
        _apply_overrides(
            raw,
            (
                "gravity_force.force_g=1e-6",
                "gravity_force.inclination_angle_deg=45",
                "wetting_config.contact_angle=90",
            ),
        )
        assert raw == {
            "gravity_force": {"force_g": 1e-6, "inclination_angle_deg": 45},
            "wetting_config": {"contact_angle": 90},
        }
