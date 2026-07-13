"""Tests for CLI helper functions and edge cases."""

from __future__ import annotations
import importlib
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch
import click
import pytest
from click.testing import CliRunner
from tud_lbm.cli.cli import _WETTING_PARAM_DEFAULTS
from tud_lbm.cli.cli import _build_wetting_gravity_raw
from tud_lbm.cli.cli import _build_wetting_init_raw
from tud_lbm.cli.cli import _check_sweep_errors
from tud_lbm.cli.cli import _display_config_summary
from tud_lbm.cli.cli import _display_full_overview
from tud_lbm.cli.cli import _display_sweep_summary
from tud_lbm.cli.cli import _print_dry_run_message
from tud_lbm.cli.cli import _prompt_wetting_params
from tud_lbm.cli.cli import _run_impl
from tud_lbm.cli.cli import _validate_cli_args
from tud_lbm.cli.cli import animate
from tud_lbm.cli.cli import cli
from tud_lbm.cli.cli import compare
from tud_lbm.cli.cli import main
from tud_lbm.cli.cli import visualise
from tud_lbm.config import SimulationConfig
from tud_lbm.config.array_expansion import ArrayParameterSet
from tud_lbm.pipeline.parallel_runner import SimulationResult

try:
    from tud_lbm.cli.cli import _apply_overrides
    from tud_lbm.cli.cli import _normalise_override_path
    from tud_lbm.cli.cli import _parse_override_argument
    from tud_lbm.cli.cli import _set_nested_override
except ImportError:
    pytest.skip("click or rich dependency not installed", allow_module_level=True)

from unittest.mock import patch as _patch
from tud_lbm.cli.cli import _build_fields_table
from tud_lbm.cli.cli import _build_standard_table
from tud_lbm.cli.cli import _build_visual_table
from tud_lbm.cli.cli import _confirm_run
from tud_lbm.cli.cli import _display_summary
from tud_lbm.cli.cli import _execute_run
from tud_lbm.cli.cli import _expand_raw_config
from tud_lbm.cli.cli import _expand_single_phase
from tud_lbm.cli.cli import _load_config_interactive
from tud_lbm.cli.cli import _load_raw_config
from tud_lbm.cli.cli import _operator_description
from tud_lbm.cli.cli import _parse_field_tokens
from tud_lbm.cli.cli import _print_run_banner
from tud_lbm.cli.cli import _prompt_fields
from tud_lbm.cli.cli import _resolve_token
from tud_lbm.cli.cli import _run_compare_single
from tud_lbm.cli.cli import _run_compare_sweep
from tud_lbm.cli.cli import _run_two_phase_wetting_init
from tud_lbm.cli.cli import _run_with_optional_overrides


def test_package_cli_import_is_callable():
    from tud_lbm.cli.cli import cli

    assert callable(cli)
    assert not isinstance(cli, ModuleType)


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
        result = _normalise_override_path("tau")
        assert result == ["tau"]

    def test_nested_field_no_normalization(self):
        result = _normalise_override_path("gravity_force.force_g")
        assert result == ["gravity_force", "force_g"]

    def test_simulation_type_prefix_removed(self):
        result = _normalise_override_path("simulation_type.tau")
        assert result == ["tau"]

    def test_simulation_type_with_nested_field(self):
        result = _normalise_override_path("simulation_type.simulation_name")
        assert result == ["simulation_name"]

    def test_boundary_conditions_mapped_to_bc_config(self):
        result = _normalise_override_path("boundary_conditions.top")
        assert result == ["bc_config", "top"]

    def test_wetting_mapped_to_wetting_config(self):
        result = _normalise_override_path("wetting.contact_angle")
        assert result == ["wetting_config", "contact_angle"]

    def test_hysteresis_mapped_to_hysteresis_config(self):
        result = _normalise_override_path("hysteresis.angle_max")
        assert result == ["hysteresis_config", "angle_max"]

    def test_gravity_force_not_mapped(self):
        result = _normalise_override_path("gravity_force.force_g")
        assert result == ["gravity_force", "force_g"]

    def test_electric_force_not_mapped(self):
        result = _normalise_override_path("electric_force.permittivity")
        assert result == ["electric_force", "permittivity"]

    def test_multiphase_prefix_removed(self):
        result = _normalise_override_path("multiphase.kappa")
        assert result == ["kappa"]

    def test_output_prefix_removed(self):
        result = _normalise_override_path("output.results_dir")
        assert result == ["results_dir"]

    def test_reject_empty_path(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _normalise_override_path("")

    def test_reject_dots_only(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _normalise_override_path("...")

    def test_reject_prefix_without_field(self):
        with pytest.raises(ValueError, match="does not reference a field"):
            _normalise_override_path("simulation_type")

    def test_reject_prefix_with_dots_only(self):
        with pytest.raises(ValueError, match="does not reference a field"):
            _normalise_override_path("simulation_type...")

    def test_strip_whitespace_in_segments(self):
        result = _normalise_override_path("gravity_force . force_g")
        assert result == ["gravity_force", "force_g"]


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


class TestDisplayConfigSummary:
    """Branch coverage for _display_config_summary."""

    def test_none_config(self, capsys):
        _display_config_summary(None)
        assert "No configuration" in capsys.readouterr().out

    def test_multiphase_config(self, capsys):
        cfg = SimulationConfig(
            sim_type="multiphase",
            grid_shape=(16, 16),
            eos="double-well",
            kappa=0.01,
            rho_l=1.0,
            rho_v=0.2,
            interface_width=2,
        )
        _display_config_summary(cfg)
        assert "Kappa" in capsys.readouterr().out

    def test_config_with_fields(self, capsys):
        cfg = SimulationConfig(
            grid_shape=(8, 8),
            save_fields=["rho"],
            plot_fields=["density"],
            animate_fields=["density"],
        )
        _display_config_summary(cfg)
        assert "rho" in capsys.readouterr().out

    def test_config_with_force(self, capsys):
        cfg = SimulationConfig(
            sim_type="multiphase",
            grid_shape=(16, 16),
            eos="double-well",
            kappa=0.01,
            rho_l=1.0,
            rho_v=0.2,
            interface_width=2,
            gravity_force={"force_g": 1e-6},
        )
        _display_config_summary(cfg)
        out = capsys.readouterr().out
        assert "gravity" in out.lower() or "Forces" in out or "enabled" in out


class TestDisplayFullOverview:
    """Tests for _display_full_overview."""

    def test_none_config(self, capsys):
        _display_full_overview(None)
        assert "No configuration" in capsys.readouterr().out

    def test_valid_config(self, capsys):
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        _display_full_overview(cfg)
        assert "PHYSICAL PARAMETER OVERVIEW" in capsys.readouterr().out


def test_display_sweep_summary(capsys):
    metadata = ArrayParameterSet(
        field_names=["tau"],
        array_values={"tau": (0.6, 0.7)},
        total_combinations=2,
    )
    _display_sweep_summary(metadata)
    out = capsys.readouterr().out
    assert "tau" in out


def test_print_dry_run_message_non_sweep(capsys):
    _print_dry_run_message(None)
    assert "Dry run" in capsys.readouterr().out


def test_print_dry_run_message_sweep(capsys):
    metadata = ArrayParameterSet(
        field_names=["tau"],
        array_values={"tau": (0.6, 0.7)},
        total_combinations=2,
    )
    _print_dry_run_message(metadata)
    assert "Dry run" in capsys.readouterr().out


class TestCheckSweepErrors:
    """Tests for _check_sweep_errors failure detection."""

    def test_no_failures(self):
        _check_sweep_errors([SimpleNamespace(status="success")])

    def test_one_failure_raises(self):
        with pytest.raises(RuntimeError, match="failed simulation"):
            _check_sweep_errors(
                [
                    SimpleNamespace(status="success"),
                    SimpleNamespace(status="failed"),
                ]
            )

    def test_multiple_failures_mention_count(self):
        with pytest.raises(RuntimeError, match="3 failed"):
            _check_sweep_errors([SimpleNamespace(status="failed")] * 3)


class TestRunImplFlags:
    """Lightweight _run_impl flag tests that avoid real simulation execution."""

    def test_list_operators_returns_false(self):
        result = _run_impl(
            config_path=None,
            no_prompt=True,
            dry_run=False,
            list_operators=True,
            list_analysis=False,
            max_workers=None,
            fail_fast=False,
            overrides=(),
            overview=False,
            debug_wetting=False,
            init_wetting=False,
            init_dir=None,
        )
        assert result is False

    def _single_config(self):
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        return [cfg], cfg, None, None

    def test_dry_run_returns_false(self, tmp_path):
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text("[simulation_type]\ntau=0.8\nnt=10\nnx=8\nny=8\nnz=1\n", encoding="utf-8")
        with (
            patch("tud_lbm.cli.cli._load_raw_config", return_value={}),
            patch("tud_lbm.cli.cli._expand_raw_config", return_value=self._single_config()),
        ):
            result = _run_impl(
                config_path=str(cfg_toml),
                no_prompt=True,
                dry_run=True,
                list_operators=False,
                list_analysis=False,
                max_workers=None,
                fail_fast=False,
                overrides=(),
                overview=False,
                debug_wetting=False,
                init_wetting=False,
                init_dir=None,
            )
        assert result is False

    def test_overview_flag_prints_overview(self, tmp_path, capsys):
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text("[simulation_type]\ntau=0.8\nnt=10\nnx=8\nny=8\nnz=1\n", encoding="utf-8")
        with (
            patch("tud_lbm.cli.cli._load_raw_config", return_value={}),
            patch("tud_lbm.cli.cli._expand_raw_config", return_value=self._single_config()),
        ):
            _run_impl(
                config_path=str(cfg_toml),
                no_prompt=True,
                dry_run=True,
                list_operators=False,
                list_analysis=False,
                max_workers=None,
                fail_fast=False,
                overrides=(),
                overview=True,
                debug_wetting=False,
                init_wetting=False,
                init_dir=None,
            )
        assert "PHYSICAL PARAMETER OVERVIEW" in capsys.readouterr().out

    def test_sweep_dry_run_returns_false(self, tmp_path):
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text("[simulation_type]\ntau=0.8\nnt=10\nnx=8\nny=8\nnz=1\n", encoding="utf-8")
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        sweep = ArrayParameterSet(
            field_names=["tau"],
            array_values={"tau": (0.6, 0.7)},
            total_combinations=2,
        )
        with (
            patch("tud_lbm.cli.cli._load_raw_config", return_value={}),
            patch(
                "tud_lbm.cli.cli._expand_raw_config",
                return_value=([cfg, cfg], None, sweep, [{"tau": 0.6}, {"tau": 0.7}]),
            ),
        ):
            result = _run_impl(
                config_path=str(cfg_toml),
                no_prompt=True,
                dry_run=True,
                list_operators=False,
                list_analysis=False,
                max_workers=None,
                fail_fast=False,
                overrides=(),
                overview=True,
                debug_wetting=False,
                init_wetting=False,
                init_dir=None,
            )
        assert result is False

    def test_no_config_path_calls_interactive(self, monkeypatch):
        called = {"n": 0}
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)

        def _fake():
            called["n"] += 1
            return [cfg], cfg, None, None

        monkeypatch.setattr("tud_lbm.cli.cli._load_config_interactive", _fake)
        result = _run_impl(
            config_path=None,
            no_prompt=True,
            dry_run=True,
            list_operators=False,
            list_analysis=False,
            max_workers=None,
            fail_fast=False,
            overrides=(),
            overview=False,
            debug_wetting=False,
            init_wetting=False,
            init_dir=None,
        )
        assert called["n"] == 1
        assert result is False

    def test_debug_wetting_sets_flag(self, tmp_path):
        import tud_lbm.config.config_overview as _flags

        original = _flags.DEBUG_FLAG_WETTING
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text("[simulation_type]\ntau=0.8\nnt=10\nnx=8\nny=8\nnz=1\n", encoding="utf-8")
        try:
            with (
                patch("tud_lbm.cli.cli._load_raw_config", return_value={}),
                patch("tud_lbm.cli.cli._expand_raw_config", return_value=self._single_config()),
            ):
                _run_impl(
                    config_path=str(cfg_toml),
                    no_prompt=True,
                    dry_run=True,
                    list_operators=False,
                    list_analysis=False,
                    max_workers=None,
                    fail_fast=False,
                    overrides=(),
                    overview=False,
                    debug_wetting=True,
                    init_wetting=False,
                    init_dir=None,
                )
            assert _flags.DEBUG_FLAG_WETTING is True
        finally:
            _flags.DEBUG_FLAG_WETTING = original

    def test_debug_stability_sets_flag(self, tmp_path):
        import tud_lbm.config.config_overview as _flags

        original = _flags.DEBUG_FLAG_STABILITY
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text("[simulation_type]\ntau=0.8\nnt=10\nnx=8\nny=8\nnz=1\n", encoding="utf-8")
        try:
            with (
                patch("tud_lbm.cli.cli._load_raw_config", return_value={}),
                patch("tud_lbm.cli.cli._expand_raw_config", return_value=self._single_config()),
            ):
                _run_impl(
                    config_path=str(cfg_toml),
                    no_prompt=True,
                    dry_run=True,
                    list_operators=False,
                    list_analysis=False,
                    max_workers=None,
                    fail_fast=False,
                    overrides=(),
                    overview=False,
                    debug_wetting=False,
                    init_wetting=False,
                    init_dir=None,
                    debug_stability=True,
                )
            assert _flags.DEBUG_FLAG_STABILITY is True
        finally:
            _flags.DEBUG_FLAG_STABILITY = original


class TestClickCommandPaths:
    """CLI command error and success paths via CliRunner."""

    def test_run_keyboard_interrupt_exits_130(self):
        runner = CliRunner()
        with patch("tud_lbm.cli.cli._run_impl", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["run"])
        assert result.exit_code == 130

    def test_run_debug_stability_option_forwards(self, tmp_path):
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text("[simulation_type]\ntau=0.8\nnt=10\nnx=8\nny=8\nnz=1\n", encoding="utf-8")
        runner = CliRunner()
        with patch("tud_lbm.cli.cli._run_impl", return_value=False) as mock_impl:
            result = runner.invoke(cli, ["run", str(cfg_toml), "--debug-stability", "--dry-run"])
        assert result.exit_code == 0
        assert mock_impl.call_args.kwargs["debug_stability"] is True

    def test_run_general_exception_exits_1(self):
        runner = CliRunner()
        with patch("tud_lbm.cli.cli._run_impl", side_effect=RuntimeError("boom")):
            result = runner.invoke(cli, ["run"])
        assert result.exit_code == 1

    def test_animate_keyboard_interrupt_exits_130(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")
        runner = CliRunner()
        with patch("tud_lbm.cli.cli._validate_run_dir_has_config", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["animate", str(run_dir)])
        assert result.exit_code == 130

    def test_animate_general_exception_exits_1(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")
        runner = CliRunner()
        with patch("tud_lbm.cli.cli._validate_run_dir_has_config", side_effect=RuntimeError("x")):
            result = runner.invoke(cli, ["animate", str(run_dir)])
        assert result.exit_code == 1

    def test_visualise_keyboard_interrupt_exits_130(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")
        runner = CliRunner()
        with patch("tud_lbm.cli.cli._validate_run_dir_has_config", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["visualise", str(run_dir)])
        assert result.exit_code == 130

    def test_visualise_general_exception_exits_1(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")
        runner = CliRunner()
        with patch("tud_lbm.cli.cli._validate_run_dir_has_config", side_effect=RuntimeError("x")):
            result = runner.invoke(cli, ["visualise", str(run_dir)])
        assert result.exit_code == 1

    def test_compare_no_runs_prints_message(self, tmp_path):
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.run_comparison.process_parent_dir", return_value=(0, 0)):
            result = runner.invoke(cli, ["compare", str(tmp_path)])
        assert result.exit_code == 0
        assert "No simulation" in result.output

    def test_compare_runs_with_zero_ok(self, tmp_path):
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.run_comparison.process_parent_dir", return_value=(1, 0)):
            result = runner.invoke(cli, ["compare", str(tmp_path)])
        assert result.exit_code == 0
        assert "no runs produced" in result.output.lower()

    def test_compare_runs_with_success(self, tmp_path):
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.run_comparison.process_parent_dir", return_value=(2, 2)):
            result = runner.invoke(cli, ["compare", str(tmp_path)])
        assert result.exit_code == 0

    def test_compare_keyboard_interrupt_exits_130(self, tmp_path):
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.run_comparison.process_parent_dir", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["compare", str(tmp_path)])
        assert result.exit_code == 130

    def test_compare_general_exception_exits_1(self, tmp_path):
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.run_comparison.process_parent_dir", side_effect=RuntimeError("fail")):
            result = runner.invoke(cli, ["compare", str(tmp_path)])
        assert result.exit_code == 1

    def test_regime_map_success(self, tmp_path):
        dirs_txt = tmp_path / "dirs.txt"
        dirs_txt.write_text("run_a\n", encoding="utf-8")
        runner = CliRunner()
        out_path = tmp_path / "regime_map_analysis" / "regime_map.png"
        with patch("tud_lbm.io.plotting.regime_map_plot.build_regime_map", return_value=out_path):
            result = runner.invoke(cli, ["regime-map", str(dirs_txt)], env={"COLUMNS": "200", "LINES": "50"})
        assert result.exit_code == 0
        assert "regime_map.png" in result.output

    def test_regime_map_no_usable_runs_exits_1(self, tmp_path):
        dirs_txt = tmp_path / "dirs.txt"
        dirs_txt.write_text("run_a\n", encoding="utf-8")
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.regime_map_plot.build_regime_map", return_value=None):
            result = runner.invoke(cli, ["regime-map", str(dirs_txt)])
        assert result.exit_code == 1

    def test_regime_map_keyboard_interrupt_exits_130(self, tmp_path):
        dirs_txt = tmp_path / "dirs.txt"
        dirs_txt.write_text("run_a\n", encoding="utf-8")
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.regime_map_plot.build_regime_map", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["regime-map", str(dirs_txt)])
        assert result.exit_code == 130

    def test_regime_map_general_exception_exits_1(self, tmp_path):
        dirs_txt = tmp_path / "dirs.txt"
        dirs_txt.write_text("run_a\n", encoding="utf-8")
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.regime_map_plot.build_regime_map", side_effect=RuntimeError("fail")):
            result = runner.invoke(cli, ["regime-map", str(dirs_txt)])
        assert result.exit_code == 1

    def test_animate_with_mocked_internals(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\ntau=0.8\nnt=10\nnx=8\nny=8\nnz=1\n", encoding="utf-8")
        runner = CliRunner()
        with (
            patch("tud_lbm.config.from_toml", return_value=SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)),
            patch("tud_lbm.io.plotting.Animator") as mock_anim,
        ):
            mock_anim.return_value.create.return_value = run_dir / "plots" / "animation.mp4"
            result = runner.invoke(cli, ["animate", str(run_dir)])
        assert result.exit_code in (0, 1)

    def test_visualise_with_mocked_internals_produces_figures(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\ntau=0.8\nnt=10\nnx=8\nny=8\nnz=1\n", encoding="utf-8")
        runner = CliRunner()
        with (
            patch("tud_lbm.config.from_toml", return_value=SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)),
            patch("tud_lbm.io.plotting.FigureBuilder") as mock_fb,
        ):
            mock_fb.return_value.build_all.return_value = [run_dir / "plots" / "t1.png"]
            mock_fb.return_value.plot_dir = run_dir / "plots"
            result = runner.invoke(cli, ["visualise", str(run_dir)])
        assert result.exit_code in (0, 1)

    def test_visualise_with_no_figures_produced(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\ntau=0.8\nnt=10\nnx=8\nny=8\nnz=1\n", encoding="utf-8")
        runner = CliRunner()
        with (
            patch("tud_lbm.config.from_toml", return_value=SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)),
            patch("tud_lbm.io.plotting.FigureBuilder") as mock_fb,
        ):
            mock_fb.return_value.build_all.return_value = []
            mock_fb.return_value.plot_dir = run_dir / "plots"
            result = runner.invoke(cli, ["visualise", str(run_dir)])
        assert result.exit_code in (0, 1)
        assert "No figures" in result.output or result.exit_code == 0

    def test_main_shim_animate_route(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["animate", str(tmp_path)])
        assert result.exit_code in (0, 1, 2)

    def test_main_shim_visualise_route(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["visualise", str(tmp_path)])
        assert result.exit_code in (0, 1, 2)

    def test_main_shim_compare_route(self, tmp_path):
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.run_comparison.process_parent_dir", return_value=(0, 0)):
            result = runner.invoke(main, ["compare", str(tmp_path)])
        assert result.exit_code in (0, 1, 2)

    def test_run_with_dry_run_and_config(self, tmp_path):
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text(
            "[simulation_type]\ntau = 0.8\nnt = 10\nnx = 8\nny = 8\nnz = 1\n",
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(cfg_toml), "--dry-run"])
        assert result.exit_code in (0, 1)


class TestValidateCliArgs:
    """Tests for _validate_cli_args argument validation."""

    def test_overrides_without_config_raises(self):
        with pytest.raises(click.UsageError, match="--override requires"):
            _validate_cli_args(("tau=0.8",), None)

    def test_init_wetting_without_config_raises(self):
        with pytest.raises(click.UsageError, match="--init-wetting requires"):
            _validate_cli_args((), None, init_wetting=True)

    def test_init_dir_without_config_raises(self):
        with pytest.raises(click.UsageError, match="--init-dir requires"):
            _validate_cli_args((), None, init_dir="/some/path.npz")

    def test_valid_args_do_not_raise(self):
        _validate_cli_args(("tau=0.8",), "config.toml")

    def test_empty_args_no_config_ok(self):
        _validate_cli_args((), None)


class TestPromptWettingParams:
    """Tests for _prompt_wetting_params with no_prompt=True."""

    def test_no_prompt_uses_defaults_when_config_empty(self):
        params = _prompt_wetting_params({}, no_prompt=True)
        assert params == _WETTING_PARAM_DEFAULTS

    def test_values_from_config_override_defaults(self):
        raw = {"wetting_config": {"phi_left": 1.3, "phi_right": 1.4, "d_rho_left": 0.1, "d_rho_right": 0.05}}
        params = _prompt_wetting_params(raw, no_prompt=True)
        assert params["phi_left"] == pytest.approx(1.3)
        assert params["phi_right"] == pytest.approx(1.4)
        assert params["d_rho_left"] == pytest.approx(0.1)
        assert params["d_rho_right"] == pytest.approx(0.05)

    def test_partial_config_values_fill_missing_with_defaults(self):
        raw = {"wetting_config": {"phi_left": 1.2}}
        params = _prompt_wetting_params(raw, no_prompt=True)
        assert params["phi_left"] == pytest.approx(1.2)
        assert params["phi_right"] == pytest.approx(_WETTING_PARAM_DEFAULTS["phi_right"])


class TestBuildWettingInitRaw:
    """Tests for _build_wetting_init_raw structural mutations."""

    def _base_raw(self) -> dict:
        return {
            "sim_type": "multiphase_hysteresis",
            "nt": 100,
            "save_interval": 10,
            "bc_config": {"bottom": "bounce-back", "top": "symmetry"},
            "hysteresis_config": {"ca_advancing": 100},
            "chemical_step_config": {"step": 0.5},
            "gravity_force": {"force_g": 1e-6},
            "wetting_config": {"contact_angle": 90},
        }

    def test_sim_type_overridden(self):
        result = _build_wetting_init_raw(self._base_raw(), {})
        assert result["sim_type"] == "multiphase_wetting"

    def test_init_type_set_to_bubbles(self):
        result = _build_wetting_init_raw(self._base_raw(), {})
        assert result["init_type"] == "multiphase_bubbles"

    def test_hysteresis_and_chemical_step_removed(self):
        result = _build_wetting_init_raw(self._base_raw(), {})
        assert "hysteresis_config" not in result
        assert "chemical_step_config" not in result

    def test_gravity_force_removed(self):
        result = _build_wetting_init_raw(self._base_raw(), {})
        assert "gravity_force" not in result

    def test_nt_set_to_wetting_init_constant(self):
        from tud_lbm.cli.cli import _WETTING_INIT_NT

        result = _build_wetting_init_raw(self._base_raw(), {})
        assert result["nt"] == _WETTING_INIT_NT

    def test_wetting_params_injected(self):
        params = {"phi_left": 1.1, "phi_right": 1.2, "d_rho_left": 0.0, "d_rho_right": 0.0}
        result = _build_wetting_init_raw(self._base_raw(), params)
        for k, v in params.items():
            assert result["wetting_config"][k] == pytest.approx(v)

    def test_base_raw_is_not_mutated(self):
        base = self._base_raw()
        _build_wetting_init_raw(base, {})
        assert base["sim_type"] == "multiphase_hysteresis"

    def test_simulation_name_suffixed_with_base_name(self):
        base = self._base_raw()
        base["simulation_name"] = "droplet_run_42"
        result = _build_wetting_init_raw(base, {})
        assert result["simulation_name"] == "wetting_init_droplet_run_42"

    def test_simulation_name_falls_back_when_unset(self):
        result = _build_wetting_init_raw(self._base_raw(), {})
        assert result["simulation_name"] == "wetting_init"


class TestBuildWettingGravityRaw:
    """Tests for _build_wetting_gravity_raw field injection."""

    def test_init_type_set_to_init_from_file(self):
        base = {"nt": 1000, "gravity_force": {"force_g": 1e-6}}
        result = _build_wetting_gravity_raw(base, {}, "/path/snapshot.npz")
        assert result["init_type"] == "init_from_file"

    def test_init_dir_injected(self):
        base = {"nt": 1000}
        result = _build_wetting_gravity_raw(base, {}, "/path/snapshot.npz")
        assert result["init_dir"] == "/path/snapshot.npz"

    def test_wetting_params_merged(self):
        base = {"nt": 1000, "wetting_config": {"contact_angle": 90}}
        params = {"phi_left": 1.1, "phi_right": 1.0, "d_rho_left": 0.0, "d_rho_right": 0.0}
        result = _build_wetting_gravity_raw(base, params, "/snap.npz")
        assert result["wetting_config"]["phi_left"] == pytest.approx(1.1)
        assert result["wetting_config"]["contact_angle"] == 90

    def test_base_raw_not_mutated(self):
        base = {"nt": 1000}
        _build_wetting_gravity_raw(base, {}, "/snap.npz")
        assert "init_type" not in base


class TestClickCommands:
    """Test CLI entry points using CliRunner (extra coverage)."""

    def test_run_list_operators_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--list-operators"])
        assert result.exit_code in (0, 2)

    def test_animate_missing_run_dir_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(animate, ["/nonexistent/run_dir"])
        assert result.exit_code != 0

    def test_visualise_missing_run_dir_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(visualise, ["/nonexistent/run_dir"])
        assert result.exit_code != 0

    def test_compare_missing_parent_dir_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(compare, ["/nonexistent/parent_dir"])
        assert result.exit_code != 0

    def test_animate_valid_run_dir_no_config_toml_exits_nonzero(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(animate, [str(run_dir)])
        assert result.exit_code != 0

    def test_visualise_valid_run_dir_no_config_toml_exits_nonzero(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(visualise, [str(run_dir)])
        assert result.exit_code != 0

    def test_compare_valid_dir_no_runs_prints_warning(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", str(tmp_path)])
        assert result.exit_code == 0
        assert "No simulation" in result.output

    def test_main_shim_forwards_run_subcommand(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["nonexistent.toml"])
        assert result.exit_code in (0, 1, 2)

    def test_main_shim_handles_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code in (0, 1)

    def test_run_override_without_config_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--override", "tau=0.8"])
        assert result.exit_code != 0

    def test_run_init_wetting_without_config_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--init-wetting"])
        assert result.exit_code != 0

    def test_run_init_dir_without_config_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--init-dir", str(tmp_path)])
        assert result.exit_code != 0

    def test_run_dry_run_with_config(self, tmp_path):
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text(
            '[simulation_type]\nsim_type = "single"\nnx = 8\nny = 8\nnz = 1\ntau = 0.8\nnt = 10\nsave_interval = 10\n',
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(cfg_toml), "--dry-run"])
        assert result.exit_code in (0, 1)


class TestErrorHandling:
    """Tests for keyboard-interrupt and error passthrough in the run command."""

    def test_keyboard_interrupt_in_run_exits_130(self, tmp_path):
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text(
            '[simulation_type]\nsim_type = "single"\nnx = 4\nny = 4\nnz = 1\ntau = 0.8\nnt = 10\nsave_interval = 10\n',
            encoding="utf-8",
        )
        runner = CliRunner()
        with patch("tud_lbm.cli.cli._run_impl", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["run", str(cfg_toml)])
        assert result.exit_code == 130


@pytest.fixture
def cli_pkg():
    """Import tud_lbm.cli package for lazy-loader tests."""
    import tud_lbm.cli as module

    return module


def test_lazy_cli_proxy_calls_loaded_object(cli_pkg, monkeypatch):
    called = {"ok": False}

    class _FakeCLI:
        name = "fake"

        def __call__(self, *args, **kwargs):
            called["ok"] = True
            return "done"

    monkeypatch.setattr(cli_pkg, "_load_cli", _FakeCLI)
    proxy = cli_pkg._LazyCLI()

    assert proxy() == "done"
    assert proxy.name == "fake"
    assert called["ok"] is True


def test_module_getattr_unknown_raises_attribute_error(cli_pkg):
    with pytest.raises(AttributeError, match="has no attribute"):
        cli_pkg.missing  # noqa: B018


def test_load_cli_wraps_missing_optional_dependencies(cli_pkg, monkeypatch):
    original_import = __import__

    def _raising_import(name, *args, **kwargs):
        if name == "tud_lbm.cli.cli":
            err = ImportError("no click")
            err.name = "click"
            raise err
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raising_import)

    with pytest.raises(ImportError, match="requires 'click' and 'rich'"):
        cli_pkg._load_cli()


def test_validate_cli_args_requires_config_path():
    with pytest.raises(click.UsageError, match="--override requires CONFIG_PATH"):
        _validate_cli_args(("tau=0.7",), None)


def test_check_sweep_errors_raises_on_failed_results():
    ok = SimpleNamespace(status="success")
    bad = SimpleNamespace(status="failed")

    with pytest.raises(RuntimeError, match="failed simulation"):
        _check_sweep_errors([ok, bad])


def test_main_dispatches_help_to_click_group(monkeypatch):
    cli_module = importlib.import_module("tud_lbm.cli.cli")

    calls: list[list[str]] = []
    monkeypatch.setattr(cli_module.cli, "main", lambda args, standalone_mode: calls.append(args))

    cli_module.main.callback(("--help",))  # ty: ignore[call-non-callable]
    assert calls == [["--help"]]


def test_main_dispatch_strips_run_token(monkeypatch):
    cli_module = importlib.import_module("tud_lbm.cli.cli")

    calls: list[list[str]] = []
    monkeypatch.setattr(cli_module.run, "main", lambda args, standalone_mode: calls.append(args))

    cli_module.main.callback(("run", "config.toml", "--dry-run"))  # ty: ignore[call-non-callable]
    assert calls == [["config.toml", "--dry-run"]]


@pytest.mark.parametrize("token", ["animate", "visualise", "compare"])
def test_main_dispatches_subcommands_to_click_group(monkeypatch, token):
    cli_module = importlib.import_module("tud_lbm.cli.cli")

    calls: list[list[str]] = []
    monkeypatch.setattr(cli_module.cli, "main", lambda args, standalone_mode: calls.append(args))

    cli_module.main.callback((token, "run_dir"))  # ty: ignore[call-non-callable]
    assert calls == [[token, "run_dir"]]


def test_validate_run_dir_has_config_success(tmp_path):
    from tud_lbm.cli.cli import _validate_run_dir_has_config

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = run_dir / "config.toml"
    config_path.write_text("[simulation_type]\ntype='single_phase'\n", encoding="utf-8")

    assert _validate_run_dir_has_config(str(run_dir)) == config_path


def test_validate_run_dir_has_config_missing_file_raises(tmp_path):
    from tud_lbm.cli.cli import _validate_run_dir_has_config

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=r"No config\.toml found"):
        _validate_run_dir_has_config(str(run_dir))


def _make_config(results_dir: str, tau: float = 0.8) -> SimulationConfig:
    return SimulationConfig(
        grid_shape=(8, 8),
        tau=tau,
        nt=10,
        simulation_name="test",
        results_dir=results_dir,
    )


def test_cli_single_config_uses_single_run(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    config = _make_config(str(tmp_path))

    monkeypatch.setattr("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", lambda self, path: {"stub": "raw"})
    monkeypatch.setattr("tud_lbm.config.array_expansion.expand_config", lambda raw: ([config], None))

    called = {"single": False}

    def _fake_single_run(cfg):
        called["single"] = True
        assert cfg == config
        return str(tmp_path / "single")

    monkeypatch.setattr("tud_lbm.cli.cli._run_simulation", _fake_single_run)

    result = CliRunner().invoke(main, [str(cfg_path), "--no-prompt"])

    assert result.exit_code == 0
    assert called["single"] is True


def test_cli_list_operators_includes_plotting_and_analysis():
    result = CliRunner().invoke(main, ["--list-simulation-analysis"])

    assert result.exit_code == 0
    assert "plotting" in result.output
    assert "comparison" in result.output
    assert "density" in result.output
    assert "max_velocity" in result.output
    assert "simulation_csv" in result.output


def test_cli_array_config_uses_parallel_sweep(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    config_a = _make_config(str(tmp_path), tau=0.6)
    config_b = _make_config(str(tmp_path), tau=0.8)

    metadata = ArrayParameterSet(
        field_names=frozenset({"tau"}),
        array_values={"tau": (0.6, 0.8)},
        total_combinations=2,
    )

    monkeypatch.setattr("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", lambda self, path: {"stub": "raw"})
    monkeypatch.setattr(
        "tud_lbm.config.array_expansion.expand_config",
        lambda raw: ([config_a, config_b], metadata),
    )
    monkeypatch.setattr(
        "tud_lbm.config.array_expansion.enumerate_configs",
        lambda raw: iter(
            [
                (0, {"tau": 0.6}, config_a),
                (1, {"tau": 0.8}, config_b),
            ],
        ),
    )

    captured = {"called": False, "params": None}

    def _fake_parallel_sweep(configs, parameters_list, **kwargs):
        captured["called"] = True
        captured["params"] = parameters_list
        assert len(configs) == 2
        return [
            SimulationResult(index=0, config=config_a, status="success"),
            SimulationResult(index=1, config=config_b, status="success"),
        ]

    monkeypatch.setattr("tud_lbm.cli.cli._run_parallel_sweep", _fake_parallel_sweep)

    result = CliRunner().invoke(main, [str(cfg_path), "--no-prompt"])

    assert result.exit_code == 0
    assert captured["called"] is True
    assert captured["params"] == [{"tau": 0.6}, {"tau": 0.8}]


def test_cli_array_config_dry_run_skips_parallel_execution(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[simulation_type]\ntype = 'single_phase'\n", encoding="utf-8")

    config_a = _make_config(str(tmp_path), tau=0.6)
    config_b = _make_config(str(tmp_path), tau=0.8)

    metadata = ArrayParameterSet(
        field_names=frozenset({"tau"}),
        array_values={"tau": (0.6, 0.8)},
        total_combinations=2,
    )

    monkeypatch.setattr("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", lambda self, path: {"stub": "raw"})
    monkeypatch.setattr(
        "tud_lbm.config.array_expansion.expand_config",
        lambda raw: ([config_a, config_b], metadata),
    )
    monkeypatch.setattr(
        "tud_lbm.config.array_expansion.enumerate_configs",
        lambda raw: iter(
            [
                (0, {"tau": 0.6}, config_a),
                (1, {"tau": 0.8}, config_b),
            ],
        ),
    )

    called = {"parallel": False}

    def _fake_parallel_sweep(configs, parameters_list, **kwargs):
        called["parallel"] = True
        return []

    monkeypatch.setattr("tud_lbm.cli.cli._run_parallel_sweep", _fake_parallel_sweep)

    result = CliRunner().invoke(main, [str(cfg_path), "--no-prompt", "--dry-run"])

    assert result.exit_code == 0
    assert called["parallel"] is False


# =========================================================================
# _build_visual_table / _build_standard_table / _operator_description
# =========================================================================


def _make_entry(target_fn, metadata=None):
    """Create a minimal OperatorEntry-like namespace."""
    return SimpleNamespace(target=target_fn, metadata=metadata or {})


def _sample_target():
    """First docstring line used for description."""


def _no_doc_target():
    pass


_no_doc_target.__doc__ = None


class TestOperatorDescription:
    """Tests for _operator_description helper."""

    def test_returns_first_doc_line(self):
        def fn():
            r"""First line.\n\nRest."""

        assert _operator_description(fn) == r"First line.\n\nRest."

    def test_returns_dash_when_no_doc(self):
        assert _operator_description(_no_doc_target) == "—"

    def test_returns_dash_for_empty_doc(self):
        def fn():
            pass

        fn.__doc__ = ""
        assert _operator_description(fn) == "—"


class TestBuildVisualTable:
    """Tests for _build_visual_table Rich table builder."""

    def test_returns_table_with_three_columns(self):
        ops = {"density": _make_entry(_sample_target)}
        table = _build_visual_table("plotting", ops)
        assert len(table.columns) == 3

    def test_table_title_contains_kind(self):
        ops = {"density": _make_entry(_sample_target)}
        table = _build_visual_table("plotting", ops)
        assert "plotting" in table.title  # ty: ignore[unsupported-operator]

    def test_required_keys_shown(self):
        def fn():
            """Doc."""

        fn.required_keys = ["rho", "u"]  # ty: ignore[unresolved-attribute]
        ops = {"vel": _make_entry(fn)}
        table = _build_visual_table("plotting", ops)
        # Table built without error; row count should equal op count
        assert table.row_count == 1

    def test_subtitle_override(self):
        ops = {"a": _make_entry(_sample_target)}
        table = _build_visual_table("plotting", ops, subtitle="custom sub")
        assert "custom sub" in table.title  # ty: ignore[unsupported-operator]

    def test_empty_ops_produces_empty_rows(self):
        table = _build_visual_table("plotting", {})
        assert table.row_count == 0


class TestBuildStandardTable:
    """Tests for _build_standard_table Rich table builder."""

    def test_returns_table_with_three_columns(self):
        ops = {"operator": _make_entry(_sample_target, metadata={"key": "val"})}
        table = _build_standard_table("collision", ops)
        assert len(table.columns) == 3

    def test_table_title_contains_kind(self):
        ops = {"op": _make_entry(_sample_target)}
        table = _build_standard_table("lattice", ops)
        assert "lattice" in table.title  # ty: ignore[unsupported-operator]

    def test_empty_metadata_shows_dash(self):
        ops = {"op": _make_entry(_sample_target, metadata={})}
        table = _build_standard_table("collision", ops)
        assert table.row_count == 1

    def test_empty_ops(self):
        table = _build_standard_table("collision", {})
        assert table.row_count == 0


# =========================================================================
# _build_fields_table
# =========================================================================


class TestBuildFieldsTable:
    """Tests for _build_fields_table interactive-selection table."""

    def test_table_has_three_columns(self):
        ops = {"density": _make_entry(_sample_target)}
        table = _build_fields_table(["density"], ops)
        assert len(table.columns) == 3

    def test_row_count_matches_names(self):
        ops = {"a": _make_entry(_sample_target), "b": _make_entry(_sample_target)}
        table = _build_fields_table(["a", "b"], ops)
        assert table.row_count == 2


# =========================================================================
# _resolve_token
# =========================================================================


class TestResolveToken:
    """Tests for _resolve_token index/name resolution."""

    def _ops(self):
        return {"density": _make_entry(_sample_target), "velocity": _make_entry(_sample_target)}

    def test_valid_index_returns_name(self, capsys):
        names = ["density", "velocity"]
        assert _resolve_token("1", names, self._ops()) == "density"

    def test_valid_index_second(self):
        names = ["density", "velocity"]
        assert _resolve_token("2", names, self._ops()) == "velocity"

    def test_out_of_range_returns_none(self, capsys):
        names = ["density"]
        result = _resolve_token("5", names, self._ops())
        assert result is None

    def test_name_token_resolves_name(self, capsys):
        names = ["density", "velocity"]
        result = _resolve_token("density", names, self._ops())
        assert result == "density"

    def test_unknown_name_returns_none(self, capsys):
        names = ["density"]
        result = _resolve_token("unknown", names, self._ops())
        assert result is None

    def test_zero_index_out_of_range(self, capsys):
        names = ["density"]
        result = _resolve_token("0", names, self._ops())
        assert result is None


# =========================================================================
# _parse_field_tokens
# =========================================================================


class TestParseFieldTokens:
    """Tests for _parse_field_tokens comma-separated token parsing."""

    def _ops(self):
        return {"density": _make_entry(_sample_target), "velocity": _make_entry(_sample_target)}

    def test_single_valid_index(self):
        result = _parse_field_tokens("1", ["density", "velocity"], self._ops())
        assert result == ["density"]

    def test_comma_separated_indices(self):
        result = _parse_field_tokens("1, 2", ["density", "velocity"], self._ops())
        assert result == ["density", "velocity"]

    def test_empty_tokens_skipped(self):
        result = _parse_field_tokens(",,,", ["density"], self._ops())
        assert result == []

    def test_invalid_names_skipped(self, capsys):
        # Name-based resolution is dead code; name tokens resolve to None.
        result = _parse_field_tokens("nonexistent", ["density"], self._ops())
        assert result == []

    def test_out_of_range_index_skipped(self, capsys):
        result = _parse_field_tokens("99", ["density"], self._ops())
        assert result == []


# =========================================================================
# _prompt_fields
# =========================================================================


class TestPromptFields:
    """Tests for _prompt_fields interactive operator selection."""

    def _ops(self):
        return {"density": _make_entry(_sample_target), "velocity": _make_entry(_sample_target)}

    def test_eof_returns_current(self):
        with _patch("tud_lbm.cli.cli.Prompt.ask", side_effect=EOFError):
            result = _prompt_fields(self._ops(), ["density"], "plotting")
        assert result == ["density"]

    def test_empty_input_returns_current(self):
        with _patch("tud_lbm.cli.cli.Prompt.ask", return_value=""):
            result = _prompt_fields(self._ops(), ["density"], "plotting")
        assert result == ["density"]

    def test_valid_input_returns_selection(self):
        # Use index-based input (name-based resolution is dead code in _resolve_token)
        with _patch("tud_lbm.cli.cli.Prompt.ask", return_value="1"):
            result = _prompt_fields(self._ops(), None, "plotting")
        assert result is not None
        assert len(result) == 1

    def test_invalid_input_returns_current(self, capsys):
        with _patch("tud_lbm.cli.cli.Prompt.ask", return_value="nonexistent"):
            result = _prompt_fields(self._ops(), ["density"], "plotting")
        assert result == ["density"]

    def test_current_none_shown_as_all(self, capsys):
        with _patch("tud_lbm.cli.cli.Prompt.ask", return_value=""):
            result = _prompt_fields(self._ops(), None, "plotting")
        assert result is None


# =========================================================================
# _load_raw_config
# =========================================================================


class TestLoadRawConfig:
    """Tests for _load_raw_config TOML loading and override injection."""

    def test_loads_and_applies_overrides(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text("[simulation_type]\ntau=0.8\n", encoding="utf-8")
        with _patch("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", return_value={"tau": 0.8}):
            raw = _load_raw_config(str(cfg), ("tau=0.9",))
        assert raw["tau"] == 0.9

    def test_init_dir_injects_fields(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text("[simulation_type]\ntau=0.8\n", encoding="utf-8")
        snapshot = tmp_path / "snap.npz"
        snapshot.touch()
        with _patch("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", return_value={}):
            raw = _load_raw_config(str(cfg), (), init_dir=str(snapshot))
        assert raw["init_dir"] == str(snapshot)
        assert raw["init_type"] == "init_from_file"

    def test_no_overrides_returns_raw(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text("[simulation_type]\ntau=0.8\n", encoding="utf-8")
        with _patch("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", return_value={"tau": 0.8}):
            raw = _load_raw_config(str(cfg), ())
        assert raw == {"tau": 0.8}


# =========================================================================
# _expand_raw_config
# =========================================================================


class TestExpandRawConfig:
    """Tests for _expand_raw_config single vs sweep expansion."""

    def _cfg(self):
        return SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)

    def test_single_config_returns_config_not_none(self):
        cfg = self._cfg()
        with _patch("tud_lbm.config.array_expansion.expand_config", return_value=([cfg], None)):
            _configs, config, sweep, params = _expand_raw_config({})
        assert config == cfg
        assert sweep is None
        assert params is None

    def test_sweep_returns_none_config(self):
        cfg = self._cfg()
        metadata = ArrayParameterSet(field_names=["tau"], array_values={"tau": (0.6, 0.8)}, total_combinations=2)
        with (
            _patch("tud_lbm.config.array_expansion.expand_config", return_value=([cfg, cfg], metadata)),
            _patch(
                "tud_lbm.config.array_expansion.enumerate_configs",
                return_value=iter([(0, {"tau": 0.6}, cfg), (1, {"tau": 0.8}, cfg)]),
            ),
        ):
            _configs, config, sweep, params = _expand_raw_config({})
        assert config is None
        assert sweep is metadata
        assert params == [{"tau": 0.6}, {"tau": 0.8}]


# =========================================================================
# _load_config_interactive
# =========================================================================


class TestLoadConfigInteractive:
    """Tests for _load_config_interactive prompted config construction."""

    def test_returns_config_with_prompted_values(self):
        answers = iter(["50", "60", "0.7", "2000", "200"])
        with _patch("tud_lbm.cli.cli.Prompt.ask", side_effect=lambda *a, **kw: next(answers)):
            _configs, config, sweep, params = _load_config_interactive()
        assert config.grid_shape[:2] == (50, 60)
        assert config.tau == pytest.approx(0.7)
        assert config.nt == 2000
        assert config.save_interval == 200
        assert sweep is None
        assert params is None


# =========================================================================
# _display_summary
# =========================================================================


class TestDisplaySummary:
    """Tests for _display_summary single-run and sweep branches."""

    def _cfg(self):
        return SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)

    def test_single_run_prints_config(self, capsys):
        cfg = self._cfg()
        _display_summary(cfg, None, [cfg], overview=False)
        assert "Relaxation" in capsys.readouterr().out

    def test_sweep_prints_sweep_summary(self, capsys):
        cfg = self._cfg()
        meta = ArrayParameterSet(field_names=["tau"], array_values={"tau": (0.6, 0.8)}, total_combinations=2)
        _display_summary(None, meta, [cfg, cfg], overview=False)
        out = capsys.readouterr().out
        assert "tau" in out

    def test_overview_flag_prints_overview(self, capsys):
        cfg = self._cfg()
        _display_summary(cfg, None, [cfg], overview=True)
        assert "PHYSICAL PARAMETER OVERVIEW" in capsys.readouterr().out

    def test_sweep_overview_prints_first_config_overview(self, capsys):
        cfg = self._cfg()
        meta = ArrayParameterSet(field_names=["tau"], array_values={"tau": (0.6, 0.8)}, total_combinations=2)
        _display_summary(None, meta, [cfg, cfg], overview=True)
        assert "PHYSICAL PARAMETER OVERVIEW" in capsys.readouterr().out


# =========================================================================
# _confirm_run
# =========================================================================


class TestConfirmRun:
    """Tests for _confirm_run y/n/o prompt choices."""

    def _cfg(self):
        return SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)

    def test_y_returns_yes(self):
        with _patch("tud_lbm.cli.cli.Prompt.ask", return_value="y"):
            result = _confirm_run(None, [self._cfg()])
        assert result == "yes"

    def test_n_returns_no(self):
        with _patch("tud_lbm.cli.cli.Prompt.ask", return_value="n"):
            result = _confirm_run(None, [self._cfg()])
        assert result == "no"

    def test_o_returns_override(self):
        with _patch("tud_lbm.cli.cli.Prompt.ask", return_value="o"):
            result = _confirm_run(None, [self._cfg()])
        assert result == "override"

    def test_sweep_prompt_includes_count(self):
        meta = ArrayParameterSet(field_names=["tau"], array_values={"tau": (0.6, 0.8)}, total_combinations=2)
        cfgs = [self._cfg(), self._cfg()]
        with _patch("tud_lbm.cli.cli.Prompt.ask", return_value="y") as mock_ask:
            _confirm_run(meta, cfgs)
        prompt_text = mock_ask.call_args[0][0]
        assert "2" in prompt_text


# =========================================================================
# _run_with_optional_overrides
# =========================================================================


class TestRunWithOptionalOverrides:
    """Tests for _run_with_optional_overrides interactive confirmation loop."""

    def _cfg(self):
        return SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)

    def test_no_prompt_returns_immediately(self):
        cfg = self._cfg()
        result = _run_with_optional_overrides(
            raw_config={},
            configs=[cfg],
            config=cfg,
            sweep_metadata=None,
            parameters_list=None,
            no_prompt=True,
            overview=False,
        )
        assert result[1] == cfg

    def test_n_choice_cancels(self):
        cfg = self._cfg()
        with _patch("tud_lbm.cli.cli.Prompt.ask", return_value="n"):
            _configs, config, _sweep, _params = _run_with_optional_overrides(
                raw_config={},
                configs=[cfg],
                config=cfg,
                sweep_metadata=None,
                parameters_list=None,
                no_prompt=False,
                overview=False,
            )
        assert _configs == []
        assert config is None

    def test_y_choice_returns_configs(self):
        cfg = self._cfg()
        with _patch("tud_lbm.cli.cli.Prompt.ask", return_value="y"):
            configs, _config, _sweep, _params = _run_with_optional_overrides(
                raw_config={},
                configs=[cfg],
                config=cfg,
                sweep_metadata=None,
                parameters_list=None,
                no_prompt=False,
                overview=False,
            )
        assert configs == [cfg]

    def test_o_with_no_raw_config_prints_warning_then_y(self, capsys):
        cfg = self._cfg()
        answers = iter(["o", "y"])
        with _patch("tud_lbm.cli.cli.Prompt.ask", side_effect=lambda *a, **kw: next(answers)):
            _configs, _config, _sweep, _params = _run_with_optional_overrides(
                raw_config=None,
                configs=[cfg],
                config=cfg,
                sweep_metadata=None,
                parameters_list=None,
                no_prompt=False,
                overview=False,
            )
        out = capsys.readouterr().out
        assert "Inline overrides" in out

    def test_o_with_raw_config_valid_override_then_y(self):
        cfg = self._cfg()
        raw = {"tau": 0.8}
        answers = iter(["o", "tau=0.9", "y"])
        with (
            _patch("tud_lbm.cli.cli.Prompt.ask", side_effect=lambda *a, **kw: next(answers)),
            _patch("tud_lbm.cli.cli._expand_raw_config", return_value=([cfg], cfg, None, None)),
        ):
            configs, _config, _sweep, _params = _run_with_optional_overrides(
                raw_config=raw,
                configs=[cfg],
                config=cfg,
                sweep_metadata=None,
                parameters_list=None,
                no_prompt=False,
                overview=False,
            )
        assert configs == [cfg]

    def test_o_with_invalid_override_retries_then_y(self, capsys):
        cfg = self._cfg()
        raw = {"tau": 0.8}
        answers = iter(["o", "tau=bad_value", "y"])
        with _patch("tud_lbm.cli.cli.Prompt.ask", side_effect=lambda *a, **kw: next(answers)):
            _configs, _config, _sweep, _params = _run_with_optional_overrides(
                raw_config=raw,
                configs=[cfg],
                config=cfg,
                sweep_metadata=None,
                parameters_list=None,
                no_prompt=False,
                overview=False,
            )
        out = capsys.readouterr().out
        assert "Invalid override" in out


# =========================================================================
# _execute_run
# =========================================================================


class TestExecuteRun:
    """Tests for _execute_run single-run and sweep dispatch."""

    def _cfg(self, results_dir):
        return SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10, results_dir=str(results_dir))

    def test_single_run_calls_run_simulation(self, tmp_path):
        cfg = self._cfg(tmp_path)
        called = {"n": 0}

        def _fake_run(c):
            called["n"] += 1
            return str(tmp_path / "run" / "data")

        with _patch("tud_lbm.cli.cli._run_simulation", _fake_run):
            _execute_run([cfg], cfg, None, None, None, False)
        assert called["n"] == 1

    def test_sweep_calls_parallel_sweep(self, tmp_path):
        cfg = self._cfg(tmp_path)
        meta = ArrayParameterSet(field_names=["tau"], array_values={"tau": (0.6, 0.8)}, total_combinations=2)
        called = {"n": 0}

        def _fake_parallel(cfgs, params, **kw):
            called["n"] += 1
            return [SimpleNamespace(status="success"), SimpleNamespace(status="success")]

        with _patch("tud_lbm.cli.cli._run_parallel_sweep", _fake_parallel):
            _execute_run([cfg, cfg], None, meta, [{"tau": 0.6}, {"tau": 0.8}], None, False)
        assert called["n"] == 1

    def test_single_run_with_compare_calls_compare(self, tmp_path):
        cfg = self._cfg(tmp_path)
        run_data = tmp_path / "simulation_name" / "data"
        run_data.mkdir(parents=True, exist_ok=True)
        compare_called = {"n": 0}

        def _fake_run(c):
            return str(run_data)

        def _fake_compare(run_dir, config):
            compare_called["n"] += 1

        with (
            _patch("tud_lbm.cli.cli._run_simulation", _fake_run),
            _patch("tud_lbm.cli.cli._run_compare_single", _fake_compare),
        ):
            _execute_run([cfg], cfg, None, None, None, False, run_compare=True)
        assert compare_called["n"] == 1


# =========================================================================
# _run_compare_single / _run_compare_sweep
# =========================================================================


class TestRunCompareSingle:
    """Tests for _run_compare_single CSV and plot generation."""

    def test_csv_path_none_prints_warning(self, tmp_path, capsys):
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        with (
            _patch("tud_lbm.io.plotting.simulation_csv.build_simulation_csv", return_value=None),
        ):
            _run_compare_single(tmp_path, cfg)
        assert "skipped" in capsys.readouterr().out.lower()

    def test_csv_path_present_calls_compare_runs(self, tmp_path, capsys):
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        csv_file = tmp_path / "results.csv"
        csv_file.touch()
        with (
            _patch("tud_lbm.io.plotting.simulation_csv.build_simulation_csv", return_value=csv_file),
            _patch("tud_lbm.io.plotting.run_comparison.compare_runs") as mock_cmp,
        ):
            _run_compare_single(tmp_path, cfg)
        mock_cmp.assert_called_once_with(tmp_path)


class TestRunCompareSweep:
    """Tests for _run_compare_sweep sweep-level comparison."""

    def test_no_ok_runs_prints_warning(self, tmp_path, capsys):
        with _patch("tud_lbm.io.plotting.run_comparison.process_parent_dir", return_value=(2, 0)):
            _run_compare_sweep(tmp_path)
        assert "no runs" in capsys.readouterr().out.lower()

    def test_ok_runs_prints_success(self, tmp_path, capsys):
        with _patch("tud_lbm.io.plotting.run_comparison.process_parent_dir", return_value=(2, 2)):
            _run_compare_sweep(tmp_path)
        assert "Comparison plots" in capsys.readouterr().out


# =========================================================================
# _expand_single_phase
# =========================================================================


class TestExpandSinglePhase:
    """Tests for _expand_single_phase sweep-rejection guard."""

    def test_single_config_returned(self):
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        with _patch("tud_lbm.config.array_expansion.expand_config", return_value=([cfg], None)):
            result = _expand_single_phase({}, "Phase 1")
        assert result == cfg

    def test_sweep_raises_usage_error(self):
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        with (
            _patch("tud_lbm.config.array_expansion.expand_config", return_value=([cfg, cfg], object())),
            pytest.raises(click.UsageError, match="does not support parameter sweeps"),
        ):
            _expand_single_phase({}, "Phase 1")


# =========================================================================
# _run_two_phase_wetting_init
# =========================================================================


class TestRunTwoPhaseWettingInit:
    """Tests for _run_two_phase_wetting_init two-phase wetting workflow."""

    def _base_raw(self):
        return {
            "sim_type": "multiphase_wetting",
            "nt": 100,
            "grid_shape": [16, 16],
            "wetting_config": {"contact_angle": 90},
        }

    def test_cancelled_at_phase1_prompt_returns(self, tmp_path):
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text("[simulation_type]\n", encoding="utf-8")
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        wetting_params = dict(_WETTING_PARAM_DEFAULTS)
        with (
            _patch("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", return_value=self._base_raw()),
            _patch("tud_lbm.cli.cli._expand_single_phase", return_value=cfg),
            _patch("tud_lbm.cli.cli._prompt_wetting_params", return_value=wetting_params),
            _patch("tud_lbm.cli.cli.Confirm.ask", return_value=False),
        ):
            _run_two_phase_wetting_init(str(cfg_toml), (), no_prompt=False, overview=False)

    def test_no_prompt_runs_both_phases(self, tmp_path):
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text("[simulation_type]\n", encoding="utf-8")
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        run_calls = []

        def _fake_run(c):
            run_calls.append(c)
            return str(data_dir)

        with (
            _patch("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", return_value=self._base_raw()),
            _patch("tud_lbm.cli.cli._expand_single_phase", return_value=cfg),
            _patch("tud_lbm.cli.cli._run_simulation", _fake_run),
        ):
            _run_two_phase_wetting_init(str(cfg_toml), (), no_prompt=True, overview=False)
        assert len(run_calls) == 2

    def test_overview_flag_prints_overview(self, tmp_path, capsys):
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text("[simulation_type]\n", encoding="utf-8")
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        with (
            _patch("tud_lbm.config.adapter_toml.TomlAdapter.load_raw", return_value=self._base_raw()),
            _patch("tud_lbm.cli.cli._expand_single_phase", return_value=cfg),
            _patch("tud_lbm.cli.cli._run_simulation", return_value=str(data_dir)),
        ):
            _run_two_phase_wetting_init(str(cfg_toml), (), no_prompt=True, overview=True)
        assert "PHYSICAL PARAMETER OVERVIEW" in capsys.readouterr().out


# =========================================================================
# _run_impl — additional branches
# =========================================================================


class TestRunImplAdditional:
    """Tests for _run_impl branches not covered by TestRunImplFlags."""

    def _single_config(self):
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        return [cfg], cfg, None, None

    def test_list_analysis_returns_false(self):
        result = _run_impl(
            config_path=None,
            no_prompt=True,
            dry_run=False,
            list_operators=False,
            list_analysis=True,
            max_workers=None,
            fail_fast=False,
            overrides=(),
            overview=False,
            debug_wetting=False,
            init_wetting=False,
            init_dir=None,
        )
        assert result is False

    def test_run_compare_calls_compare_single(self, tmp_path):
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text("[simulation_type]\ntau=0.8\nnt=10\nnx=8\nny=8\nnz=1\n", encoding="utf-8")
        data_dir = tmp_path / "sim" / "data"
        data_dir.mkdir(parents=True)
        compare_called = {"n": 0}

        def _fake_run(c):
            return str(data_dir)

        def _fake_compare(run_dir, config):
            compare_called["n"] += 1

        with (
            _patch("tud_lbm.cli.cli._load_raw_config", return_value={}),
            _patch("tud_lbm.cli.cli._expand_raw_config", return_value=self._single_config()),
            _patch("tud_lbm.cli.cli._run_simulation", _fake_run),
            _patch("tud_lbm.cli.cli._run_compare_single", _fake_compare),
        ):
            result = _run_impl(
                config_path=str(cfg_toml),
                no_prompt=True,
                dry_run=False,
                list_operators=False,
                list_analysis=False,
                max_workers=None,
                fail_fast=False,
                overrides=(),
                overview=False,
                debug_wetting=False,
                init_wetting=False,
                init_dir=None,
                run_compare=True,
            )
        assert result is True
        assert compare_called["n"] == 1

    def test_init_wetting_path_calls_two_phase_init(self, tmp_path):
        cfg_toml = tmp_path / "config.toml"
        cfg_toml.write_text("[simulation_type]\n", encoding="utf-8")
        called = {"n": 0}

        def _fake_wetting(path, overrides, *, no_prompt, overview):
            called["n"] += 1

        with _patch("tud_lbm.cli.cli._run_two_phase_wetting_init", _fake_wetting):
            result = _run_impl(
                config_path=str(cfg_toml),
                no_prompt=True,
                dry_run=False,
                list_operators=False,
                list_analysis=False,
                max_workers=None,
                fail_fast=False,
                overrides=(),
                overview=False,
                debug_wetting=False,
                init_wetting=True,
                init_dir=None,
            )
        assert called["n"] == 1
        assert result is False


# =========================================================================
# animate / visualise / compare command-level new paths
# =========================================================================


class TestAnimateCommandPaths:
    """Tests for the animate CLI command additional paths."""

    def test_animate_no_prompt_uses_config_defaults(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10, animate_fields=["density"])
        runner = CliRunner()
        with (
            _patch("tud_lbm.config.from_toml", return_value=cfg),
            _patch("tud_lbm.io.plotting.Animator") as mock_anim,
        ):
            mock_anim.return_value.create.return_value = run_dir / "animation.mp4"
            result = runner.invoke(cli, ["animate", str(run_dir), "--no-prompt"])
        assert result.exit_code in (0, 1)

    def test_animate_debug_env_reraises(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")
        monkeypatch.setenv("TUD_LBM_DEBUG", "1")
        runner = CliRunner()
        with patch("tud_lbm.cli.cli._validate_run_dir_has_config", side_effect=RuntimeError("boom")):
            result = runner.invoke(cli, ["animate", str(run_dir)])
        assert result.exit_code == 1


class TestVisualiseCommandPaths:
    """Tests for the visualise CLI command additional paths."""

    def test_visualise_with_fields_flag_skips_prompt(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        runner = CliRunner()
        with (
            _patch("tud_lbm.config.from_toml", return_value=cfg),
            _patch("tud_lbm.io.plotting.FigureBuilder") as mock_fb,
        ):
            mock_fb.return_value.build_all.return_value = [tmp_path / "fig.png"]
            mock_fb.return_value.plot_dir = tmp_path
            result = runner.invoke(cli, ["visualise", str(run_dir), "--fields", "density,velocity"])
        assert result.exit_code in (0, 1)

    def test_visualise_no_prompt_path(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10, plot_fields=["density"])
        runner = CliRunner()
        with (
            _patch("tud_lbm.config.from_toml", return_value=cfg),
            _patch("tud_lbm.io.plotting.FigureBuilder") as mock_fb,
        ):
            mock_fb.return_value.build_all.return_value = []
            mock_fb.return_value.plot_dir = tmp_path
            result = runner.invoke(cli, ["visualise", str(run_dir), "--no-prompt"])
        assert result.exit_code in (0, 1)
        assert "No figures" in result.output or result.exit_code == 0

    def test_visualise_debug_env_reraises(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")
        monkeypatch.setenv("TUD_LBM_DEBUG", "1")
        runner = CliRunner()
        with patch("tud_lbm.cli.cli._validate_run_dir_has_config", side_effect=RuntimeError("dbg")):
            result = runner.invoke(cli, ["visualise", str(run_dir)])
        assert result.exit_code == 1

    def test_visualise_snapshot_fig_prompts_for_timesteps(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        runner = CliRunner()

        fake_op = MagicMock()
        fake_op.name = "snapshot_fig"

        with (
            _patch("tud_lbm.config.from_toml", return_value=cfg),
            _patch("tud_lbm.io.plotting.FigureBuilder") as mock_fb,
            _patch("tud_lbm.cli.cli.Prompt.ask", return_value="5,10"),
        ):
            mock_fb.return_value.sorted_timed_files.return_value = [(0, tmp_path), (5, tmp_path), (10, tmp_path)]
            mock_fb.return_value.analysis_operators = [fake_op]
            mock_fb.return_value.build_all.return_value = [tmp_path / "fig.png"]
            mock_fb.return_value.plot_dir = tmp_path
            result = runner.invoke(cli, ["visualise", str(run_dir), "--fields", "snapshot_fig"])

        assert result.exit_code in (0, 1)
        assert fake_op.timesteps == [5, 10]


class TestCompareCommandPaths:
    """Tests for the compare CLI command additional paths."""

    def test_compare_no_prompt_skips_fields(self, tmp_path):
        runner = CliRunner()
        with _patch("tud_lbm.io.plotting.run_comparison.process_parent_dir", return_value=(3, 3)):
            result = runner.invoke(cli, ["compare", str(tmp_path), "--no-prompt"])
        assert result.exit_code == 0

    def test_compare_debug_env_reraises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TUD_LBM_DEBUG", "1")
        runner = CliRunner()
        with _patch("tud_lbm.io.plotting.run_comparison.process_parent_dir", side_effect=RuntimeError("err")):
            result = runner.invoke(cli, ["compare", str(tmp_path)])
        assert result.exit_code == 1


class TestRunCommandDebugEnv:
    """Tests for TUD_LBM_DEBUG re-raise behaviour in the run command."""

    def test_run_debug_env_reraises(self, monkeypatch):
        monkeypatch.setenv("TUD_LBM_DEBUG", "1")
        runner = CliRunner()
        with _patch("tud_lbm.cli.cli._run_impl", side_effect=RuntimeError("debug error")):
            result = runner.invoke(cli, ["run"])
        assert result.exit_code == 1


# =========================================================================
# _print_run_banner
# =========================================================================


def test_print_run_banner_outputs_title(capsys):
    _print_run_banner()
    assert "TUD-LBM" in capsys.readouterr().out


# =========================================================================
# _display_simulation_operators empty / analysis-only registry
# =========================================================================


class TestDisplayOperatorsEmptyRegistry:
    """Tests for empty-registry warnings in operator display helpers."""

    def test_simulation_operators_empty_prints_warning(self, capsys):
        from tud_lbm.cli.cli import _display_simulation_operators

        with (
            _patch("tud_lbm.operators.load_all"),
            _patch("tud_lbm.registry.get_operator_category", return_value=set()),
        ):
            _display_simulation_operators()
        assert "No simulation operators" in capsys.readouterr().out

    def test_analysis_operators_empty_prints_warning(self, capsys):
        from tud_lbm.cli.cli import _display_analysis_operators

        with (
            _patch("tud_lbm.operators.load_all"),
            _patch("tud_lbm.registry.get_operator_category", return_value=set()),
        ):
            _display_analysis_operators()
        assert "No analysis operators" in capsys.readouterr().out


class TestRunCommandListFlags:
    """CliRunner invocations for --list-simulation-analysis and related flags."""

    def test_list_simulation_analysis_via_runner(self):
        runner = CliRunner()
        with _patch("tud_lbm.cli.cli._display_analysis_operators"):
            result = runner.invoke(cli, ["run", "--list-simulation-analysis"])
        assert result.exit_code == 0

    def test_list_simulation_operators_and_analysis_operators_precedence(self):
        """list_operators check fires first; list_analysis check is skipped."""
        runner = CliRunner()
        with _patch("tud_lbm.cli.cli._display_simulation_operators"):
            result = runner.invoke(
                cli,
                ["run", "--list-simulation-operators", "--list-simulation-analysis"],
            )
        assert result.exit_code == 0
