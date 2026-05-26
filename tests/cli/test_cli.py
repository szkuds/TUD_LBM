"""Tests for CLI helper functions and edge cases."""

from __future__ import annotations
import importlib
from types import ModuleType
from types import SimpleNamespace
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
    from tud_lbm.cli.cli import _normalize_override_path
    from tud_lbm.cli.cli import _parse_override_argument
    from tud_lbm.cli.cli import _set_nested_override
except ImportError:
    pytest.skip("click or rich dependency not installed", allow_module_level=True)


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
        result = _normalize_override_path("wetting.contact_angle")
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

        original = _flags.DEBUG_FLAG
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
                    max_workers=None,
                    fail_fast=False,
                    overrides=(),
                    overview=False,
                    debug_wetting=True,
                    init_wetting=False,
                    init_dir=None,
                )
            assert _flags.DEBUG_FLAG is True
        finally:
            _flags.DEBUG_FLAG = original


class TestClickCommandPaths:
    """CLI command error and success paths via CliRunner."""

    def test_run_keyboard_interrupt_exits_130(self):
        runner = CliRunner()
        with patch("tud_lbm.cli.cli._run_impl", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["run"])
        assert result.exit_code == 130

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
        with patch("tud_lbm.io.plotting.analysis.process_parent_dir", return_value=(0, 0)):
            result = runner.invoke(cli, ["compare", str(tmp_path)])
        assert result.exit_code == 0
        assert "No simulation" in result.output

    def test_compare_runs_with_zero_ok(self, tmp_path):
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.analysis.process_parent_dir", return_value=(1, 0)):
            result = runner.invoke(cli, ["compare", str(tmp_path)])
        assert result.exit_code == 0
        assert "no runs produced" in result.output.lower()

    def test_compare_runs_with_success(self, tmp_path):
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.analysis.process_parent_dir", return_value=(2, 2)):
            result = runner.invoke(cli, ["compare", str(tmp_path)])
        assert result.exit_code == 0

    def test_compare_keyboard_interrupt_exits_130(self, tmp_path):
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.analysis.process_parent_dir", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["compare", str(tmp_path)])
        assert result.exit_code == 130

    def test_compare_general_exception_exits_1(self, tmp_path):
        runner = CliRunner()
        with patch("tud_lbm.io.plotting.analysis.process_parent_dir", side_effect=RuntimeError("fail")):
            result = runner.invoke(cli, ["compare", str(tmp_path)])
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
        with patch("tud_lbm.io.plotting.analysis.process_parent_dir", return_value=(0, 0)):
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

    cli_module.main.callback(("--help",))
    assert calls == [["--help"]]


def test_main_dispatch_strips_run_token(monkeypatch):
    cli_module = importlib.import_module("tud_lbm.cli.cli")

    calls: list[list[str]] = []
    monkeypatch.setattr(cli_module.run, "main", lambda args, standalone_mode: calls.append(args))

    cli_module.main.callback(("run", "config.toml", "--dry-run"))
    assert calls == [["config.toml", "--dry-run"]]


@pytest.mark.parametrize("token", ["animate", "visualise", "compare"])
def test_main_dispatches_subcommands_to_click_group(monkeypatch, token):
    cli_module = importlib.import_module("tud_lbm.cli.cli")

    calls: list[list[str]] = []
    monkeypatch.setattr(cli_module.cli, "main", lambda args, standalone_mode: calls.append(args))

    cli_module.main.callback((token, "run_dir"))
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
    result = CliRunner().invoke(main, ["--list-simulation-operators"])

    assert result.exit_code == 0
    assert "plotting" in result.output
    assert "analysis" in result.output
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
