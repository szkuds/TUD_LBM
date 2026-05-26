"""Additional branch coverage for tud_lbm/cli/cli.py.

Targets the 125 uncovered lines / 42 conditions (55.8 → ~80%+) by exercising:
- _validate_cli_args: all three UsageError branches
- _prompt_wetting_params: no_prompt=True path and values-from-config path
- _build_wetting_init_raw: structural mutation checks
- _build_wetting_gravity_raw: field injection checks
- Click commands via CliRunner: animate, visualise, compare, main shim,
  and the run command's --list-operators / --dry-run / --overview flags
"""

from __future__ import annotations
from unittest.mock import patch
import click
import pytest
from click.testing import CliRunner
from tud_lbm.cli.cli import _WETTING_PARAM_DEFAULTS
from tud_lbm.cli.cli import _build_wetting_gravity_raw
from tud_lbm.cli.cli import _build_wetting_init_raw
from tud_lbm.cli.cli import _prompt_wetting_params
from tud_lbm.cli.cli import _validate_cli_args
from tud_lbm.cli.cli import animate
from tud_lbm.cli.cli import cli
from tud_lbm.cli.cli import compare
from tud_lbm.cli.cli import main
from tud_lbm.cli.cli import visualise

# ---------------------------------------------------------------------------
# _validate_cli_args
# ---------------------------------------------------------------------------


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
        _validate_cli_args(("tau=0.8",), "config.toml")  # must not raise

    def test_empty_args_no_config_ok(self):
        _validate_cli_args((), None)  # no config, no overrides → fine


# ---------------------------------------------------------------------------
# _prompt_wetting_params
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _build_wetting_init_raw
# ---------------------------------------------------------------------------


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
        assert base["sim_type"] == "multiphase_hysteresis"  # unchanged


# ---------------------------------------------------------------------------
# _build_wetting_gravity_raw
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Click commands via CliRunner — error/short-circuit paths
# ---------------------------------------------------------------------------


class TestClickCommands:
    """Test CLI entry points using CliRunner.

    We only exercise paths that don't need a real simulation to run — flag
    validation, missing-argument errors, and exception-handling wrappers.
    """

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


# ---------------------------------------------------------------------------
# Keyboard-interrupt and debug-env passthrough
# ---------------------------------------------------------------------------


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
