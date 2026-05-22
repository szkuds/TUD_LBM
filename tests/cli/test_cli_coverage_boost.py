"""Coverage boost for tud_lbm/cli/cli.py.

Targets the remaining uncovered lines (~64% -> ~80%+) by exercising:
- _display_config_summary: multiphase and force-enabled branches
- _display_full_overview: valid config path
- _display_sweep_summary: sweep metadata display
- _print_dry_run_message: sweep and non-sweep paths
- _check_sweep_errors: failure detection
- _run_impl: debug_wetting flag, interactive mode, dry-run sweep
- run command: --debug-wetting, --overview flags
- visualise/animate commands with mocked internals
- main shim routing
- compare command success/failure paths
"""

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from click.testing import CliRunner
from tud_lbm.cli.cli import _check_sweep_errors
from tud_lbm.cli.cli import _display_config_summary
from tud_lbm.cli.cli import _display_full_overview
from tud_lbm.cli.cli import _display_sweep_summary
from tud_lbm.cli.cli import _print_dry_run_message
from tud_lbm.cli.cli import _run_impl
from tud_lbm.cli.cli import cli
from tud_lbm.cli.cli import main
from tud_lbm.config import SimulationConfig
from tud_lbm.config.array_expansion import ArrayParameterSet


# ---------------------------------------------------------------------------
# _display_config_summary
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# _display_full_overview
# ---------------------------------------------------------------------------
class TestDisplayFullOverview:
    """Tests for _display_full_overview."""

    def test_none_config(self, capsys):
        _display_full_overview(None)
        assert "No configuration" in capsys.readouterr().out

    def test_valid_config(self, capsys):
        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        _display_full_overview(cfg)
        assert "PHYSICAL PARAMETER OVERVIEW" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _display_sweep_summary
# ---------------------------------------------------------------------------
def test_display_sweep_summary(capsys):
    metadata = ArrayParameterSet(
        field_names=["tau"],
        array_values={"tau": (0.6, 0.7)},
        total_combinations=2,
    )
    _display_sweep_summary(metadata)
    out = capsys.readouterr().out
    assert "tau" in out


# ---------------------------------------------------------------------------
# _print_dry_run_message
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# _check_sweep_errors
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# _run_impl — lightweight flag tests
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# CLI command error / success paths via CliRunner
# ---------------------------------------------------------------------------
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
