"""The exit-code contract shared by every CLI command.

These are parametrized over ``cli.commands``, so a command added later is
covered without editing this file.
"""

from __future__ import annotations
import pytest
from src.cli.app import cli
from src.cli.commands import cli as loaded_cli

#: Where each command's work happens — the seam an injected failure must cross.
_SEAM = {
    "run": "src.cli.commands.run._run_impl",
    "benchmark": "src.cli.commands.benchmark.run_benchmark",
    "animate": "src.cli.commands.visualise._validate_run_dir_has_config",
    "visualise": "src.cli.commands.visualise._validate_run_dir_has_config",
    "compare": "src.cli.analysis_routing.analyse_tree",
    "regime-map": "src.simulation_io.plotting.regime_map_plot.build_regime_map",
    "analyse": "src.cli.commands.analysis._load_single_config",
}

_EXIT_INTERRUPTED = 130
_EXIT_USAGE = 2


def _args(name: str, tmp_path) -> list[str]:
    """Minimal valid arguments for *name*, enough to reach the seam."""
    if name == "run":
        cfg = tmp_path / "config.toml"
        cfg.write_text("", encoding="utf-8")
        return [name, str(cfg)]
    if name == "regime-map":
        txt = tmp_path / "dirs.txt"
        txt.write_text("", encoding="utf-8")
        return [name, str(txt)]
    if name == "benchmark":
        cfg = tmp_path / "config.toml"
        cfg.write_text("", encoding="utf-8")
        return [name, str(cfg)]
    if name == "analyse":
        cfg = tmp_path / "config.toml"
        cfg.write_text("", encoding="utf-8")
        return [name, str(cfg), "--surface-tension"]
    return [name, str(tmp_path)]


def test_every_command_has_a_seam():
    """A new command must be added to _SEAM rather than silently untested."""
    assert set(loaded_cli.commands) == set(_SEAM)


@pytest.mark.parametrize("name", sorted(_SEAM))
def test_interrupt_exits_130(runner, monkeypatch, tmp_path, name):
    monkeypatch.setattr(_SEAM[name], _raise(KeyboardInterrupt))

    result = runner.invoke(cli, _args(name, tmp_path))

    assert result.exit_code == _EXIT_INTERRUPTED


@pytest.mark.parametrize("name", sorted(_SEAM))
def test_unexpected_error_exits_1_with_message(runner, monkeypatch, tmp_path, name):
    monkeypatch.setattr(_SEAM[name], _raise(RuntimeError("boom")))

    result = runner.invoke(cli, _args(name, tmp_path))

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "boom" in result.output


@pytest.mark.parametrize("name", sorted(_SEAM))
def test_debug_env_reraises(runner, monkeypatch, tmp_path, name):
    """TUD_LBM_DEBUG keeps the traceback instead of swallowing it."""
    monkeypatch.setenv("TUD_LBM_DEBUG", "1")
    monkeypatch.setattr(_SEAM[name], _raise(RuntimeError("boom")))

    result = runner.invoke(cli, _args(name, tmp_path))

    assert isinstance(result.exception, RuntimeError)


def test_usage_error_exits_2_not_1(runner):
    """--override without CONFIG_PATH is a usage error, so click's code 2.

    This changed from 1: the guard used to raise inside the command's own
    try/except and was reported as a generic failure.
    """
    result = runner.invoke(cli, ["run", "--override", "tau=0.7"])

    assert result.exit_code == _EXIT_USAGE


def test_analyse_without_an_analysis_is_a_usage_error(runner, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("", encoding="utf-8")

    result = runner.invoke(cli, ["analyse", str(cfg)])

    assert result.exit_code == _EXIT_USAGE
    assert "at least one analysis" in result.output


def _raise(exc: type[BaseException] | BaseException):
    """A stand-in callable that always raises *exc*."""

    def _boom(*_args: object, **_kwargs: object):
        raise exc

    return _boom
