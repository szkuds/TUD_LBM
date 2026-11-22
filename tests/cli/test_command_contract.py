"""The exit-code contract shared by every CLI command.

These are parametrized over every leaf command of ``cli.commands``, subcommands
of a group included, so a command added later is covered without editing this
file.
"""

from __future__ import annotations
import click
import pytest
from src.cli.app import cli
from src.cli.commands import cli as loaded_cli

#: Where each command's work happens — the seam an injected failure must cross.
_SEAM = {
    "run": "src.cli.commands.run._run_impl",
    "benchmark": "src.cli.commands.benchmark.run_benchmark",
    "visualise fields": "src.cli.commands.visualise._validate_run_dir_has_config",
    "visualise analysis": "src.cli.commands.visualise._validate_run_dir_has_config",
    "visualise setup": "src.cli.commands.visualise._validate_run_dir_has_config",
    "compare runs": "src.cli.analysis_routing.resolve_run_set",
    "compare regime-map": "src.cli.analysis_routing.resolve_run_set",
    "analyse surface-tension": "src.cli.commands.analysis._load_single_config",
    "analyse coexistence": "src.cli.commands.analysis._load_single_config",
    "analyse length-scale": "src.cli.commands.analysis._validate_run_dir_has_config",
    "calibration stage": "src.cli.commands.calibration._load_raw_config",
    "calibration collect": "src.simulation_io.analysis.surface_tension.find_sweep_runs",
    "calibration refresh": "src.config.adapter_toml.TomlAdapter",
    "sweep stage": "src.cli.commands.sweep._load_raw_config",
}

_EXIT_INTERRUPTED = 130
_EXIT_USAGE = 2


#: Commands whose first argument is a file: (file name, arguments after it).
_FILE_ARGS: dict[str, tuple[str, list[str]]] = {
    "run": ("config.toml", []),
    "benchmark": ("config.toml", []),
    "calibration stage": ("config.toml", []),
    "analyse surface-tension": ("config.toml", []),
    "analyse coexistence": ("config.toml", []),
    "sweep stage": ("config.toml", ["--set", "tau=[0.8]"]),
}


def _leaf_commands(group: click.Group, prefix: tuple[str, ...] = ()) -> set[str]:
    """Every command of *group* that does work, as space-joined paths."""
    leaves: set[str] = set()
    for name, command in group.commands.items():
        path = (*prefix, name)
        if isinstance(command, click.Group):
            leaves |= _leaf_commands(command, path)
        else:
            leaves.add(" ".join(path))
    return leaves


def _args(name: str, tmp_path) -> list[str]:
    """Minimal valid arguments for *name*, enough to reach the seam."""
    words = name.split()
    if name not in _FILE_ARGS:
        return [*words, str(tmp_path)]
    filename, after = _FILE_ARGS[name]
    path = tmp_path / filename
    path.write_text("", encoding="utf-8")
    return [*words, str(path), *after]


def test_every_command_has_a_seam():
    """A new command must be added to _SEAM rather than silently untested."""
    assert _leaf_commands(loaded_cli) == set(_SEAM)


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


def test_removed_commands_are_unknown(runner):
    """The overlapping commands are gone, not aliased."""
    for name in ("animate", "setup-figure", "regime-map"):
        result = runner.invoke(cli, [name])

        assert result.exit_code == _EXIT_USAGE
        assert "No such command" in result.output


def _raise(exc: type[BaseException] | BaseException):
    """A stand-in callable that always raises *exc*."""

    def _boom(*_args: object, **_kwargs: object):
        raise exc

    return _boom
