"""The ``tud-lbm setup-figure`` command."""

from __future__ import annotations
from src.cli.commands import cli
from tests.io.test_setup_figure import step_config
from tests.support.run_dirs import build_run_dir

_EXIT_USAGE = 2


def test_cli_writes_the_figure(runner, tmp_path):
    run_dir = build_run_dir(tmp_path, config=step_config())
    out = tmp_path / "setup.png"

    result = runner.invoke(cli, ["setup-figure", str(run_dir), "--output", str(out)])

    assert result.exit_code == 0, result.output
    assert out.is_file()


def test_cli_rejects_unknown_timestep(runner, tmp_path):
    run_dir = build_run_dir(tmp_path, config=step_config())

    result = runner.invoke(cli, ["setup-figure", str(run_dir), "--timestep", "999"])

    assert result.exit_code == _EXIT_USAGE
