"""The ``visualise`` group and its ``fields`` / ``analysis`` subcommands.

Two click behaviours are load-bearing here and pinned first:

1. ``invoke_without_command=True`` with a required positional on the *group* —
   click binds the argument before dispatching, so ``visualise ./out fields``
   means ``RUN_DIR=./out`` plus subcommand ``fields``.
2. ``cli_command`` cannot wrap the group callback, because click invokes
   subcommands only after that callback returns. The work functions carry it.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from tests.support.run_dirs import build_run_dir
from tests.support.run_dirs import wetting_config

# Importing the commands package is what registers commands on the group.
from tud_lbm.cli.commands import cli

if TYPE_CHECKING:
    from pathlib import Path

_EXIT_INTERRUPTED = 130


def _plot_names(run_dir: Path) -> set[str]:
    """Basenames of every figure produced under the run's plots directory."""
    return {p.name for p in (run_dir / "plots").rglob("*.png")}


def test_group_help_lists_both_subcommands(runner):
    result = runner.invoke(cli, ["visualise", "--help"])

    assert result.exit_code == 0
    assert "fields" in result.output
    assert "analysis" in result.output


def test_bare_visualise_builds_both_kinds(runner, run_dir):
    result = runner.invoke(cli, ["visualise", str(run_dir), "--no-prompt"])

    assert result.exit_code == 0, result.output
    assert _plot_names(run_dir)


def test_fields_subcommand_builds_only_field_snapshots(runner, run_dir):
    result = runner.invoke(cli, ["visualise", str(run_dir), "--no-prompt", "fields"])

    assert result.exit_code == 0, result.output
    assert "analysis" not in {p.parent.name for p in (run_dir / "plots").rglob("*.png")}


def test_analysis_subcommand_ignores_field_operators(runner, tmp_path):
    """A config naming both kinds must not render density panels here.

    ``plot_fields`` is ``["density", "ca_theta_vs_x"]``; restricted to the
    analysis kind only the second survives.
    """
    config = wetting_config()
    run_dir = build_run_dir(tmp_path, config=config)

    result = runner.invoke(cli, ["visualise", str(run_dir), "--no-prompt", "analysis"])

    assert result.exit_code == 0, result.output
    assert "density" not in result.output.split("Fields")[-1].split("\n")[0]


def test_run_dir_named_fields_binds_as_argument(runner, tmp_path):
    """A directory literally named ``fields`` is the RUN_DIR, not the subcommand."""
    run_dir = build_run_dir(tmp_path / "outer", config=wetting_config())
    renamed = run_dir.parent / "fields"
    run_dir.rename(renamed)

    result = runner.invoke(cli, ["visualise", str(renamed), "--no-prompt"])

    assert result.exit_code == 0, result.output
    # Bound as RUN_DIR, so figures land inside it rather than in a sibling.
    assert (renamed / "plots").exists()


@pytest.mark.parametrize("argv_tail", [[], ["fields"], ["analysis"]])
def test_subcommand_errors_still_map_to_exit_codes(runner, monkeypatch, run_dir, argv_tail):
    """The error contract must survive click's deferred subcommand dispatch."""

    def _interrupt(*_args: object, **_kwargs: object):
        raise KeyboardInterrupt

    monkeypatch.setattr("tud_lbm.cli.commands.visualise._load_run_config", _interrupt)

    result = runner.invoke(cli, ["visualise", str(run_dir), "--no-prompt", *argv_tail])

    assert result.exit_code == _EXIT_INTERRUPTED


def test_fields_flag_bypasses_the_prompt(runner, run_dir):
    """With --fields there is no prompt, so no stdin is consumed."""
    result = runner.invoke(cli, ["visualise", str(run_dir), "--fields", "density"], input="")

    assert result.exit_code == 0, result.output
    assert "Select fields" not in result.output
