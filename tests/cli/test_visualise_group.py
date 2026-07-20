"""The ``visualise`` group and its ``fields`` / ``analysis`` subcommands.

Two click behaviours are load-bearing here and pinned first:

1. ``invoke_without_command=True`` with a required positional on the *group* —
   click binds the argument before dispatching, so ``visualise ./out fields``
   means ``RUN_DIR=./out`` plus subcommand ``fields``.
2. ``cli_command`` cannot wrap the group callback, because click invokes
   subcommands only after that callback returns. The work functions carry it.

Import ``src.cli.commands`` rather than ``src.cli.app``: importing the commands
package is what registers the commands onto the group.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pytest
from src.cli.commands import cli
from tests.support.run_dirs import build_run_dir
from tests.support.run_dirs import wetting_config

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

    monkeypatch.setattr("src.cli.commands.visualise._load_run_config", _interrupt)

    result = runner.invoke(cli, ["visualise", str(run_dir), "--no-prompt", *argv_tail])

    assert result.exit_code == _EXIT_INTERRUPTED


def test_fields_flag_bypasses_the_prompt(runner, run_dir):
    """With --fields there is no prompt, so no stdin is consumed."""
    result = runner.invoke(cli, ["visualise", str(run_dir), "--fields", "density"], input="")

    assert result.exit_code == 0, result.output
    assert "Select fields" not in result.output


def test_single_snapshot_builds_visual_beside_standalone_npz(runner, tmp_path):
    snapshot = tmp_path / "radius_100.25_init.npz"
    np.savez(
        snapshot,
        rho=np.ones((8, 8, 1, 1, 1)),
        u=np.zeros((8, 8, 1, 1, 2)),
    )

    result = runner.invoke(cli, ["visualise", str(snapshot), "--single", "--no-prompt"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "radius_100.25_init.png").exists()
    assert len(list(tmp_path.glob("*.png"))) == 1


def test_single_snapshot_rejects_non_npz_path(runner, run_dir):
    result = runner.invoke(cli, ["visualise", str(run_dir), "--single", "--no-prompt"])

    assert result.exit_code == 2
    assert "existing .npz snapshot file" in result.output


def test_visualise_rejects_file_without_single(runner, tmp_path):
    snapshot = tmp_path / "snapshot.npz"
    np.savez(snapshot, rho=np.ones((8, 8)))

    result = runner.invoke(cli, ["visualise", str(snapshot), "--no-prompt"])

    assert result.exit_code == 2
    assert "PATH must point to an existing run directory unless --single is provided." in result.output


def test_animate_rejects_file_run_dir(runner, tmp_path):
    path_arg = tmp_path / "not-a-run-directory"
    path_arg.touch()

    result = runner.invoke(cli, ["animate", str(path_arg), "--no-prompt"])

    assert result.exit_code == 2
    assert "Directory" in result.output
