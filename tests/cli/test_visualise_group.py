"""The ``visualise`` group: ``fields``, ``analysis`` and ``setup`` for one run.

Import ``src.cli.commands`` rather than ``src.cli.app``: importing the commands
package is what registers the commands onto the group.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pytest
from src.cli.commands import cli
from tests.support.run_dirs import build_run_dir
from tests.support.run_dirs import wetting

if TYPE_CHECKING:
    from pathlib import Path

_EXIT_INTERRUPTED = 130
_EXIT_USAGE = 2


def _plot_names(run_dir: Path) -> set[str]:
    """Basenames of every figure produced under the run's plots directory."""
    return {p.name for p in (run_dir / "plots").rglob("*.png")}


def _write_snapshot(path: Path) -> Path:
    np.savez(path, rho=np.ones((8, 8, 1, 1, 1)), u=np.zeros((8, 8, 1, 1, 2)))
    return path


def test_group_help_lists_every_product(runner):
    result = runner.invoke(cli, ["visualise", "--help"])

    assert result.exit_code == 0
    for name in ("fields", "analysis", "setup"):
        assert name in result.output


def test_bare_visualise_with_a_run_dir_is_a_usage_error(runner, run_dir):
    """There is no default product: a run directory is not a subcommand."""
    result = runner.invoke(cli, ["visualise", str(run_dir)])

    assert result.exit_code == _EXIT_USAGE


def test_fields_builds_only_field_snapshots(runner, run_dir):
    result = runner.invoke(cli, ["visualise", "fields", str(run_dir), "--no-prompt"])

    assert result.exit_code == 0, result.output
    assert _plot_names(run_dir)
    assert "analysis" not in {p.parent.name for p in (run_dir / "plots").rglob("*.png")}


def test_analysis_ignores_field_operators_and_writes_the_csv(runner, tmp_path):
    """A config naming both kinds must not render density panels here.

    ``plot_fields`` is ``["density", "ca_theta_vs_x"]``; restricted to the
    analysis kind only the second survives.
    """
    run_dir = build_run_dir(tmp_path, config=wetting())

    result = runner.invoke(cli, ["visualise", "analysis", str(run_dir), "--no-prompt"])

    assert result.exit_code == 0, result.output
    assert "Fields        : ca_theta_vs_x" in result.output
    assert "snapshots" not in {p.parent.name for p in (run_dir / "plots").rglob("*.png")}
    assert (run_dir / "simulation_data.csv").exists()


def test_fields_option_restricts_to_the_kind(runner, run_dir):
    """An analysis operator named to ``fields`` is dropped, not rendered."""
    result = runner.invoke(cli, ["visualise", "fields", str(run_dir), "--fields", "density,ca_theta_vs_x"])

    assert result.exit_code == 0, result.output
    assert "Fields        : density\n" in result.output


@pytest.mark.parametrize("product", ["fields", "analysis", "setup"])
def test_errors_map_to_exit_codes(runner, monkeypatch, run_dir, product):
    def _interrupt(*_args: object, **_kwargs: object):
        raise KeyboardInterrupt

    monkeypatch.setattr("src.cli.commands.visualise._load_run_config", _interrupt)

    result = runner.invoke(cli, ["visualise", product, str(run_dir)])

    assert result.exit_code == _EXIT_INTERRUPTED


def test_fields_flag_bypasses_the_prompt(runner, run_dir):
    """With --fields there is no prompt, so no stdin is consumed."""
    result = runner.invoke(cli, ["visualise", "fields", str(run_dir), "--fields", "density"], input="")

    assert result.exit_code == 0, result.output
    assert "Select fields" not in result.output


def test_snapshot_target_builds_visual_in_plots_tree(runner, tmp_path):
    snapshot = _write_snapshot(tmp_path / "radius_100.25_init.npz")

    result = runner.invoke(cli, ["visualise", "fields", str(snapshot)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "plots" / "snapshots" / "radius_100.25_init.png").exists()
    assert list(tmp_path.glob("*.png")) == []


def test_snapshot_in_data_dir_writes_to_run_plots(runner, tmp_path):
    """A snapshot under ``<run>/data/`` belongs to that run's plots tree."""
    data_dir = tmp_path / "run" / "data"
    data_dir.mkdir(parents=True)
    snapshot = _write_snapshot(data_dir / "timestep_10.npz")

    result = runner.invoke(cli, ["visualise", "fields", str(snapshot)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "run" / "plots" / "snapshots" / "timestep_10.png").exists()
    assert list(data_dir.glob("*.png")) == []
    assert not (data_dir / "plots").exists()


def test_fields_rejects_a_file_that_is_not_a_snapshot(runner, tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("", encoding="utf-8")

    result = runner.invoke(cli, ["visualise", "fields", str(other)])

    assert result.exit_code == _EXIT_USAGE
    assert "run directory or a .npz snapshot" in result.output


def test_snapshot_without_a_grid_field_is_a_usage_error(runner, tmp_path):
    snapshot = tmp_path / "snapshot.npz"
    np.savez(snapshot, t=np.array(3))

    result = runner.invoke(cli, ["visualise", "fields", str(snapshot)])

    assert result.exit_code == _EXIT_USAGE
    assert "'rho' or 'u'" in result.output


def test_analysis_rejects_a_file(runner, tmp_path):
    snapshot = _write_snapshot(tmp_path / "snapshot.npz")

    result = runner.invoke(cli, ["visualise", "analysis", str(snapshot)])

    assert result.exit_code == _EXIT_USAGE
    assert "Directory" in result.output


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--output", "a.mp4"], "only with --animate"),
        (["--fps", "5"], "only with --animate"),
        (["--animate", "--skip", "2"], "--skip does not apply"),
    ],
)
def test_animation_options_are_checked(runner, run_dir, extra, message):
    result = runner.invoke(cli, ["visualise", "fields", str(run_dir), "--no-prompt", *extra])

    assert result.exit_code == _EXIT_USAGE
    assert message in result.output


def test_animate_rejects_a_snapshot(runner, tmp_path):
    snapshot = _write_snapshot(tmp_path / "snapshot.npz")

    result = runner.invoke(cli, ["visualise", "fields", str(snapshot), "--animate"])

    assert result.exit_code == _EXIT_USAGE
    assert "not a single snapshot" in result.output


def test_fields_applies_interface_overlay(runner, run_dir):
    result = runner.invoke(
        cli,
        [
            "visualise",
            "fields",
            str(run_dir),
            "--fields",
            "density",
            "--overlay",
            "interface",
            "--interface-levels",
            "measured",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Overlays      : interface" in result.output
    assert "timestep_5.png" in _plot_names(run_dir)


def test_unknown_interface_level_exits_with_error(runner, run_dir):
    result = runner.invoke(
        cli,
        ["visualise", "fields", str(run_dir), "--fields", "interface", "--interface-levels", "median"],
    )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "median" in result.output


class _FakeAnimator:
    """Records what ``visualise fields --animate`` hands the animator."""

    captured: dict[str, object] = {}  # noqa: RUF012

    def __init__(self, *, config, run_dir, fps, dpi, fields, overlays):
        type(self).captured = {
            "levels": config.interface_levels,
            "fps": fps,
            "dpi": dpi,
            "fields": fields,
            "overlays": overlays,
        }
        self._run_dir = run_dir

    def create(self, output):
        return self._run_dir / "plots" / (output or "animation.mp4")


def test_animate_passes_selection_to_the_animator(runner, run_dir, monkeypatch):
    monkeypatch.setattr("src.simulation_io.plotting.Animator", _FakeAnimator)

    result = runner.invoke(
        cli,
        [
            "visualise",
            "fields",
            str(run_dir),
            "--animate",
            "--fields",
            "density,ca_theta_vs_x",
            "--overlay",
            "interface",
            "--interface-levels",
            "config",
            "--fps",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    # An animation frame may hold analysis panels too.
    assert _FakeAnimator.captured == {
        "levels": ["config"],
        "fps": 4,
        "dpi": 150,
        "fields": ["density", "ca_theta_vs_x"],
        "overlays": ["interface"],
    }
    assert "Animation saved to" in result.output


def test_animate_defaults_to_ten_fps(runner, run_dir, monkeypatch):
    monkeypatch.setattr("src.simulation_io.plotting.Animator", _FakeAnimator)

    result = runner.invoke(cli, ["visualise", "fields", str(run_dir), "--animate", "--no-prompt"])

    assert result.exit_code == 0, result.output
    assert _FakeAnimator.captured["fps"] == 10
