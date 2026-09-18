"""Operator prompts mark which entries the run's stored config already lists."""

from __future__ import annotations
import pytest
from src.cli.commands import cli
from src.cli.field_select import build_choices
from tests.support.run_dirs import build_run_dir
from tests.support.run_dirs import wetting_config


def _entries(kind: str) -> dict:
    """Registered operators of *kind*, after import-time registration."""
    import src.simulation_io.plotting  # noqa: F401  registers the plotting/analysis operators
    from src.registry import get_operators

    return get_operators(kind)


def test_choices_mark_configured_operators():
    available = _entries("plotting")
    configured = ["density"]

    choices = build_choices(available, configured)

    by_name = {c.name: c for c in choices}
    assert by_name["density"].in_config is True
    assert all(not c.in_config for c in choices if c.name != "density")


def test_configured_operators_are_listed_first():
    available = _entries("plotting")
    target = max(available)

    choices = build_choices(available, [target])

    assert choices[0].name == target


def test_choices_carry_kind_and_description():
    choices = build_choices(_entries("analysis"), [])

    assert choices
    assert all(c.kind == "analysis" for c in choices)
    assert all(c.description for c in choices)


def test_visualise_prompt_shows_config_status(runner, run_dir):
    """The prompt marks in-config vs not, and footers the count."""
    result = runner.invoke(cli, ["visualise", str(run_dir)], input="\n")

    assert result.exit_code == 0, result.output
    assert "in config" in result.output
    assert "not in config" in result.output
    assert "not listed in this run's config.toml" in result.output


def test_animate_marks_against_animate_fields(runner, tmp_path, monkeypatch):
    """Animate reads animate_fields; visualise reads plot_fields."""
    captured: dict[str, object] = {}

    def _capture(available, current, *, configured, label, config_label):
        captured["configured"] = list(configured)
        return ["density"]

    monkeypatch.setattr("src.cli.commands.visualise.prompt_fields_marked", _capture)
    monkeypatch.setattr("src.simulation_io.plotting.Animator.create", lambda self, output: tmp_path / "a.mp4")

    config = wetting_config(animate_fields=["velocity"], plot_fields=["density"])
    run_dir = build_run_dir(tmp_path, config=config)

    result = runner.invoke(cli, ["animate", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert captured["configured"] == ["velocity"]


_OVERLAY_QUESTION = "Overlay interface contour on field panels?"


def test_prompt_overlays_is_opt_out(monkeypatch):
    from src.cli import field_select

    answers = iter([True, False])
    monkeypatch.setattr(field_select.Confirm, "ask", lambda *_a, **_k: next(answers))
    plotting = _entries("plotting")

    assert field_select.prompt_overlays(plotting) == ["interface"]
    assert field_select.prompt_overlays(plotting) == []


def test_prompt_overlays_keeps_default_on_end_of_input(monkeypatch):
    from src.cli import field_select

    def _eof(*_args, **_kwargs):
        raise EOFError

    monkeypatch.setattr(field_select.Confirm, "ask", _eof)

    assert field_select.prompt_overlays(_entries("plotting")) == ["interface"]


def test_interactive_visualise_overlays_interface_by_default(runner, run_dir):
    result = runner.invoke(cli, ["visualise", str(run_dir), "fields"], input="\n\n")

    assert result.exit_code == 0, result.output
    assert _OVERLAY_QUESTION in result.output
    assert "Overlays      : interface" in result.output


def test_interactive_visualise_overlay_can_be_declined(runner, run_dir):
    result = runner.invoke(cli, ["visualise", str(run_dir), "fields"], input="\nn\n")

    assert result.exit_code == 0, result.output
    assert _OVERLAY_QUESTION in result.output
    assert "Overlays      : none" in result.output


def test_declining_overrides_overlay_fields_from_config(runner, tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def _capture(self, config, run_dir, dpi, fields, overlays):
        captured["overlays"] = overlays
        raise KeyboardInterrupt

    monkeypatch.setattr("src.simulation_io.plotting.FigureBuilder.__init__", _capture)
    run_dir = build_run_dir(tmp_path, config=wetting_config(overlay_fields=["interface"]))

    runner.invoke(cli, ["visualise", str(run_dir), "fields"], input="\nn\n")

    assert captured["overlays"] == []


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["analysis"], id="analysis-subcommand"),
        pytest.param(["--overlay", "interface", "fields"], id="explicit-overlay"),
        pytest.param(["--no-prompt", "fields"], id="no-prompt"),
        pytest.param(["--fields", "density", "fields"], id="explicit-fields"),
    ],
)
def test_overlay_question_is_skipped(runner, run_dir, args):
    result = runner.invoke(cli, ["visualise", str(run_dir), *args], input="\n\n")

    assert result.exit_code == 0, result.output
    assert _OVERLAY_QUESTION not in result.output


def test_analysis_only_selection_skips_overlay_question(runner, run_dir):
    result = runner.invoke(cli, ["visualise", str(run_dir)], input="ca_theta_vs_x\n\n")

    assert result.exit_code == 0, result.output
    assert _OVERLAY_QUESTION not in result.output


@pytest.mark.parametrize(("answer", "expected"), [("\n", ["interface"]), ("n\n", [])])
def test_interactive_animate_asks_overlay_question(runner, tmp_path, monkeypatch, answer, expected):
    captured: dict[str, object] = {}

    class _FakeAnimator:
        def __init__(self, *, config, run_dir, fps, fields, overlays):
            captured["overlays"] = overlays

        def create(self, output):
            return tmp_path / "a.mp4"

    monkeypatch.setattr("src.simulation_io.plotting.Animator", _FakeAnimator)
    run_dir = build_run_dir(tmp_path)

    result = runner.invoke(cli, ["animate", str(run_dir)], input="\n" + answer)

    assert result.exit_code == 0, result.output
    assert _OVERLAY_QUESTION in result.output
    assert captured["overlays"] == expected
