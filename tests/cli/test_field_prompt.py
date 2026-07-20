"""Operator prompts mark which entries the run's stored config already lists."""

from __future__ import annotations
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
    target = sorted(available)[-1]

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
