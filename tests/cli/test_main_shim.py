"""Behavioural tests for the legacy ``...:main`` launcher shim.

The shim exists for wrappers still targeting ``tud_lbm.cli.cli:main``. It must
forward every real subcommand to the command group, and only fall through to
``run`` for the legacy ``tud-lbm CONFIG.toml`` form.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from tud_lbm.cli.cli import cli
from tud_lbm.cli.cli import main

if TYPE_CHECKING:
    from click.testing import CliRunner

#: Every subcommand except ``run``, which the shim handles by a separate branch.
_FORWARDED = sorted(set(cli.commands) - {"run"})


@pytest.mark.parametrize("name", _FORWARDED)
def test_main_forwards_subcommand_help_to_group(runner: CliRunner, name: str) -> None:
    """``main <subcommand> --help`` reaches that subcommand, not ``run``."""
    result = runner.invoke(main, [name, "--help"])

    assert result.exit_code == 0, result.output
    assert name in result.output
    # "run"'s help mentions CONFIG_PATH; reaching it means the shim misrouted.
    assert "CONFIG_PATH" not in result.output


def test_main_forwards_group_help(runner: CliRunner) -> None:
    """``main --help`` renders the group help listing every subcommand."""
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    for name in cli.commands:
        assert name in result.output


@pytest.fixture
def spy_run_impl(monkeypatch) -> dict[str, object]:
    """Capture the config path ``_run_impl`` is called with."""
    seen: dict[str, object] = {}

    def fake_run_impl(config_path, _overrides, _max_workers, _init_dir, _flags) -> bool:
        seen["config_path"] = config_path
        return False

    monkeypatch.setattr("tud_lbm.cli.cli._run_impl", fake_run_impl)
    return seen


def test_main_strips_explicit_run_token(runner: CliRunner, tmp_path, spy_run_impl) -> None:
    """``main run CONFIG`` drops the token and invokes ``run`` with CONFIG."""
    config = tmp_path / "some_config.toml"
    config.write_text("")

    result = runner.invoke(main, ["run", str(config)])

    assert result.exit_code == 0, result.output
    assert spy_run_impl["config_path"] == str(config)


def test_main_treats_bare_path_as_config(runner: CliRunner, tmp_path, spy_run_impl) -> None:
    """The legacy ``tud-lbm CONFIG.toml`` form still routes to ``run``."""
    config = tmp_path / "some_config.toml"
    config.write_text("")

    result = runner.invoke(main, [str(config)])

    assert result.exit_code == 0, result.output
    assert spy_run_impl["config_path"] == str(config)
