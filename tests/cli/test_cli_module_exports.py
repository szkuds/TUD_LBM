"""Tests for lazy CLI module exports and dispatch helpers."""

from __future__ import annotations
import importlib
from types import SimpleNamespace
import click
import pytest


@pytest.fixture
def cli_pkg():
    """Import tud_lbm.cli package for lazy-loader tests."""
    import tud_lbm.cli as module

    return module


def test_lazy_cli_proxy_calls_loaded_object(cli_pkg, monkeypatch):
    called = {"ok": False}

    class _FakeCLI:
        name = "fake"

        def __call__(self, *args, **kwargs):
            called["ok"] = True
            return "done"

    monkeypatch.setattr(cli_pkg, "_load_cli", _FakeCLI)
    proxy = cli_pkg._LazyCLI()

    assert proxy() == "done"
    assert proxy.name == "fake"
    assert called["ok"] is True


def test_module_getattr_unknown_raises_attribute_error(cli_pkg):
    with pytest.raises(AttributeError, match="has no attribute"):
        cli_pkg.missing  # noqa: B018


def test_load_cli_wraps_missing_optional_dependencies(cli_pkg, monkeypatch):
    original_import = __import__

    def _raising_import(name, *args, **kwargs):
        if name == "tud_lbm.cli.cli":
            err = ImportError("no click")
            err.name = "click"
            raise err
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raising_import)

    with pytest.raises(ImportError, match="requires 'click' and 'rich'"):
        cli_pkg._load_cli()


def test_validate_cli_args_requires_config_path():
    from tud_lbm.cli.cli import _validate_cli_args

    with pytest.raises(click.UsageError, match="--override requires CONFIG_PATH"):
        _validate_cli_args(("tau=0.7",), None)


def test_check_sweep_errors_raises_on_failed_results():
    from tud_lbm.cli.cli import _check_sweep_errors

    ok = SimpleNamespace(status="success")
    bad = SimpleNamespace(status="failed")

    with pytest.raises(RuntimeError, match="failed simulation"):
        _check_sweep_errors([ok, bad])


def test_main_dispatches_help_to_click_group(monkeypatch):
    cli_module = importlib.import_module("tud_lbm.cli.cli")

    calls: list[list[str]] = []
    monkeypatch.setattr(cli_module.cli, "main", lambda args, standalone_mode: calls.append(args))

    cli_module.main.callback(("--help",))
    assert calls == [["--help"]]


def test_main_dispatch_strips_run_token(monkeypatch):
    cli_module = importlib.import_module("tud_lbm.cli.cli")

    calls: list[list[str]] = []
    monkeypatch.setattr(cli_module.run, "main", lambda args, standalone_mode: calls.append(args))

    cli_module.main.callback(("run", "config.toml", "--dry-run"))
    assert calls == [["config.toml", "--dry-run"]]
