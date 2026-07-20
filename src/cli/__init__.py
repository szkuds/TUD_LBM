"""Command-line interface for TUD-LBM simulations.

Provides a CLI entry point for running LBM simulations from configuration
files with interactive prompts and rich terminal output.

Usage:
    tud-lbm run app_setup.toml
    tud-lbm run app_setup.toml --no-prompt
    tud-lbm run app_setup.toml --dry-run

Functions:
    cli: CLI group entry point (requires click).
"""

from __future__ import annotations
from typing import Any


def _load_cli() -> Any:  # noqa: ANN401
    """Import and return the click CLI command object."""
    try:
        from src.cli.commands import cli as click_cli
    except ImportError as e:
        if getattr(e, "name", None) not in {"click", "rich"}:
            raise
        msg = "The CLI requires 'click' and 'rich'. Install with: pip install click rich"
        raise ImportError(msg) from e
    else:
        return click_cli


class _LazyCLI:
    """Callable proxy that lazily loads the real click command object."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        return _load_cli()(*args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(_load_cli(), name)


cli = _LazyCLI()


def __getattr__(name: str):  # noqa: ANN202
    """Lazy-load CLI only when accessed (requires click)."""
    if name == "cli":
        return cli
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
