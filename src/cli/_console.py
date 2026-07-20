"""Shared rich console, banners, and the standard CLI exit-code contract."""

from __future__ import annotations
import functools
import os
import sys
from typing import TYPE_CHECKING
from typing import ParamSpec
from typing import TypeVar
import click
from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from collections.abc import Callable

console = Console()

_CLI_SUBTITLE = "Delft University of Technology"

_P = ParamSpec("_P")
_R = TypeVar("_R")

#: Exit code for user interruption, matching the shell's 128 + SIGINT.
_EXIT_INTERRUPTED = 130


def banner(title: str) -> None:
    """Print the TUD-LBM banner panel with a command-specific *title*."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]TUD-LBM[/bold blue] - {title}",
            subtitle=_CLI_SUBTITLE,
        ),
    )
    console.print()


def success(message: str, *, title: str = "Success") -> None:
    """Print a green success panel."""
    console.print()
    console.print(Panel.fit(f"[bold green]{message}[/bold green]", title=title))


def cli_command(*, title: str, interrupt_message: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Print the banner, then map exceptions onto the standard CLI exit codes.

    ``KeyboardInterrupt`` exits 130. Any other exception prints a red error line
    and exits 1, unless ``TUD_LBM_DEBUG`` is set, in which case it is re-raised
    so the traceback survives.

    ``click.UsageError`` and ``SystemExit`` propagate untouched: click renders
    usage errors itself (exit code 2), and commands that call ``sys.exit``
    directly must keep the code they chose.
    """

    def decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            banner(title)
            try:
                return fn(*args, **kwargs)
            except KeyboardInterrupt:
                console.print(f"\n[yellow]{interrupt_message}[/yellow]")
                sys.exit(_EXIT_INTERRUPTED)
            except (click.UsageError, click.exceptions.Exit, SystemExit):
                raise
            except Exception as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")
                if os.environ.get("TUD_LBM_DEBUG"):
                    raise
                sys.exit(1)

        return wrapper

    return decorator
