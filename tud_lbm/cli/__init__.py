"""Command-line interface for TUD-LBM simulations.

Provides a CLI entry point for running LBM simulations from configuration
files with interactive prompts and rich terminal output.

Usage:
    tud-lbm app_setup.toml
    tud-lbm app_setup.toml --no-prompt
    tud-lbm app_setup.toml --dry-run

Functions:
    main: CLI entry point for running simulations (requires click).
"""


def __getattr__(name):
    """Lazy-load CLI only when accessed (requires click)."""
    if name == "main":
        try:
            from .cli import main
            return main
        except ImportError as e:
            if "click" in str(e):
                raise ImportError(
                    "The CLI requires 'click' to be installed. "
                    "Install with: pip install click"
                ) from e
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main"]
