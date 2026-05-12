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


def __getattr__(name: str):  # noqa: ANN202
    """Lazy-load CLI only when accessed (requires click)."""
    if name != "cli":
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    try:
        from tud_lbm.cli.cli import cli

        return cli  # noqa: TRY300
    except ImportError as e:
        if "click" not in str(e):
            raise
        msg = "The CLI requires 'click' to be installed. Install with: pip install click"
        raise ImportError(msg) from e
