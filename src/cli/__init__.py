"""Command-line interface for TUD-LBM simulations.

Provides a CLI entry point for running LBM simulations from configuration
files with interactive prompts and rich terminal output.

Usage:
    tud-lbm config.toml
    tud-lbm config.toml --no-prompt
    tud-lbm config.toml --dry-run

Functions:
    main: CLI entry point for running simulations.
"""

from .cli import main

__all__ = ["main"]
