"""The root ``tud-lbm`` click group.

Imports nothing from ``commands`` so the command modules can import it.
"""

from __future__ import annotations
import click


@click.group()
@click.version_option(package_name="src")
def cli() -> None:
    """TUD-LBM - Lattice Boltzmann Method Solver."""
