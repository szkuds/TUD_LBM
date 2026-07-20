"""Click command modules.

Importing this package registers every command on the ``cli`` group. It is the
entry point declared in ``pyproject.toml``.
"""

from __future__ import annotations
from tud_lbm.cli.app import cli

# Imported for their side effect: each module registers commands on ``cli``.
from tud_lbm.cli.commands import analysis as analysis
from tud_lbm.cli.commands import run as run
from tud_lbm.cli.commands import visualise as visualise

__all__ = ["analysis", "cli", "run", "visualise"]
