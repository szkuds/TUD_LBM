"""Click command modules.

Importing this package registers every command on the ``cli`` group. It is the
entry point declared in ``pyproject.toml``.
"""

from __future__ import annotations
from src.cli.app import cli

# Imported for their side effect: each module registers commands on ``cli``.
from src.cli.commands import analysis as analysis
from src.cli.commands import benchmark as benchmark
from src.cli.commands import run as run
from src.cli.commands import visualise as visualise

__all__ = ["analysis", "benchmark", "cli", "run", "visualise"]
