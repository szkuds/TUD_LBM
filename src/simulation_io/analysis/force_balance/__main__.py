"""``python -m src.simulation_io.analysis.force_balance CONFIG [-g GROUP ...] [--out DIR]``.

The same as ``tud-lbm force-balance``.
"""

from __future__ import annotations
import sys
from src.cli.commands import cli

if __name__ == "__main__":
    cli(["force-balance", *sys.argv[1:]], prog_name="python -m src.simulation_io.analysis.force_balance")
