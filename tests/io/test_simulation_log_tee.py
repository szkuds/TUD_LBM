"""The ``simulation.log`` tee must survive a console that cannot encode the output.

A redirected Windows console is cp1252; the run overview prints Greek. The
console may degrade, the log file may not, and the run must never raise.
"""

from __future__ import annotations
import io
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import pytest
from src.simulation_io.save import SimulationIO

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def cp1252_console(monkeypatch) -> Iterator[io.BytesIO]:
    """Stand in a cp1252 console for ``sys.__stdout__`` and undo the tee afterwards."""
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252", write_through=True)
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    monkeypatch.setattr(sys, "__stdout__", console)
    monkeypatch.setattr(sys, "__stderr__", console)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    yield raw
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


def test_tee_degrades_on_the_console_and_keeps_the_log_exact(tmp_path, cp1252_console):
    io_handler = SimulationIO(base_dir=str(tmp_path))

    sys.stdout.write("σ = 0.02\n")
    sys.stdout.flush()

    assert cp1252_console.getvalue().decode("cp1252") == "? = 0.02\n"
    log = (Path(io_handler.run_dir) / "simulation.log").read_text(encoding="utf-8")
    assert "σ = 0.02" in log
