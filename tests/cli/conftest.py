"""Fixtures for CLI behavioural tests."""

from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from click.testing import CliRunner
from tests.support.run_dirs import build_run_dir
from tests.support.run_dirs import wetting_config

if TYPE_CHECKING:
    from pathlib import Path
    from tud_lbm.config import SimulationConfig


@pytest.fixture
def runner() -> CliRunner:
    """A click test runner."""
    return CliRunner()


@pytest.fixture
def run_config() -> SimulationConfig:
    """The config serialised into the :func:`run_dir` fixture."""
    return wetting_config()


@pytest.fixture
def run_dir(tmp_path: Path, run_config: SimulationConfig) -> Path:
    """A synthetic run directory with snapshots and a ``config.toml``.

    ``plot_fields`` is ``["density", "ca_theta_vs_x"]`` — one field operator and
    one analysis operator — so tests can distinguish the two kinds and check
    which operators are marked as present in the stored config.
    """
    return build_run_dir(tmp_path, config=run_config)
