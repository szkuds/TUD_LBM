"""Characterization tests pinning ``simulation_data.csv`` output.

These guard the droplet-metric refactor: the CSV that ``build_simulation_csv``
produces must not drift except where a change is deliberate and documented.
"""

from __future__ import annotations
from itertools import pairwise
from pathlib import Path
import pandas as pd
import pytest
from src.simulation_io.plotting.simulation_csv import build_simulation_csv
from tests.support.run_dirs import NONUNIFORM_ITERATIONS
from tests.support.run_dirs import SAVE_INTERVAL
from tests.support.run_dirs import UNIFORM_ITERATIONS
from tests.support.run_dirs import build_run_dir
from tests.support.run_dirs import wetting_config

_FIXTURES = Path(__file__).parent / "fixtures"

_GOLDEN = {
    "uniform": UNIFORM_ITERATIONS,
    "nonuniform": NONUNIFORM_ITERATIONS,
}


@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_simulation_csv_matches_golden(tmp_path: Path, name: str) -> None:
    """The CSV matches its committed golden fixture exactly."""
    run_dir = build_run_dir(tmp_path, iterations=_GOLDEN[name])

    out = build_simulation_csv(run_dir, wetting_config())

    assert out is not None
    produced = pd.read_csv(out)
    expected = pd.read_csv(_FIXTURES / f"simulation_data_golden_{name}.csv")
    pd.testing.assert_frame_equal(produced, expected)


def test_uniform_run_gaps_equal_save_interval() -> None:
    """The uniform fixture's gaps match save_interval.

    Differentiating by nominal interval and by actual gap must agree here, so
    this fixture stays byte-identical across the backward-diff change.
    """
    gaps = {b - a for a, b in pairwise(UNIFORM_ITERATIONS)}

    assert gaps == {SAVE_INTERVAL}


def test_nonuniform_run_gaps_differ_from_save_interval() -> None:
    """The non-uniform fixture's gaps diverge from save_interval.

    This is the fixture that distinguishes the two differentiation schemes.
    """
    gaps = {b - a for a, b in pairwise(NONUNIFORM_ITERATIONS)}

    assert gaps != {SAVE_INTERVAL}
    assert any(gap != SAVE_INTERVAL for gap in gaps)
