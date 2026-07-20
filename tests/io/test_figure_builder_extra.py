"""Additional branch coverage for src/simulation_io/plotting/figure_builder.py.

Targets the 2 uncovered conditions (91.3 → ~100%):
- layout for n=2 and n=3 (hits `_SMALL_LAYOUTS.get(n)` for values not
  yet exercised — existing tests only hit n=1, n=4, n≥5).
- build_all with skip > 0 (the `files[skip:]` slice path).
"""

from __future__ import annotations
import matplotlib as mpl

mpl.use("Agg")

import numpy as np
from src.config import SimulationConfig
from src.simulation_io.plotting.figure_builder import FigureBuilder

# ---------------------------------------------------------------------------
# layout — missing small-layout entries
# ---------------------------------------------------------------------------


def test_layout_2():
    ncols, nrows = FigureBuilder.layout(2)
    assert (ncols, nrows) == (2, 1)


def test_layout_3():
    ncols, nrows = FigureBuilder.layout(3)
    assert (ncols, nrows) == (2, 2)


# ---------------------------------------------------------------------------
# build_all with skip > 0
# ---------------------------------------------------------------------------


def _make_run_dir(tmp_path, n_steps: int = 3):
    run_dir = tmp_path / "run"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    for step in range(1, n_steps + 1):
        np.savez(
            data_dir / f"timestep_{step}.npz",
            rho=np.ones((8, 8, 1, 1, 1)),
            u=np.zeros((8, 8, 1, 1, 2)),
        )
    return run_dir


def test_build_all_skip_zero_processes_all_files(tmp_path):
    run_dir = _make_run_dir(tmp_path, n_steps=3)
    cfg = SimulationConfig(plot_fields=["density"])
    builder = FigureBuilder(cfg, run_dir)
    saved = builder.build_all(skip=0)
    assert len(saved) == 3


def test_build_all_skip_one_processes_remaining_files(tmp_path):
    run_dir = _make_run_dir(tmp_path, n_steps=3)
    cfg = SimulationConfig(plot_fields=["density"])
    builder = FigureBuilder(cfg, run_dir)
    saved = builder.build_all(skip=1)
    # With 3 files and skip=1, only timesteps 2 and 3 should be processed.
    assert len(saved) == 2


def test_build_all_skip_exceeds_file_count_returns_empty(tmp_path):
    run_dir = _make_run_dir(tmp_path, n_steps=2)
    cfg = SimulationConfig(plot_fields=["density"])
    builder = FigureBuilder(cfg, run_dir)
    saved = builder.build_all(skip=10)
    assert saved == []


def test_build_all_no_data_dir_returns_empty(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = SimulationConfig(plot_fields=["density"])
    builder = FigureBuilder(cfg, run_dir)
    saved = builder.build_all()
    assert saved == []


def test_build_single_renders_only_requested_snapshot(tmp_path):
    run_dir = _make_run_dir(tmp_path, n_steps=3)
    cfg = SimulationConfig(plot_fields=["density"])
    builder = FigureBuilder(cfg, run_dir)

    saved = builder.build_single(run_dir / "data" / "timestep_2.npz")

    assert saved is not None
    assert saved == run_dir / "data" / "timestep_2.png"
    assert saved.exists()
    assert len(list((run_dir / "data").glob("*.png"))) == 1
