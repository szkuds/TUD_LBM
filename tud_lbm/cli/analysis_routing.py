"""Routing between the CLI's analysis entry points and the metric layer.

Every command that analyses saved snapshots — ``compare``, ``run --compare``,
and ``regime-map`` — goes through :func:`analyse_run` or :func:`analyse_tree`.
Both share one :class:`~tud_lbm.io.analysis.droplet_metrics.DropletSeries` per
run, so a run's ``.npz`` files are read once regardless of how many analysis
figures are requested.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
from tud_lbm.io.plotting.run_comparison import _COMPARISON_DIR
from tud_lbm.io.plotting.run_comparison import _CONFIG_TOML
from tud_lbm.io.plotting.run_comparison import _safe_load_config
from tud_lbm.io.plotting.run_comparison import compare_runs
from tud_lbm.io.plotting.simulation_csv import build_simulation_csv

if TYPE_CHECKING:
    from tud_lbm.config import SimulationConfig

#: Path fragments marking directories that are not simulation runs.
_SKIP_DIRS = ("init", _COMPARISON_DIR)


def analyse_run(
    run_dir: str | Path,
    config: SimulationConfig,
    *,
    fields: list[str] | None = None,
) -> Path | None:
    """Export ``simulation_data.csv`` and per-run analysis figures for one run.

    Args:
        run_dir: A single run directory containing ``data/timestep_*.npz``.
        config: The run's configuration, passed in memory rather than re-read
            from ``config.toml`` so expanded sweep fields survive.
        fields: Analysis operator names to render. ``None`` exports the CSV only.

    Returns:
        The CSV path, or ``None`` when the run has no analysable data.
    """
    from tud_lbm.io.plotting.figure_builder import FigureBuilder

    run_dir = Path(run_dir)
    csv_path = build_simulation_csv(run_dir, config)
    if fields:
        # Reads from the same cached series the CSV export just populated.
        FigureBuilder(config=config, run_dir=run_dir, fields=fields).build_analysis()
    return csv_path


def find_run_dirs(parent_dir: str | Path) -> list[Path]:
    """Run directories under *parent_dir*, identified by a ``config.toml``.

    Directories whose path mentions ``init`` or the comparison output folder
    are not runs and are skipped.
    """
    parent = Path(parent_dir)
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for toml in sorted(parent.rglob(_CONFIG_TOML)):
        candidate = toml.parent
        if candidate in seen or any(skip in str(candidate).lower() for skip in _SKIP_DIRS):
            continue
        seen.add(candidate)
        run_dirs.append(candidate)
    return run_dirs


def analyse_tree(
    parent_dir: str | Path,
    *,
    fields: list[str] | None = None,
) -> tuple[int, int]:
    """Analyse every run under *parent_dir*, then build cross-run plots.

    Cross-run comparison plots are only generated when at least one run
    produced CSV data.

    Args:
        parent_dir: Directory containing run directories, at any nesting depth.
        fields: Analysis operator names to render per run.

    Returns:
        ``(n_runs_found, n_runs_with_csv)``.
    """
    parent = Path(parent_dir)
    run_dirs = find_run_dirs(parent)
    if not run_dirs:
        return 0, 0

    print(f"Found {len(run_dirs)} simulation(s) to process.")

    n_ok = 0
    for run_dir in run_dirs:
        config = _safe_load_config(run_dir / _CONFIG_TOML)
        if config is None:
            continue
        if analyse_run(run_dir, config, fields=fields) is not None:
            n_ok += 1

    if n_ok > 0:
        print("\nGenerating comparison plots...")
        compare_runs(parent)
        print(f"Done. Comparison plots in {parent / _COMPARISON_DIR}")

    return len(run_dirs), n_ok
