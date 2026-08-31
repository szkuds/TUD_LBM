"""Run-directory layout — the single source for output directory and file names.

Every artefact written into (or read back out of) a run directory is addressed
through the names below, so the on-disk layout is described in one place
instead of by string literals repeated across the IO, plotting, analysis and
CLI layers::

    <run_dir>/
    |-- config.toml                     CONFIG_FILENAME
    |-- physical_parameters.txt
    |-- simulation_data.csv             SIMULATION_CSV_FILENAME
    |-- data/                           DATA_DIRNAME
    |   `-- timestep_<N>.npz            SNAPSHOT_PREFIX / SNAPSHOT_GLOB
    `-- plots/                          PLOTS_DIRNAME
        |-- snapshots/                  SNAPSHOTS_DIRNAME
        `-- analysis/                   ANALYSIS_DIRNAME
            `-- acceleration_analysis.png   ACCELERATION_PLOT_FILENAME

Aggregate analyses spanning several runs write into their own directory rather
than into any one run: ``COMPARISON_DIRNAME`` for ``tud-lbm compare`` and
``REGIME_MAP_DIRNAME`` for ``tud-lbm regime-map``.

These are names, not paths — join them onto a run directory at the call site.
The results *root* under which run directories are created is
:data:`src.config.config_overview.BASE_RESULTS_DIR`.
"""

from __future__ import annotations

# --- Per-run directory layout ------------------------------------------------

#: Config snapshot written beside a run's output; also what identifies a
#: directory as a run directory.
CONFIG_FILENAME = "config.toml"

#: Saved field snapshots (``.npz``/VTK), relative to the run directory.
DATA_DIRNAME = "data"

#: Figures, relative to the run directory.
PLOTS_DIRNAME = "plots"

#: Per-timestep field figures, relative to ``PLOTS_DIRNAME``.
SNAPSHOTS_DIRNAME = "snapshots"

#: Snapshot-history (``analysis`` operator) figures, relative to ``PLOTS_DIRNAME``.
ANALYSIS_DIRNAME = "analysis"

# --- Snapshot file naming ----------------------------------------------------

#: Snapshot files are ``f"{SNAPSHOT_PREFIX}{iteration}"`` plus the writer's
#: extension; the trailing integer is parsed back out when snapshots are read
#: in timestep order, so the prefix, the glob and the parser must agree.
SNAPSHOT_PREFIX = "timestep_"

#: Glob matching every NumPy snapshot inside ``DATA_DIRNAME``.
SNAPSHOT_GLOB = f"{SNAPSHOT_PREFIX}*.npz"

# --- Per-run artefact filenames ----------------------------------------------

#: Per-timestep droplet-metric export at the run-directory root.
SIMULATION_CSV_FILENAME = "simulation_data.csv"

#: Acceleration/regime diagnostic figure, written under ``plots/analysis/``.
ACCELERATION_PLOT_FILENAME = "acceleration_analysis.png"

# --- Multi-run analysis output -----------------------------------------------

#: Output directory for ``tud-lbm compare``.
COMPARISON_DIRNAME = "comparison_analysis"

#: Output directory and figure name for ``tud-lbm regime-map``.
REGIME_MAP_DIRNAME = "regime_map_analysis"
REGIME_MAP_FILENAME = "regime_map.png"
