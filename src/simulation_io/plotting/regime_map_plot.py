"""Aggregate Ca(t) regime classification across many run directories.

Reads `simulation_data.csv` (building it on demand if missing) for every run
directory it is given, classifies each run into
pinning/dissipative/capillary/steady/unknown (see
:mod:`src.simulation_io.analysis.accelerations.regime_classification`), and plots the
result against any pair of registered dimensionless numbers — the regime map.

Classification is axis-agnostic: :func:`process_run_dir` keeps every number a run
could resolve, and the axis pair is applied only when plotting. Defaults are
``Bo_parallel`` (x) against ``Oh`` (y), the pair this map was originally written
for.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from src.config.analysis_options import AxisScale
from src.config.analysis_options import Smoothing
from src.config.run_config import ACCELERATION_PLOT_FILENAME
from src.config.run_config import ANALYSIS_DIRNAME
from src.config.run_config import CONFIG_FILENAME
from src.config.run_config import PLOTS_DIRNAME
from src.config.run_config import regime_map_filename
from src.simulation_io.analysis.accelerations import classify_regime
from src.simulation_io.analysis.accelerations import compute_acceleration
from src.simulation_io.analysis.accelerations import save_diagnostic_plot
from src.simulation_io.analysis.droplet_metrics import droplet_series_for_run
from src.simulation_io.analysis.physical_parameters import compute_dimensionless_numbers
from src.simulation_io.analysis.physical_parameters import dimensionless_label
from src.simulation_io.plotting._simulation_csv import build_simulation_csv
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE
from src.simulation_io.plotting.figure_config import REGIME_COLORS
from src.simulation_io.plotting.figure_config import REGIME_MARKERS
from src.simulation_io.plotting.run_comparison import _clean_dir_label
from src.simulation_io.plotting.run_comparison import _safe_load_config

if TYPE_CHECKING:
    from collections.abc import Sequence
    import pandas as pd
    from src.config import SimulationConfig
    from src.simulation_io.analysis.accelerations import RegimeResult
    from src.simulation_io.analysis.physical_parameters import DimensionlessNumbers

# Output names come from src.config.run_config and regime styling from
# plotting.figure_config; this one is a local invariant, not configuration: a
# run needs two rows before a difference can be taken.
_MIN_CSV_ROWS = 2


#: Axis pair used when the caller names none -- what this map plotted before
#: the axes became selectable.
DEFAULT_X_KEY = "bo_parallel"
DEFAULT_Y_KEY = "oh"


@dataclass(frozen=True)
class RunRegimeEntry:
    """One classified run, carrying every dimensionless number it resolved.

    The whole set is kept rather than two chosen values so that classifying a
    run -- by far the expensive part, since it reads every snapshot -- does not
    have to know which axes will be plotted.
    """

    run_dir: Path
    label: str
    numbers: DimensionlessNumbers
    regime: str


def _load_or_build_csv(run_dir: Path, config: SimulationConfig) -> tuple[pd.DataFrame, float] | None:
    """Metric table and ``r_zero`` for *run_dir*, also writing ``simulation_data.csv``.

    The frame is built in memory from the shared droplet series rather than
    read back from the file just written. ``build_simulation_csv`` is still
    called for its on-disk side effect; it hits the same cached series, so the
    snapshots are read only once. ``r_zero`` is returned from the series' own
    scales rather than re-resolved, which would re-open the init snapshot.
    """
    # Preserves the historical gate: unsupported sim_types produce no CSV.
    if build_simulation_csv(run_dir, config) is None:
        return None
    series = droplet_series_for_run(run_dir, config)
    if series is None:
        return None
    return series.to_dataframe(), series.scales.r_zero


def _regime_annotation(result: RegimeResult) -> str:
    """One line stating the verdict and the numbers it rests on.

    Drawn onto the run's own acceleration figure so a label can be checked -- and
    its distance from the significance threshold seen -- without re-running the
    classifier.
    """
    parts = [str(result.regime)]
    if result.drift is not None:
        parts.append(f"drift {100 * result.drift:+.2f}%")
    if result.t_statistic is not None:
        parts.append(f"t {result.t_statistic:+.2f}")
    if result.window is not None:
        parts.append(f"window [{result.window[0]}, {result.window[1]}]")
    if not result.is_robust:
        parts.append("sign flips with window")
    return " | ".join(parts)


def process_run_dir(run_dir: Path, *, smoothing: Smoothing = Smoothing.RAW) -> RunRegimeEntry | None:
    """Classify one run directory, or return ``None`` (with a warning) when unusable."""
    config = _safe_load_config(run_dir / CONFIG_FILENAME)
    if config is None:
        return None

    built = _load_or_build_csv(run_dir, config)
    if built is None:
        print(f"  Skipped {run_dir}: no simulation data available")
        return None
    df, r_zero = built
    if len(df) < _MIN_CSV_ROWS:
        print(f"  Skipped {run_dir}: simulation_data.csv has fewer than {_MIN_CSV_ROWS} rows")
        return None

    accel_result = compute_acceleration(df, smoothing=smoothing)
    regime_result = classify_regime(df["cm_x"].to_numpy(dtype=float), r_zero, accel_result)
    save_diagnostic_plot(
        accel_result,
        regime_result.window,
        run_dir / PLOTS_DIRNAME / ANALYSIS_DIRNAME / ACCELERATION_PLOT_FILENAME,
        annotation=_regime_annotation(regime_result),
    )

    label = config.simulation_name or _clean_dir_label(run_dir.name)
    return RunRegimeEntry(
        run_dir=run_dir,
        label=label,
        numbers=compute_dimensionless_numbers(config),
        regime=regime_result.regime,
    )


def _axis_values(entries: Sequence[RunRegimeEntry], x_key: str, y_key: str) -> list[RunRegimeEntry]:
    """Entries resolving both axes, naming those dropped.

    A run may classify perfectly well yet lack the number a given axis asks for
    -- ``Bo`` on a run with no gravity, say -- so the axis gate lives here
    rather than in classification.
    """
    usable: list[RunRegimeEntry] = []
    for entry in entries:
        missing = [key for key in (x_key, y_key) if entry.numbers.get(key) is None]
        if missing:
            print(f"  Skipped {entry.run_dir}: {' and '.join(missing)} could not be resolved")
            continue
        usable.append(entry)
    return usable


def plot_regime_map(
    entries: list[RunRegimeEntry],
    out_path: str | Path,
    *,
    x_key: str = DEFAULT_X_KEY,
    y_key: str = DEFAULT_Y_KEY,
    xscale: AxisScale = AxisScale.LINEAR,
    yscale: AxisScale = AxisScale.LINEAR,
) -> Path | None:
    """Scatter *x_key* against *y_key*, grouped by regime.

    Returns ``None`` when no entry resolves both axes, so the caller can say so
    rather than saving an empty figure.
    """
    import matplotlib.pyplot as plt

    usable = _axis_values(entries, x_key, y_key)
    if not usable:
        return None

    fig, ax = plt.subplots(figsize=DEFAULT_STYLE.comparison_figsize)
    regimes_present = sorted({entry.regime for entry in usable})
    for regime in regimes_present:
        in_regime = [e for e in usable if e.regime == regime]
        ax.scatter(
            [e.numbers.get(x_key) for e in in_regime],
            [e.numbers.get(y_key) for e in in_regime],
            color=REGIME_COLORS[regime],
            marker=REGIME_MARKERS[regime],
            label=regime,
            s=DEFAULT_STYLE.scatter_marker_size * 4,
            alpha=DEFAULT_STYLE.scatter_alpha,
        )

    ax.set_xlabel(dimensionless_label(x_key), fontsize=DEFAULT_STYLE.comparison_axis_label_fontsize)
    ax.set_ylabel(dimensionless_label(y_key), fontsize=DEFAULT_STYLE.comparison_axis_label_fontsize)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.tick_params(axis="both", labelsize=DEFAULT_STYLE.comparison_tick_label_fontsize)
    ax.legend(fontsize=DEFAULT_STYLE.comparison_legend_fontsize, loc="best")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_STYLE.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_regime_map(
    run_dirs: Sequence[Path],
    out_dir: str | Path,
    *,
    smoothing: Smoothing = Smoothing.RAW,
    x_key: str = DEFAULT_X_KEY,
    y_key: str = DEFAULT_Y_KEY,
    xscale: AxisScale = AxisScale.LINEAR,
    yscale: AxisScale = AxisScale.LINEAR,
) -> Path | None:
    """Classify every run in *run_dirs* and save the regime map into *out_dir*.

    ``smoothing`` controls the acceleration curve used for peak detection in
    each run's diagnostic plot (see
    :func:`src.simulation_io.analysis.accelerations.acceleration_analysis.compute_acceleration`).

    ``x_key``/``y_key`` name any two registered dimensionless numbers (see
    :func:`src.simulation_io.analysis.physical_parameters.dimensionless_keys`),
    and ``xscale``/``yscale`` set each axis linear or logarithmic. The figure is
    named after the pair, so several pairs coexist in one output directory.

    Returns the path to the saved figure, or ``None`` when no run classified or
    none resolved both axes.
    """
    print(f"Found {len(run_dirs)} run director(y/ies) to classify.")

    entries: list[RunRegimeEntry] = []
    for run_dir in run_dirs:
        entry = process_run_dir(run_dir, smoothing=smoothing)
        if entry is not None:
            print(f"  {entry.run_dir} -> {entry.regime}  ({_axis_summary(entry, x_key, y_key)})")
            entries.append(entry)

    if not entries:
        return None

    out_path = Path(out_dir) / regime_map_filename(x_key, y_key)
    saved = plot_regime_map(entries, out_path, x_key=x_key, y_key=y_key, xscale=xscale, yscale=yscale)
    if saved is None:
        return None
    print(f"Saved {saved}")
    return saved


def _axis_summary(entry: RunRegimeEntry, x_key: str, y_key: str) -> str:
    """``"bo_parallel=0.6, oh=0.3"`` for the per-run progress line."""
    parts = []
    for key in (x_key, y_key):
        value = entry.numbers.get(key)
        parts.append(f"{key}={'unresolved' if value is None else f'{value:.4g}'}")
    return ", ".join(parts)
