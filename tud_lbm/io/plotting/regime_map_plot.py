"""Aggregate Ca(t) regime classification across many run directories.

Reads `simulation_data.csv` (building it on demand if missing) for every run
directory listed in a plain-text file, classifies each run into
pinning/viscous/inertial/unknown (see
:mod:`tud_lbm.io.analysis.accelerations.regime_classification`), and plots the
result as Bo_parallel vs Oh — the regime map.
"""

from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import pandas as pd
from tud_lbm.io.analysis.accelerations import Smoothing
from tud_lbm.io.analysis.accelerations import classify_regime
from tud_lbm.io.analysis.accelerations import compute_acceleration
from tud_lbm.io.analysis.accelerations import save_diagnostic_plot
from tud_lbm.io.analysis.physical_parameters import compute_dimensionless_numbers
from tud_lbm.io.plotting.figure_config import DEFAULT_STYLE
from tud_lbm.io.plotting.run_comparison import _clean_dir_label
from tud_lbm.io.plotting.run_comparison import _safe_load_config
from tud_lbm.io.plotting.simulation_csv import _CSV_FILENAME
from tud_lbm.io.plotting.simulation_csv import _resolve_r_zero
from tud_lbm.io.plotting.simulation_csv import build_simulation_csv

if TYPE_CHECKING:
    from tud_lbm.config import SimulationConfig

_CONFIG_TOML = "config.toml"
_REGIME_MAP_DIR = "regime_map_analysis"
_REGIME_MAP_FILENAME = "regime_map.png"
_DIAGNOSTIC_FILENAME = "acceleration_analysis.png"
_PLOTS_DIR = "plots"
_ANALYSIS_DIR = "analysis"
_MIN_CSV_ROWS = 2
_MIN_QUOTED_LINE_LEN = 2

_REGIME_MARKERS: dict[str, str] = {"Pinning": "o", "Dissipative": "s", "Inertial": "^", "unknown": "x"}
_REGIME_COLORS: dict[str, str] = {
    "Pinning": "tab:blue",
    "Dissipative": "tab:green",
    "Inertial": "tab:red",
    "unknown": "tab:gray",
}


@dataclass(frozen=True)
class RunRegimeEntry:
    """One classified run, ready to be plotted on the regime map."""

    run_dir: Path
    label: str
    bo_parallel: float
    oh: float
    regime: str


_SHELL_ESCAPE_RE = re.compile(r"\\[ $;='\"\\]")


def parse_run_dir_list(txt_path: str | Path) -> list[Path]:
    r"""Parse one run-dir path per non-blank, non-``#`` line.

    Lines may be shell-quoted/escaped (e.g. copied from a terminal listing of
    directory names containing spaces or special characters); such quoting is
    stripped and backslash escapes are resolved before building the path.
    This applies whether or not the line is wrapped in quotes — shell
    tab-completion commonly emits backslash-escaped, *unquoted* lines (e.g.
    ``\$Bo_\\parallel\ \=\ 0.60\;\ Oh\ \=\ 0.30\$``). A line is only treated
    as escaped if it contains a backslash before a shell metacharacter
    (space, ``$``, ``;``, ``=``, a quote, or another backslash) — a lone
    literal backslash that is simply part of the directory name (e.g.
    ``\parallel``) is left untouched.

    Each line is treated as exactly one path, even if it contains
    (escaped or unescaped) whitespace — word-splitting would silently
    truncate paths like ``$Bo_\\parallel = 0.60; Oh = 0.30$`` at the first
    space.

    Relative paths are resolved against ``txt_path``'s parent directory.

    A directory name containing a literal backslash (as in the ``\parallel``
    example above) can only be created on a POSIX filesystem — Windows always
    treats ``\`` as a path separator, so such a name can never exist as a
    single path component there. If such a path can't be found on a Windows
    host, this raises :class:`ValueError` with a clear explanation rather
    than letting a confusing :class:`FileNotFoundError` surface later.
    """
    txt_path = Path(txt_path)
    parent = txt_path.parent
    dirs: list[Path] = []
    for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        quoted = len(line) >= _MIN_QUOTED_LINE_LEN and line[0] == line[-1] and line[0] in ("'", '"')
        if quoted:
            line = line[1:-1]
        if quoted or _SHELL_ESCAPE_RE.search(line):
            line = re.sub(r"\\(.)", r"\1", line)
        if not line:
            continue
        path = Path(line)
        resolved = path if path.is_absolute() else parent / path
        if sys.platform == "win32" and "\\" in line and not resolved.exists():
            msg = (
                f"{txt_path}: run directory {line!r} contains a literal backslash and "
                "could not be found. Windows always treats '\\' as a path separator, "
                "so directory names created on POSIX systems (e.g. a Linux/macOS HPC "
                "cluster) with a backslash in the name cannot be represented as a "
                "single path component here. Run this analysis from a POSIX "
                "environment (Linux, macOS, or WSL) instead."
            )
            raise ValueError(msg)
        dirs.append(resolved)
    return dirs


def _load_or_build_csv(run_dir: Path, config: SimulationConfig) -> pd.DataFrame | None:
    csv_path = run_dir / _CSV_FILENAME
    if not csv_path.exists() and build_simulation_csv(run_dir, config) is None:
        return None
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def process_run_dir(run_dir: Path, *, smoothing: Smoothing = "raw") -> RunRegimeEntry | None:
    """Classify one run directory, or return ``None`` (with a warning) when unusable."""
    config = _safe_load_config(run_dir / _CONFIG_TOML)
    if config is None:
        return None

    df = _load_or_build_csv(run_dir, config)
    if df is None:
        print(f"  Skipped {run_dir}: no simulation data available")
        return None
    if len(df) < _MIN_CSV_ROWS:
        print(f"  Skipped {run_dir}: simulation_data.csv has fewer than {_MIN_CSV_ROWS} rows")
        return None

    dn = compute_dimensionless_numbers(config)
    if dn.oh is None or dn.bo_parallel is None:
        print(f"  Skipped {run_dir}: Oh/Bo_parallel could not be resolved (missing surface tension or gravity)")
        return None

    r_zero = _resolve_r_zero(config).value
    accel_result = compute_acceleration(df, smoothing=smoothing)
    regime_result = classify_regime(df["cm_x"].to_numpy(dtype=float), r_zero, accel_result)
    save_diagnostic_plot(
        accel_result, regime_result.window, run_dir / _PLOTS_DIR / _ANALYSIS_DIR / _DIAGNOSTIC_FILENAME
    )

    label = config.simulation_name or _clean_dir_label(run_dir.name)
    return RunRegimeEntry(
        run_dir=run_dir,
        label=label,
        bo_parallel=dn.bo_parallel,
        oh=dn.oh,
        regime=regime_result.regime,
    )


def plot_regime_map(entries: list[RunRegimeEntry], out_path: str | Path) -> Path:
    """Scatter Bo_parallel (x) vs Oh (y), grouped by regime."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=DEFAULT_STYLE.comparison_figsize)
    regimes_present = sorted({entry.regime for entry in entries})
    for regime in regimes_present:
        xs = [e.bo_parallel for e in entries if e.regime == regime]
        ys = [e.oh for e in entries if e.regime == regime]
        ax.scatter(
            xs,
            ys,
            color=_REGIME_COLORS[regime],
            marker=_REGIME_MARKERS[regime],
            label=regime,
            s=DEFAULT_STYLE.scatter_marker_size * 4,
            alpha=DEFAULT_STYLE.scatter_alpha,
        )

    ax.set_xlabel(r"$Bo_{\parallel}$", fontsize=DEFAULT_STYLE.comparison_axis_label_fontsize)
    ax.set_ylabel("Oh", fontsize=DEFAULT_STYLE.comparison_axis_label_fontsize)
    ax.tick_params(axis="both", labelsize=DEFAULT_STYLE.comparison_tick_label_fontsize)
    ax.legend(fontsize=DEFAULT_STYLE.comparison_legend_fontsize, loc="best")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_STYLE.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_regime_map(
    txt_path: str | Path, out_dir: str | Path | None = None, *, smoothing: Smoothing = "raw"
) -> Path | None:
    """Classify every run listed in ``txt_path`` and save the regime map.

    ``smoothing`` controls the acceleration curve used for peak detection in
    each run's diagnostic plot (see
    :func:`tud_lbm.io.analysis.accelerations.acceleration_analysis.compute_acceleration`).

    Returns the path to ``regime_map.png``, or ``None`` if no run produced a
    usable entry.
    """
    txt_path = Path(txt_path)
    run_dirs = parse_run_dir_list(txt_path)
    print(f"Found {len(run_dirs)} run director(y/ies) to classify.")

    entries: list[RunRegimeEntry] = []
    for run_dir in run_dirs:
        entry = process_run_dir(run_dir, smoothing=smoothing)
        if entry is not None:
            print(f"  {entry.run_dir} -> {entry.regime}  (Bo_parallel={entry.bo_parallel:.4g}, Oh={entry.oh:.4g})")
            entries.append(entry)

    if not entries:
        return None

    out_dir = Path(out_dir) if out_dir is not None else txt_path.parent / _REGIME_MAP_DIR
    out_path = out_dir / _REGIME_MAP_FILENAME
    plot_regime_map(entries, out_path)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    build_regime_map(sys.argv[1])
