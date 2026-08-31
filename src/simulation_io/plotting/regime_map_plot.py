"""Aggregate Ca(t) regime classification across many run directories.

Reads `simulation_data.csv` (building it on demand if missing) for every run
directory listed in a plain-text file, classifies each run into
pinning/viscous/inertial/unknown (see
:mod:`src.simulation_io.analysis.accelerations.regime_classification`), and plots the
result as Bo_parallel vs Oh — the regime map.
"""

from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from src.config.config_overview import BASE_RESULTS_DIR
from src.config.run_config import ACCELERATION_PLOT_FILENAME
from src.config.run_config import ANALYSIS_DIRNAME
from src.config.run_config import CONFIG_FILENAME
from src.config.run_config import PLOTS_DIRNAME
from src.config.run_config import REGIME_MAP_DIRNAME
from src.config.run_config import REGIME_MAP_FILENAME
from src.simulation_io.analysis.accelerations import Smoothing
from src.simulation_io.analysis.accelerations import classify_regime
from src.simulation_io.analysis.accelerations import compute_acceleration
from src.simulation_io.analysis.accelerations import save_diagnostic_plot
from src.simulation_io.analysis.droplet_metrics import droplet_series_for_run
from src.simulation_io.analysis.physical_parameters import compute_dimensionless_numbers
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE
from src.simulation_io.plotting.figure_config import REGIME_COLORS
from src.simulation_io.plotting.figure_config import REGIME_MARKERS
from src.simulation_io.plotting.run_comparison import _clean_dir_label
from src.simulation_io.plotting.run_comparison import _safe_load_config
from src.simulation_io.plotting.simulation_csv import build_simulation_csv

if TYPE_CHECKING:
    from collections.abc import Sequence
    import pandas as pd
    from src.config import SimulationConfig

# Output names come from src.config.run_config and regime styling from
# plotting.figure_config; the two below are local parsing invariants, not
# configuration: a run needs two rows before a difference can be taken, and a
# quoted line needs its two quote characters.
_MIN_CSV_ROWS = 2
_MIN_QUOTED_LINE_LEN = 2


@dataclass(frozen=True)
class RunRegimeEntry:
    """One classified run, ready to be plotted on the regime map."""

    run_dir: Path
    label: str
    bo_parallel: float
    oh: float
    regime: str


_SHELL_ESCAPE_RE = re.compile(r"\\[ $;='\"\\]")


def parse_run_dir_list(txt_path: str | Path, allowed_roots: Sequence[str | Path] = ()) -> list[Path]:
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

    Every entry (relative or absolute) must resolve inside at least one
    trusted root: ``BASE_RESULTS_DIR`` (this project's default results
    location) plus any extra directories passed via ``allowed_roots``. An
    entry outside all of them (e.g. a relative ``..`` escape, or an absolute
    path pointing somewhere untrusted) is rejected with :class:`ValueError`.
    Unlike the list file's own directory, ``allowed_roots`` are supplied
    directly by the caller rather than parsed from the list file's content,
    so they form a real trust boundary rather than one derived from the
    same untrusted text being validated (see SonarQube rule
    ``pythonsecurity:S6549`` — a canonicalized path is only "sanitized" if
    checked against a fixed, independently-supplied root).

    A directory name containing a literal backslash (as in the ``\parallel``
    example above) can only be created on a POSIX filesystem — Windows always
    treats ``\`` as a path separator, so such a name can never exist as a
    single path component there. On a Windows host such an entry is instead
    silently split into nested path components like any other Windows path;
    this function does not probe the filesystem to disambiguate that case,
    since doing so on unvalidated, user-controlled path text would make file
    existence itself an oracle. Callers that hit an unexpectedly missing run
    directory on Windows should check the list file for a backslash that was
    meant as a literal character rather than a path separator.

    ``txt_path`` itself (which may originate from an untrusted CLI argument,
    e.g. when this tool is driven by an external or automated caller) is
    resolved to a canonical, absolute path and confirmed to be an existing
    regular file *before* anything is read from it, so a directory, device
    file, or nonexistent path is rejected with :class:`ValueError` up front
    rather than reaching the filesystem read in an unvalidated state.
    """
    txt_path = Path(txt_path).resolve()
    if not txt_path.is_file():
        msg = f"{txt_path}: run-dir list must be an existing regular file"
        raise ValueError(msg)
    parent = txt_path.parent
    resolved_roots = [Path(BASE_RESULTS_DIR).resolve(), *(Path(root).resolve() for root in allowed_roots)]
    dirs: list[Path] = []
    for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = _unquote_and_unescape(line)
        if not line:
            continue
        dirs.append(_resolve_run_dir_entry(line, txt_path, parent, resolved_roots))
    return dirs


def _unquote_and_unescape(line: str) -> str:
    """Strip a single layer of shell quoting and resolve backslash escapes."""
    quoted = len(line) >= _MIN_QUOTED_LINE_LEN and line[0] == line[-1] and line[0] in ("'", '"')
    if quoted:
        line = line[1:-1]
    if quoted or _SHELL_ESCAPE_RE.search(line):
        line = re.sub(r"\\(.)", r"\1", line)
    return line


def _resolve_run_dir_entry(line: str, txt_path: Path, parent: Path, resolved_roots: list[Path]) -> Path:
    """Resolve one parsed line to a run directory, rejecting entries outside every allowed root.

    Validation follows the canonical-path pattern for SonarQube ``pythonsecurity:S6549``
    (transform -> normalize/resolve -> sanitize -> use): the untrusted ``line`` is first
    resolved to a canonical path, then checked with ``Path.relative_to`` against each
    trusted root inside a ``try/except ValueError`` — the exact idiom the rule's own
    compliant example uses — before the path is returned for any later filesystem use.
    """
    path = Path(line)
    resolved = (path if path.is_absolute() else parent / path).resolve()
    for root in resolved_roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved
    roots_desc = ", ".join(str(root) for root in resolved_roots)
    msg = (
        f"{txt_path}: run directory {line!r} resolves to {resolved}, which "
        f"is outside every allowed root ({roots_desc}). Pass additional "
        "trusted locations via --allowed-root."
    )
    raise ValueError(msg)


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


def process_run_dir(run_dir: Path, *, smoothing: Smoothing = "raw") -> RunRegimeEntry | None:
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

    dn = compute_dimensionless_numbers(config)
    if dn.oh is None or dn.bo_parallel is None:
        print(f"  Skipped {run_dir}: Oh/Bo_parallel could not be resolved (missing surface tension or gravity)")
        return None

    accel_result = compute_acceleration(df, smoothing=smoothing)
    regime_result = classify_regime(df["cm_x"].to_numpy(dtype=float), r_zero, accel_result)
    save_diagnostic_plot(
        accel_result, regime_result.window, run_dir / PLOTS_DIRNAME / ANALYSIS_DIRNAME / ACCELERATION_PLOT_FILENAME
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
            color=REGIME_COLORS[regime],
            marker=REGIME_MARKERS[regime],
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
    txt_path: str | Path,
    allowed_roots: Sequence[str | Path] = (),
    out_dir: str | Path | None = None,
    *,
    smoothing: Smoothing = "raw",
) -> Path | None:
    """Classify every run listed in ``txt_path`` and save the regime map.

    ``allowed_roots`` are additional trusted directories (beyond the default
    results root) that every run-dir entry must resolve within; see
    :func:`parse_run_dir_list`.

    ``smoothing`` controls the acceleration curve used for peak detection in
    each run's diagnostic plot (see
    :func:`src.simulation_io.analysis.accelerations.acceleration_analysis.compute_acceleration`).

    Returns the path to ``regime_map.png``, or ``None`` if no run produced a
    usable entry.
    """
    txt_path = Path(txt_path)
    run_dirs = parse_run_dir_list(txt_path, allowed_roots)
    print(f"Found {len(run_dirs)} run director(y/ies) to classify.")

    entries: list[RunRegimeEntry] = []
    for run_dir in run_dirs:
        entry = process_run_dir(run_dir, smoothing=smoothing)
        if entry is not None:
            print(f"  {entry.run_dir} -> {entry.regime}  (Bo_parallel={entry.bo_parallel:.4g}, Oh={entry.oh:.4g})")
            entries.append(entry)

    if not entries:
        return None

    out_dir = Path(out_dir) if out_dir is not None else txt_path.parent / REGIME_MAP_DIRNAME
    out_path = out_dir / REGIME_MAP_FILENAME
    plot_regime_map(entries, out_path)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    build_regime_map(sys.argv[1])
