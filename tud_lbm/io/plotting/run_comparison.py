"""Cross-run comparison plots and batch CSV-export CLI entry point."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from typing import TypedDict
from matplotlib.colors import TABLEAU_COLORS
from tud_lbm.io.plotting.simulation_csv import _CSV_FILENAME
from tud_lbm.io.plotting.simulation_csv import build_simulation_csv

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd
    from tud_lbm.config import SimulationConfig

_DIR_SPLIT_PARTS = 2
_LABEL_CA = r"$\mathrm{Ca}$"
_LABEL_IT_NORM = r"$\Delta\mathrm{t}/\mathrm{t}_{\mathrm{max}}$"
_LABEL_X_AVG_NORM = r"$X_{\mathrm{avg}}/R_0$"
_CONFIG_TOML = "config.toml"
_COMPARISON_DIR = "comparison_analysis"


class _PlotConfig(TypedDict, total=False):
    filename: str
    x: str
    x_pair: tuple[str, str]
    y: str
    y_pair: tuple[str, str]
    pair_labels: tuple[str, str]
    xlabel: str
    ylabel: str


_COMPARISON_PLOT_CONFIGS: list[_PlotConfig] = [
    {
        "filename": "01_Ca_vs_iteration.png",
        "x": "normalised_iteration",
        "y": "Ca",
        "xlabel": _LABEL_IT_NORM,
        "ylabel": _LABEL_CA,
    },
    {
        "filename": "02_Ca_vs_x_avg.png",
        "x": "avg_x_location_norm",
        "y": "Ca",
        "xlabel": _LABEL_X_AVG_NORM,
        "ylabel": _LABEL_CA,
    },
    {
        "filename": "03_Ca_contact_line_vs_iteration.png",
        "x": "normalised_iteration",
        "y_pair": ("Ca_cll_left", "Ca_cll_right"),
        "pair_labels": ("Trailing", "Leading"),
        "xlabel": _LABEL_IT_NORM,
        "ylabel": _LABEL_CA,
    },
    {
        "filename": "04_Ca_contact_line_vs_x_avg.png",
        "x": "avg_x_location_norm",
        "y_pair": ("Ca_cll_left", "Ca_cll_right"),
        "pair_labels": ("Trailing", "Leading"),
        "xlabel": _LABEL_X_AVG_NORM,
        "ylabel": _LABEL_CA,
    },
    {
        "filename": "05_Ca_cm_vs_iteration.png",
        "x": "normalised_iteration",
        "y": "Ca_cm",
        "xlabel": _LABEL_IT_NORM,
        "ylabel": _LABEL_CA,
    },
    {
        "filename": "06_Ca_cm_vs_x_avg.png",
        "x": "avg_x_location_norm",
        "y": "Ca_cm",
        "xlabel": _LABEL_X_AVG_NORM,
        "ylabel": _LABEL_CA,
    },
    {
        "filename": "07_contact_angles_vs_iteration.png",
        "x": "normalised_iteration",
        "y_pair": ("ca_left", "ca_right"),
        "pair_labels": ("Trailing", "Leading"),
        "xlabel": _LABEL_IT_NORM,
        "ylabel": "Contact angle (degrees)",
    },
    {
        "filename": "08_contact_angles_vs_x_avg.png",
        "x": "avg_x_location_norm",
        "y_pair": ("ca_left", "ca_right"),
        "pair_labels": ("Trailing", "Leading"),
        "xlabel": _LABEL_X_AVG_NORM,
        "ylabel": r"$\theta$ (degrees)",
    },
    {
        "filename": "09_contact_angle_vs_Ca_cll.png",
        "x_pair": ("Ca_cll_left", "Ca_cll_right"),
        "y_pair": ("ca_left", "ca_right"),
        "pair_labels": ("Trailing", "Leading"),
        "xlabel": r"$\mathrm{Ca}_{\mathrm{CL}}$ [−]",
        "ylabel": r"$\theta$ (degrees)",
    },
]

_COMPARISON_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x"]


def _safe_load_config(toml_path: Path) -> SimulationConfig | None:
    from tud_lbm.io.readers import TomlAdapter

    try:
        return TomlAdapter().load(str(toml_path))
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipped {toml_path.parent}: {exc}")
        return None


def _load_comparison_entries(parent_dir: Path) -> list[dict]:
    """Collect all processed run CSVs under *parent_dir*, sorted by advancing CA."""
    try:
        import pandas as pd
    except ImportError:
        return []

    entries = []
    for csv_path in sorted(parent_dir.rglob(_CSV_FILENAME)):
        run_dir = csv_path.parent
        if _COMPARISON_DIR in run_dir.parts:
            continue
        toml_path = run_dir / _CONFIG_TOML
        if not toml_path.exists():
            continue
        config = _safe_load_config(toml_path)
        if config is None:
            continue

        adv_ca = config.wetting_config.get("advancing_ca", 0) if config.wetting_config else 0
        label = config.simulation_name or _clean_dir_label(run_dir.name)
        entries.append(
            {
                "label": label,
                "sort_key": adv_ca,
                "data": pd.read_csv(csv_path),
            }
        )

    return sorted(entries, key=lambda e: e["sort_key"])


def _clean_dir_label(dir_name: str) -> str:
    parts = dir_name.split("_", 1)
    name = parts[1] if len(parts) == _DIR_SPLIT_PARTS and parts[0].replace("-", "").isdigit() else dir_name
    return name.replace("_", " ").capitalize()


def _plot_comparison_entry(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    pc: _PlotConfig,
    x_col: str,
    color: str,
    marker: str,
    label: str,
) -> None:
    """Plot one entry from _COMPARISON_PLOT_CONFIGS for a single run."""
    if "y_pair" in pc:
        y1, y2 = pc["y_pair"]
        l1, l2 = pc["pair_labels"]
        if "x_pair" in pc:
            x1_col, x2_col = pc["x_pair"]
            if x1_col in df.columns and y1 in df.columns:
                ax.scatter(df[x1_col], df[y1], marker=marker, s=15, color=color, alpha=1.0, label=f"{label} ({l1})")
            if x2_col in df.columns and y2 in df.columns:
                ax.scatter(df[x2_col], df[y2], marker=marker, s=15, color=color, alpha=0.5, label=f"{label} ({l2})")
        else:
            if y1 in df.columns:
                ax.scatter(df[x_col], df[y1], marker=marker, s=15, color=color, alpha=1.0, label=f"{label} ({l1})")
            if y2 in df.columns:
                ax.scatter(df[x_col], df[y2], marker=marker, s=15, color=color, alpha=0.5, label=f"{label} ({l2})")
    else:
        y_col = pc["y"]
        if y_col in df.columns:
            ax.scatter(df[x_col], df[y_col], marker=marker, s=15, color=color, label=label)


def compare_runs(parent_dir: str | Path) -> None:
    """Generate comparison scatter plots across all processed runs.

    Reads ``simulation_data.csv`` files produced by :func:`build_simulation_csv`
    from every run directory under *parent_dir* and writes plots to
    ``<parent_dir>/comparison_analysis/``.

    Args:
        parent_dir: Parent directory that contains the individual run directories.
    """
    import matplotlib.pyplot as plt
    from tud_lbm.io.plotting.figure_config import DEFAULT_STYLE

    parent_dir = Path(parent_dir)
    entries = _load_comparison_entries(parent_dir)
    if not entries:
        print("No processed simulation data found for comparison.")
        return

    out_dir = parent_dir / _COMPARISON_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = list(TABLEAU_COLORS.values())

    for pc in _COMPARISON_PLOT_CONFIGS:
        fig, ax = plt.subplots(figsize=DEFAULT_STYLE.comparison_figsize)
        for i, entry in enumerate(entries):
            color = str(colors[i % len(colors)])
            marker = _COMPARISON_MARKERS[i % len(_COMPARISON_MARKERS)]
            x_col = pc["x_pair"][0] if "x_pair" in pc else pc["x"]
            if x_col not in entry["data"].columns:
                continue
            _plot_comparison_entry(ax, entry["data"], pc, x_col, color, marker, entry["label"])

        ax.set_xlabel(pc["xlabel"], fontsize=DEFAULT_STYLE.comparison_axis_label_fontsize)
        ax.set_ylabel(pc["ylabel"], fontsize=DEFAULT_STYLE.comparison_axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=DEFAULT_STYLE.comparison_tick_label_fontsize)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=DEFAULT_STYLE.comparison_legend_fontsize, loc="best")
        ax.grid(False)
        plt.tight_layout()
        out_path = out_dir / pc["filename"]
        fig.savefig(out_path, dpi=DEFAULT_STYLE.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def process_parent_dir(
    parent_dir: str | Path,
    fields: list[str] | None = None,
) -> tuple[int, int]:
    """Discover runs, export per-run CSVs, and generate comparison plots.

    Discovers runs by searching for ``config.toml`` files recursively.
    Skips directories whose path contains ``"init"`` or ``"comparison_analysis"``.
    Writes ``simulation_data.csv`` per run, then generates 8 comparison plots
    when at least one run produced CSV data.

    Args:
        parent_dir: Absolute or relative path to the parent directory.
        fields: Comparison operator names to run as per-run analysis plots.
            When ``None``, no per-run analysis plots are generated beyond the CSV.

    Returns:
        A tuple ``(n_runs_found, n_runs_with_csv)``.
    """
    from tud_lbm.io.plotting.figure_builder import FigureBuilder

    parent = Path(parent_dir)
    skip_dirs = {"init", _COMPARISON_DIR}

    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for toml in sorted(parent.rglob(_CONFIG_TOML)):
        rd = toml.parent
        if rd in seen or any(s in str(rd).lower() for s in skip_dirs):
            continue
        seen.add(rd)
        run_dirs.append(rd)

    if not run_dirs:
        return 0, 0

    print(f"Found {len(run_dirs)} simulation(s) to process.")

    n_ok = 0
    for rd in run_dirs:
        config = _safe_load_config(rd / _CONFIG_TOML)
        if config is None:
            continue
        if build_simulation_csv(rd, config) is not None:
            n_ok += 1
        if fields:
            FigureBuilder(config=config, run_dir=rd, fields=fields).build_analysis()

    if n_ok > 0:
        print("\nGenerating comparison plots...")
        compare_runs(parent)
        print(f"Done. Comparison plots in {parent / _COMPARISON_DIR}")

    return len(run_dirs), n_ok


def main(parent_dir: str | None = None) -> None:
    """Script entry point for batch CSV export and run comparison plots."""
    if parent_dir is None:
        parent_dir = input("Enter parent directory (absolute path): ").strip()

    parent = Path(parent_dir)
    if not parent.is_dir():
        print(f"Directory not found: {parent}")
        sys.exit(1)

    n_runs, n_ok = process_parent_dir(parent_dir)

    if n_runs == 0:
        print("No simulation run directories found.")
        sys.exit(1)
    if n_ok == 0:
        print("No runs produced CSV data. Check sim_type and data files.")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
