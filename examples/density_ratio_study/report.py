"""Summarise one study phase: ``results.csv``, diagnostic figures and a markdown table.

``python -m examples.density_ratio_study.report <phase>`` reads every
``~/TUD_LBM_data/density_ratio_study/<phase>/*/summary.json`` (and ``run.npz`` when
the case ran) and writes, into the phase directory:

* ``results.csv`` — one row per case;
* ``series.png`` — vapour Mach, minimum density, vapour zig-zag amplitude and
  bubble centroid against time, one marker colour per case (scatter, per the
  project's plot-type rule);
* ``frames.png`` — the last healthy density frame of every case that ran;
* ``table.md`` — the results as a markdown table, for pasting into the report.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("~/TUD_LBM_data/density_ratio_study").expanduser()
#: Same rule as ``run_study.BUBBLE_LOST_FRACTION``.
BUBBLE_LOST_FRACTION = 0.1

#: Columns of ``table.md``, in order; everything is kept in ``results.csv``.
TABLE_COLUMNS = (
    "case",
    "status",
    "t_end",
    "c_v",
    "max_vapour_mach",
    "max_stripe",
    "stripe_growth_per_kstep",
    "rho_min",
    "gas_area_ratio",
    "gas_rise",
    "fail_mach_cell_phase",
    "fail_rhomin_near_wall",
)

_SERIES = (
    ("vapour_mach", "max vapour Mach", "log"),
    ("rho_min", r"min $\rho$", "log"),
    ("stripe_amp", "vapour zig-zag amplitude (rel.)", "log"),
    ("gas_yc", "bubble centroid y", "linear"),
)


def _growth_rate(t: np.ndarray, amp: np.ndarray) -> float | None:
    """e-folding rate of the zig-zag amplitude per 1000 steps, fitted on its last half.

    The first half is excluded: it holds the relaxation of the initial tanh profile.
    """
    ok = np.isfinite(amp) & (amp > 0.0)
    t, amp = t[ok], amp[ok]
    min_points = 4
    if t.size < min_points:
        return None
    half = t.size // 2
    slope = np.polyfit(t[half:], np.log(amp[half:]), 1)[0]
    return float(slope * 1000.0)


def load_phase(phase: str) -> list[tuple[dict[str, Any], dict[str, np.ndarray] | None]]:
    rows = []
    for summary_path in sorted((RESULTS / phase).glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        npz = summary_path.with_name("run.npz")
        series = None
        if npz.exists():
            with np.load(npz) as data:
                series = {k.removeprefix("series_"): data[k] for k in data.files if k.startswith("series_")}
                series["frame"] = data["rho"][-1]
            summary["stripe_growth_per_kstep"] = _growth_rate(series["t"], series["stripe_amp"])
            # Reclassify runs written before the driver knew the status (same rule).
            area = series["gas_area"]
            if summary.get("status") == "survived" and area[np.isfinite(area)][-1] < BUBBLE_LOST_FRACTION * area[0]:
                summary["status"] = "bubble_lost"
        rows.append((summary, series))
    return rows


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3g}"
    return "" if value is None else str(value)


def write_outputs(phase: str) -> Path:
    rows = load_phase(phase)
    out = RESULTS / phase
    keys: list[str] = []
    for summary, _ in rows:
        keys += [k for k in summary if k not in keys and k != "tb"]
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for summary, _ in rows:
            writer.writerow(summary)

    lines = ["| " + " | ".join(TABLE_COLUMNS) + " |", "|" + "---|" * len(TABLE_COLUMNS)]
    lines += ["| " + " | ".join(_fmt(summary.get(c)) for c in TABLE_COLUMNS) + " |" for summary, _ in rows]
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ran = [(s, d) for s, d in rows if d is not None]
    if ran:
        _series_figure(ran, out / "series.png")
        _frames_figure(ran, out / "frames.png")
    return out


def _series_figure(ran: list[tuple[dict[str, Any], dict[str, np.ndarray]]], path: Path) -> None:
    fig, axes = plt.subplots(len(_SERIES), 1, figsize=(10, 3.0 * len(_SERIES)), sharex=True)
    cmap = plt.get_cmap("tab20")
    for idx, (summary, series) in enumerate(ran):
        ok = np.isfinite(series["rho_min"]) & (series["rho_min"] > 0.0)
        for ax, (key, _, _) in zip(axes, _SERIES, strict=True):
            values = series[key][ok]
            ax.scatter(series["t"][ok], values, s=6, color=cmap(idx % 20), label=summary["case"])
    for ax, (_, label, scale) in zip(axes, _SERIES, strict=True):
        ax.set_ylabel(label)
        ax.set_yscale(scale)
    axes[-1].set_xlabel("time step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, markerscale=2)
    fig.tight_layout(rect=(0, 0.04 + 0.012 * math.ceil(len(ran) / 4), 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _frames_figure(ran: list[tuple[dict[str, Any], dict[str, np.ndarray]]], path: Path) -> None:
    ncol = min(7, len(ran))
    nrow = math.ceil(len(ran) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 3.6 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (summary, series) in zip(axes.flat, ran, strict=False):
        ax.imshow(series["frame"].T, origin="lower", cmap="viridis")
        ax.set_title(f"{summary['case']}\n{summary['status']} t={summary.get('t_end')}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise one density-ratio study phase.")
    parser.add_argument("phase")
    args = parser.parse_args()
    out = write_outputs(args.phase)
    print((out / "table.md").read_text())


if __name__ == "__main__":
    main()
