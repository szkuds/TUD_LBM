"""Analysis plot operator for saved simulation history."""

from __future__ import annotations
from pathlib import Path
import matplotlib.axes
import numpy as np
from registry import plotting_operator
from util.plotting.base import PlotOperator


@plotting_operator(name="analysis")
class AnalysisPlotOperator(PlotOperator):
    """Render simple diagnostic histories derived from saved snapshots."""

    name = "analysis"

    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        del data
        data_dir = Path(self.config.get("data_dir", "."))
        files = sorted(data_dir.glob("*.npz"))

        iters: list[int] = []
        umax_vals: list[float] = []
        avg_rho_vals: list[float] = []
        ratio_vals: list[float] = []

        for fp in files:
            stem = fp.stem
            try:
                step = int(stem.split("_")[-1])
            except ValueError:
                continue

            snapshot = np.load(fp)
            if "rho" not in snapshot or "u" not in snapshot:
                continue

            rho = np.asarray(snapshot["rho"])
            u = np.asarray(snapshot["u"])
            vel_mag = np.sqrt(u[..., 0, 0] ** 2 + u[..., 0, 1] ** 2)
            min_rho = float(np.min(rho))
            safe_min = min_rho if min_rho > 0 else max(min_rho, 1e-30)

            iters.append(step)
            umax_vals.append(float(np.max(vel_mag)))
            avg_rho_vals.append(float(np.mean(rho)))
            ratio_vals.append(float(np.max(rho)) / safe_min if safe_min != 0 else np.inf)

        if not iters:
            ax.text(
                0.5,
                0.5,
                "No data yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Analysis")
            ax.set_xlabel("Timestep")
            return

        ax2 = ax.twinx()
        ax.scatter(iters, umax_vals, s=10, color="tab:blue", label="max_u")
        ax.scatter(iters, avg_rho_vals, s=10, color="tab:green", label="avg_rho")
        ax2.scatter(iters, ratio_vals, s=10, color="tab:orange", label="rho max/min")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("max_u / avg_rho")
        ax2.set_ylabel("Density ratio")
        ax.set_title(f"Diagnostics  t={timestep}")
        ax.grid(True, alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")
