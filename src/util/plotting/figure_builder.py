"""Build composite per-timestep figures from registered plot operators."""

from __future__ import annotations
import math
import warnings
from pathlib import Path
import matplotlib as mpl

mpl.use("Agg")

from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from registry import get_operators

if TYPE_CHECKING:
    import os
    from config import SimulationConfig

_SMALL_LAYOUTS: dict[int, tuple[int, int]] = {
    1: (1, 1),
    2: (2, 1),
    3: (2, 2),
    4: (2, 2),
}


class FigureBuilder:
    """Build and save composite figures for saved simulation snapshots."""

    _SMALL_LAYOUTS: dict[int, tuple[int, int]] = _SMALL_LAYOUTS

    def __init__(self, config: SimulationConfig, run_dir: str | os.PathLike, dpi: int = 150) -> None:
        """Initialize figure builder with simulation config and output directory.

        Args:
            config: Simulation configuration object.
            run_dir: Directory containing simulation results and output.
            dpi: Resolution in dots per inch for saved figures.
        """
        self.config = config
        self.run_dir = Path(run_dir)
        self.dpi = dpi

        self._data_dir = self.run_dir / "data"
        self._plot_dir = self.run_dir / "plots"
        self._plot_dir.mkdir(parents=True, exist_ok=True)

        requested = self.config.plot_fields
        if not requested:
            requested = list(get_operators("plotting").keys())

        all_ops = get_operators("plotting")
        self._operators: list = []
        for name in requested:
            entry = all_ops.get(name)
            if entry is None:
                warnings.warn(
                    f"No plot operator registered for '{name}'. Available: {list(all_ops.keys())}",
                    stacklevel=2,
                )
                continue
            self._operators.append(entry.target(self.config, data_dir=self._data_dir))

    def build(
        self,
        data: dict[str, np.ndarray],
        timestep: int,
        filename: str | None = None,
    ) -> Path | None:
        """Render one timestep figure and save it to disk."""
        active_ops = [op for op in self._operators if op.is_available(data)]
        if not active_ops:
            warnings.warn(
                f"FigureBuilder: no operators have data at t={timestep}.",
                stacklevel=2,
            )
            return None

        ncols, nrows = self._layout(len(active_ops))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )

        for idx, op in enumerate(active_ops):
            row, col = divmod(idx, ncols)
            try:
                op(axes[row][col], data, timestep)
            except Exception as exc:  ## noqa: BLE001
                axes[row][col].set_title(f"{op.name} - ERROR")
                axes[row][col].text(
                    0.5,
                    0.5,
                    str(exc),
                    ha="center",
                    va="center",
                    transform=axes[row][col].transAxes,
                    fontsize=7,
                    color="red",
                )

        for idx in range(len(active_ops), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        title = self.config.simulation_name or "simulation"
        fig.suptitle(f"{title} - Timestep {timestep}", fontsize=12)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        out_name = filename or f"timestep_{timestep}.png"
        out_path = self._plot_dir / out_name
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path

    def build_all(self, skip: int = 0) -> list[Path]:
        """Build figures for every saved timestep file under data/."""
        if not self._data_dir.exists():
            return []

        timed_files: list[tuple[int, Path]] = []
        for fp in self._data_dir.glob("*.npz"):
            timestep = self._extract_timestep(fp.stem)
            if timestep is not None:
                timed_files.append((timestep, fp))

        timed_files.sort(key=lambda item: item[0])
        files = [fp for _, fp in timed_files]
        saved: list[Path] = []
        for fp in files[skip:]:
            timestep = self._extract_timestep(fp.stem)
            if timestep is None:
                continue
            raw = np.load(fp)
            data = {key: raw[key] for key in raw.files}
            path = self.build(data, timestep)
            if path is not None:
                saved.append(path)
        return saved

    @staticmethod
    def _extract_timestep(stem: str) -> int | None:
        try:
            return int(stem.rsplit("_", maxsplit=1)[-1])
        except ValueError:
            return None

    @staticmethod
    def _layout(n: int) -> tuple[int, int]:
        """Choose a compact subplot layout for *n* panels."""
        if layout := FigureBuilder._SMALL_LAYOUTS.get(n):
            return layout
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        return ncols, nrows
