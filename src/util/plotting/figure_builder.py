"""Build composite per-timestep figures from registered plot operators."""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from registry import get_operators

_DEFAULT_FIELD_ORDER = ["density", "velocity", "force", "force_ext", "analysis"]


class FigureBuilder:
    """Build and save composite figures for saved simulation snapshots."""

    def __init__(self, config: dict, run_dir: str | os.PathLike, dpi: int = 150) -> None:
        self.config = dict(config or {})
        self.run_dir = Path(run_dir)
        self.dpi = dpi

        self._data_dir = self.run_dir / "data"
        self._plot_dir = self.run_dir / "plots"
        self._plot_dir.mkdir(parents=True, exist_ok=True)

        self.config["data_dir"] = str(self._data_dir)
        requested = self.config.get("plot_fields")
        requested = requested or _DEFAULT_FIELD_ORDER

        all_ops = get_operators('plotting')
        self._operators = []
        for name in requested:
            entry = all_ops.get(name)
            if entry is None:
                warnings.warn(
                    f"No plot operator registered for '{name}'. Available: {list(all_ops.keys())}",
                    stacklevel=2,
                )
                continue
            self._operators.append(entry.target(self.config))

    def build(
        self,
        data: dict[str, np.ndarray],
        timestep: int,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
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
            except Exception as exc:  # pragma: no cover
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

        title = self.config.get("plot_title") or self.config.get("simulation_name") or "simulation"
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

        files = sorted(
            self._data_dir.glob("*.npz"),
            key=lambda path: self._extract_timestep(path.stem),
        )
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
    def _extract_timestep(stem: str) -> Optional[int]:
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return None

    @staticmethod
    def _layout(n: int) -> tuple[int, int]:
        """Choose a compact subplot layout for *n* panels."""
        if n <= 1:
            return 1, 1
        if n <= 2:
            return 2, 1
        if n <= 4:
            return 2, 2
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        return ncols, nrows

