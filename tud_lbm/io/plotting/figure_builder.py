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
from tud_lbm.registry import get_operators
from . import analysis as _analysis_mod  # noqa: F401

if TYPE_CHECKING:
    import os
    from tud_lbm.config import SimulationConfig

_SMALL_LAYOUTS: dict[int, tuple[int, int]] = {
    1: (1, 1),
    2: (2, 1),
    3: (2, 2),
    4: (2, 2),
}


class FigureBuilder:
    """Build and save composite figures for saved simulation snapshots."""

    _SMALL_LAYOUTS: dict[int, tuple[int, int]] = _SMALL_LAYOUTS

    def __init__(
        self,
        config: SimulationConfig,
        run_dir: str | os.PathLike,
        dpi: int = 150,
        fields: list[str] | None = None,
    ) -> None:
        """Initialize figure builder with simulation config and output directory.

        Args:
            config: Simulation configuration object.
            run_dir: Directory containing simulation results and output.
            dpi: Resolution in dots per inch for saved figures.
            fields: Explicit list of operator names to activate. When provided,
                overrides ``config.plot_fields``.
        """
        self.config = config
        self.run_dir = Path(run_dir)
        self.dpi = dpi

        self._data_dir = self.run_dir / "data"
        self._plot_dir = self.run_dir / "plots"
        self._analysis_dir = self._plot_dir / "analysis"
        self._field_operators: list = []
        self._analysis_operators: list = []
        # Backward-compatible alias used in existing tests.
        self._operators = self._field_operators

        # Guard: plotting only supports 2D simulations (nz=1)
        nz = getattr(config, "nz", config.grid_shape[2] if len(config.grid_shape) > 2 else 1)  # noqa: PLR2004
        if nz != 1:
            warnings.warn(
                f"Plotting is not supported for 3D simulations (nz={nz}). "
                "Save your output in VTK format and use ParaView to visualise results.",
                stacklevel=2,
            )
            return  # leave operator lists empty, all build() calls become no-ops

        self._plot_dir.mkdir(parents=True, exist_ok=True)

        requested = fields or self.config.plot_fields
        if not requested:
            # Default: only enable field plotting operators (density, velocity, force).
            # Analysis operators (max_velocity, contact_angles, etc.) must be explicitly
            # requested via plot_fields in the [output] section of the config.
            requested = list(get_operators("plotting").keys())

        all_ops = get_operators("plotting")
        all_analysis_ops = get_operators("analysis")
        for name in requested:
            entry = all_ops.get(name)
            if entry is not None:
                self._field_operators.append(entry.target(self.config, data_dir=self._data_dir))
                continue

            analysis_entry = all_analysis_ops.get(name)
            if analysis_entry is not None:
                self._analysis_operators.append(analysis_entry.target())
                continue

            known = sorted(set(all_ops.keys()) | set(all_analysis_ops.keys()))
            warnings.warn(
                f"No plot operator registered for '{name}'. Available: {known}",
                stacklevel=2,
            )

    def _sorted_timed_files(self) -> list[tuple[int, Path]]:
        timed_files: list[tuple[int, Path]] = []
        for fp in self._data_dir.glob("*.npz"):
            timestep = self._extract_timestep(fp.stem)
            if timestep is not None:
                timed_files.append((timestep, fp))
        timed_files.sort(key=lambda item: item[0])
        return timed_files

    @property
    def data_dir(self) -> Path:
        """Directory containing timestep ``.npz`` files."""
        return self._data_dir

    @property
    def plot_dir(self) -> Path:
        """Directory where generated plots are written."""
        return self._plot_dir

    @property
    def field_operators(self) -> list:
        """Configured field plot operators."""
        return self._field_operators

    @property
    def analysis_operators(self) -> list:
        """Configured analysis plot operators."""
        return self._analysis_operators

    def sorted_timed_files(self) -> list[tuple[int, Path]]:
        """Return ``(timestep, path)`` pairs sorted by timestep."""
        return self._sorted_timed_files()

    def build_analysis(self) -> list[Path]:
        """Build standalone analysis figures from all saved snapshots."""
        if not self._analysis_operators or not self._data_dir.exists():
            return []

        files = [fp for _, fp in self._sorted_timed_files()]
        if not files:
            return []

        self._analysis_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for op in self._analysis_operators:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), squeeze=False)
            try:
                precomputed = op.compute(files)
                op.render(ax[0][0], precomputed)
            except Exception as exc:  # noqa: BLE001
                ax[0][0].set_title(f"{op.name} - ERROR")
                ax[0][0].text(
                    0.5,
                    0.5,
                    str(exc),
                    ha="center",
                    va="center",
                    transform=ax[0][0].transAxes,
                    fontsize=8,
                    color="red",
                )
            plt.tight_layout()
            out_path = self._analysis_dir / f"{op.name}.png"
            fig.savefig(out_path, dpi=self.dpi)
            plt.close(fig)
            saved.append(out_path)
        return saved

    def render_figure(
        self,
        data: dict[str, np.ndarray],
        timestep: int,
        history_files: list[Path] | None = None,
    ) -> plt.Figure | None:
        """Render one timestep into a Figure without saving it."""
        field_ops = [op for op in self._field_operators if op.is_available(data)]
        analysis_ops = list(self._analysis_operators) if history_files is not None else []
        panels = field_ops + analysis_ops

        if not panels:
            return None

        ncols, nrows = self._layout(len(panels))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )

        for idx, op in enumerate(field_ops):
            row, col = divmod(idx, ncols)
            try:
                op(axes[row][col], data, timestep)
            except Exception as exc:  # noqa: BLE001
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

        offset = len(field_ops)
        for idx, op in enumerate(analysis_ops):
            row, col = divmod(offset + idx, ncols)
            try:
                op.update(axes[row][col], history_files)
            except Exception as exc:  # noqa: BLE001
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

        for idx in range(len(panels), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        title = self.config.simulation_name or "simulation"
        fig.suptitle(f"{title} - Timestep {timestep}", fontsize=12)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        return fig

    def build(
        self,
        data: dict[str, np.ndarray],
        timestep: int,
        filename: str | None = None,
    ) -> Path | None:
        """Render one timestep figure and save it to disk."""
        fig = self.render_figure(data, timestep)
        if fig is None:
            warnings.warn(
                f"FigureBuilder: no operators have data at t={timestep}.",
                stacklevel=2,
            )
            return None

        out_name = filename or f"timestep_{timestep}.png"
        out_path = self._plot_dir / out_name
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path

    def build_all(self, skip: int = 0) -> list[Path]:
        """Build figures for every saved timestep file under data/."""
        if not self._data_dir.exists():
            return []

        files = [fp for _, fp in self._sorted_timed_files()]
        saved: list[Path] = []
        for fp in files[skip:]:
            timestep = self._extract_timestep(fp.stem)
            if timestep is None:
                continue
            with np.load(fp) as raw:
                data = {key: raw[key] for key in raw.files}
            path = self.build(data, timestep)
            if path is not None:
                saved.append(path)

        self.build_analysis()
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

    @staticmethod
    def layout(n: int) -> tuple[int, int]:
        """Public wrapper for selecting a compact subplot layout."""
        return FigureBuilder._layout(n)
