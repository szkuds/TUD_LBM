"""Abstract base class for plot operators."""

from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import numpy as np
    from src.config import SimulationConfig


class PlotOperator(ABC):
    """Base class for a single-panel plotting operator."""

    name: str

    def __init__(self, config: SimulationConfig, data_dir: str | Path | None = None) -> None:
        """Initialize the plot operator with config and optional data directory.

        Args:
            config: Simulation configuration object.
            data_dir: Optional path to data directory for loading snapshots.
        """
        self.config = config
        self.data_dir = Path(data_dir) if data_dir is not None else None

    @abstractmethod
    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        """Draw this operator on the provided axes."""

    def is_available(self, data: dict[str, np.ndarray]) -> bool:  # noqa: ARG002
        """Whether this operator has enough data to render."""
        return True


class AnalysisPlot(ABC):
    """Base class for analysis plots computed from saved snapshot history."""

    name: str
    is_multi_panel: bool = False

    def __init__(self, config: SimulationConfig | None = None) -> None:
        """Initialize with optional simulation config.

        Args:
            config: Simulation configuration object. Required by config-aware
                operators (for example, capillary-number or normalized
                contact-line operators). Operators that do not need config can
                ignore it.
        """
        self.config = config
        self._primed_xlim: tuple[float, float] | None = None
        self._primed_ylims: list[tuple[float, float]] | None = None

    @abstractmethod
    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute time-series arrays from snapshot files."""

    @abstractmethod
    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:
        """Render the full analysis plot from precomputed arrays."""

    def render_figure(self, files: list[Path]) -> matplotlib.figure.Figure:
        """Build and return a complete standalone figure.

        Only called when ``is_multi_panel`` is True; overrides the
        compute()/render(ax) flow used by single-panel analysis plots.
        """
        raise NotImplementedError

    def prime(self, files: list[Path]) -> None:
        """Cache axis limits from the full dataset so animation frames use fixed axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            self.render(ax, self.compute(files))
            self._primed_xlim = ax.get_xlim()
            self._primed_ylims = [a.get_ylim() for a in fig.axes]
        finally:
            plt.close(fig)

    def update(self, ax: matplotlib.axes.Axes, files: list[Path]) -> None:
        """Render a prefix of snapshot files with fixed (primed) axis limits."""
        self.render(ax, self.compute(files))
        if self._primed_xlim is not None:
            ax.set_xlim(self._primed_xlim)
        if self._primed_ylims:
            ax.set_ylim(self._primed_ylims[0])
            if len(self._primed_ylims) > 1:
                # Apply twin-axis y-limits (e.g. dual-axis Ca/θ plots use twinx())
                siblings = ax.get_shared_x_axes().get_siblings(ax)
                twins = [a for a in siblings if a is not ax]
                for i, twin in enumerate(twins, start=1):
                    if i < len(self._primed_ylims):
                        twin.set_ylim(self._primed_ylims[i])
