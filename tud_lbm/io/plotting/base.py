"""Abstract base class for plot operators."""

from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    import numpy as np
    from tud_lbm.config import SimulationConfig


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

    def __init__(self, config: SimulationConfig | None = None) -> None:
        """Initialize with optional simulation config.

        Args:
            config: Simulation configuration object. Required by config-aware
                operators (for example, capillary-number or normalized
                contact-line operators). Operators that do not need config can
                ignore it.
        """
        self.config = config

    @abstractmethod
    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute time-series arrays from snapshot files."""

    @abstractmethod
    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:
        """Render the full analysis plot from precomputed arrays."""

    def update(self, ax: matplotlib.axes.Axes, files: list[Path]) -> None:
        """Render analysis for a prefix of snapshot files (animation-friendly)."""
        self.render(ax, self.compute(files))
