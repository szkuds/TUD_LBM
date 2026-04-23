"""Abstract base class for plot operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.axes
import numpy as np

from config import SimulationConfig


class PlotOperator(ABC):
    """Base class for a single-panel plotting operator."""

    name: str

    def __init__(
        self, config: SimulationConfig, data_dir: str | Path | None = None
    ) -> None:
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

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        """Whether this operator has enough data to render."""
        return True
