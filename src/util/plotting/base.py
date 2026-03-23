"""Abstract base class for plot operators."""

from __future__ import annotations
from abc import ABC
from abc import abstractmethod
import matplotlib.axes
import numpy as np


class PlotOperator(ABC):
    """Base class for a single-panel plotting operator."""

    name: str

    def __init__(self, config: dict) -> None:
        self.config = config

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
