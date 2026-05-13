"""Base class for output data writers."""

from abc import ABC
from abc import abstractmethod
from typing import ClassVar
import numpy as np


class OutputWriter(ABC):
    """Abstract base class for output writer implementations."""

    registry: ClassVar[dict[str, type["OutputWriter"]]] = {}
    data_dir: str

    def __init_subclass__(cls, **kwargs: dict) -> None:
        """Register OutputWriter subclasses in the registry.

        Args:
            **kwargs: Keyword arguments passed to parent class.
        """
        super().__init_subclass__(**kwargs)

        # Skip base class itself
        if cls is OutputWriter:
            return
        key = cls.__name__.lower()
        # Prevent duplicate names
        if key in OutputWriter.registry:
            msg = f"Output writer '{key}' already registered."
            raise ValueError(msg)

        OutputWriter.registry[key] = cls

    @abstractmethod
    def save_data_step(self, iteration: int, data: dict[str, np.ndarray]) -> None:
        """Save output data for a specific iteration."""
