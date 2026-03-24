from abc import ABC
from abc import abstractmethod
from typing import ClassVar
import numpy as np


class OutputWriter(ABC):
    registry: ClassVar[dict[str, type["OutputWriter"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Skip base class itself
        if cls is OutputWriter:
            return
        key = cls.__name__.lower()
        # Prevent duplicate names
        if key in OutputWriter.registry:
            raise ValueError(f"Output writer '{key}' already registered.")

        OutputWriter.registry[key] = cls

    @abstractmethod
    def save_data_step(self, iteration: int, data: dict[str, np.ndarray]) -> None:
        """Save output data for a specific iteration."""
