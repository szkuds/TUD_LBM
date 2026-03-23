from abc import ABC
from abc import abstractmethod
import numpy as np


class OutputWriter(ABC):
    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Skip base class itself
        if cls is OutputWriter:
            return

        # # --- Check write_output signature ---
        # base_sig = inspect.signature(OutputWriter.write_output)
        # sub_sig = inspect.signature(cls.write_output)

        # if base_sig != sub_sig:
        #     raise TypeError(
        #         f"{cls.__name__}.write_output has wrong signature.\n"
        #         f"Expected: {base_sig}\n"
        #         f"Got: {sub_sig}"
        #     )

        # Prevent duplicate names
        if cls.__name__ in OutputWriter.registry:
            raise ValueError(f"Output writer '{cls.__name__}' already registered.")

        OutputWriter.registry[cls.__name__] = cls

    @abstractmethod
    def save_data_step(self, iteration: int, data: dict[str, np.ndarray]) -> None:
        """Save output data for a specific iteration."""
