"""Output writer for NumPy format (.npz files)."""

from pathlib import Path
import numpy as np
from .base import OutputWriter


class Numpy(OutputWriter):
    """Output writer that saves data in NumPy .npz format."""

    def save_data_step(self, iteration: int, data: dict[str, np.ndarray]) -> None:
        """Save output data as NumPy .npz file.

        Args:
            iteration: The iteration number.
            data: Dictionary mapping field names to numpy arrays.
        """
        filename = str(Path(self.data_dir) / f"timestep_{iteration}.npz")
        np.savez(filename, **data)  # ty: ignore[invalid-argument-type]
