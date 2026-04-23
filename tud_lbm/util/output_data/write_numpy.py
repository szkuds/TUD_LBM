from pathlib import Path
import numpy as np
from .base import OutputWriter


class Numpy(OutputWriter):
    def save_data_step(self, iteration: int, data: dict[str, np.ndarray]) -> None:
        filename = str(Path(self.data_dir) / f"timestep_{iteration}.npz")
        np.savez(filename, **data)
