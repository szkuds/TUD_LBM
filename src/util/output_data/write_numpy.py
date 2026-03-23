import os
import numpy as np
from .base import OutputWriter


class numpy(OutputWriter):
    def save_data_step(self, iteration: int, data: dict[str, np.ndarray]) -> None:
        filename = os.path.join(self.data_dir, f"timestep_{iteration}.npz")
        np.savez(filename, **data)
