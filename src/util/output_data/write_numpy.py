import numpy as np
from typing import Dict
import os

from .base import OutputWriter


class numpy(OutputWriter):

    def save_data_step(self, iteration: int, data: Dict[str, np.ndarray]) -> None:
        filename = os.path.join(self.data_dir, f"timestep_{iteration}.npz")
        np.savez(filename, **data)
