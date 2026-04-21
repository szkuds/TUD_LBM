"""Output writer for VTK format."""

from pathlib import Path
import numpy as np

try:
    import pyevtk
except ModuleNotFoundError:  # pragma: no cover
    pyevtk = None  # VTK writer unavailable; will raise at call time


from .base import OutputWriter

# VTK requires 3D vector fields
_VTK_VECTOR_DIMENSION = 3


class Vtk(OutputWriter):
    """Output writer that saves data in VTK format."""

    def save_data_step(self, iteration: int, data: dict[str, np.ndarray]) -> None:
        """Write output data in VTK format.

        Args:
            iteration: The iteration number.
            data: Dictionary mapping field names to numpy arrays.

        Raises:
            RuntimeError: If pyevtk is not installed.
        """
        if pyevtk is None:
            msg = "pyevtk is required for VTK output.  Install it with:  pip install pyevtk"
            raise RuntimeError(
                msg,
            )
        filename = str(Path(self.data_dir) / f"timestep_{iteration}")

        def vtk_ready(a: np.ndarray) -> np.ndarray:
            """Ensure data is in the right shape and memory layout for VTK.

            Args:
                a: Input array.

            Returns:
                Contiguous 3D array.
            """
            return np.ascontiguousarray(
                np.atleast_3d(a),
            )

        data_vtk_ready = {}
        for key, field in data.items():
            if field.shape[-1] == field.shape[-2] == 1:  # Scalar field => reshape to (Nx, Ny, Nz or 1)
                data_vtk_ready[key] = vtk_ready(field[..., 0, 0])
            elif (
                field.shape[-1] > 1
            ):  # Vector field => reshape to length 3 tuple of (Nx, Ny, Nz or 1) for each component
                vector_list = [vtk_ready(field[..., 0, j]) for j in range(field.shape[-1])]
                if (
                    len(vector_list) < _VTK_VECTOR_DIMENSION
                ):  # Add zero z-component if only 2D vector field is present (otherwise VTK will complain)
                    vector_list.append(np.zeros_like(vector_list[0]))
                data_vtk_ready[key] = tuple(vector_list)

        pyevtk.hl.imageToVTK(
            filename,
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            cellData=data_vtk_ready,
        )
