import os
import numpy as np

try:
    import pyevtk
except ModuleNotFoundError:  # pragma: no cover
    pyevtk = None  # VTK writer unavailable; will raise at call time


from .base import OutputWriter


class vtk(OutputWriter):
    def save_data_step(self, iteration: int, data: dict[str, np.ndarray]) -> None:
        """Write output data in VTK format."""
        if pyevtk is None:
            raise RuntimeError(
                "pyevtk is required for VTK output.  Install it with:  pip install pyevtk",
            )
        filename = os.path.join(self.data_dir, f"timestep_{iteration}")

        vtk_ready = lambda a: np.ascontiguousarray(
            np.atleast_3d(a),
        )  # Ensure data is in the right shape and memory layout for VTK
        data_vtk_ready = {}
        for key, field in data.items():
            if field.shape[-1] == field.shape[-2] == 1:  # Scalar field => reshape to (Nx, Ny, Nz or 1)
                data_vtk_ready[key] = vtk_ready(field[..., 0, 0])
            elif (
                field.shape[-1] > 1
            ):  # Vector field => reshape to length 3 tuple of (Nx, Ny, Nz or 1) for each component
                vector_list = [vtk_ready(field[..., 0, j]) for j in range(field.shape[-1])]
                if (
                    len(vector_list) < 3
                ):  # Add zero z-component if only 2D vector field is present (otherwise VTK will complain)
                    vector_list.append(np.zeros_like(vector_list[0]))
                data_vtk_ready[key] = tuple(vector_list)
            # elif field.shape[-2] > 1:  # Population field => reshape to (Q, Nx, Ny, Nz or 1)
            #   data_vtk_ready[key] = tuple([vtk_ready(field[..., j, 0]) for j in range(field.shape[-2])])

        pyevtk.hl.imageToVTK(
            filename,
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            cellData=data_vtk_ready,
        )
