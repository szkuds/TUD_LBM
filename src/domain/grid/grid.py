from typing import Tuple

import numpy as np

#TODO: This could be spilt into Grid2D and Grid3D to make it neat


class Grid(object):
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
        self.dim = len(self.shape)
        if self.dim == 2:
            self.nx, self.ny = self.shape
        if self.dim == 3:
            self.nx, self.ny, self.nz = self.shape

    def get_edges(self):
        grid = np.indices(self.shape)
        if self.dim == 2:
            edges = {
                "left": (grid[0][0, :], grid[1][0, :]),
                "right": (grid[0][-1, :], grid[1][-1, :]),
                "bottom": (grid[0][:, 0], grid[1][:, 0]),
                "top": (grid[0][:, -1], grid[1][:, -1]),
            }
            return edges
        elif self.dim == 3:
            edges = {
                "left": (grid[0][0, :, :], grid[1][0, :, :], grid[2][0, :, :]),
                "right": (grid[0][-1, :, :], grid[1][-1, :, :], grid[2][-1, :, :]),
                "bottom": (grid[0][:, 0, :], grid[1][:, 0, :], grid[2][:, 0, :]),
                "top": (grid[0][:, -1, :], grid[1][:, -1, :], grid[2][:, -1, :]),
                "front": (grid[0][:, :, 0], grid[1][:, :, 0], grid[2][:, :, 0]),
                "back": (grid[0][:, :, -1], grid[1][:, :, -1], grid[2][:, :, -1]),
            }
            return edges
        else:
            raise NotImplementedError(
                "Edge extraction for grids with dim != 2 or 3 is not implemented."
            )
