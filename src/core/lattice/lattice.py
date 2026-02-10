import numpy as np
import re
from numpy import ndarray


class Lattice(object):

    def __init__(self, name: str) -> None:
        self.name: str = name
        dq = re.findall(r"\d+", name)
        self.d: int = int(dq[0])
        self.q: int = int(dq[1])

        # Construct the properties of a lattice
        self.c: ndarray = self.construct_lattice_velocities
        self.w: ndarray = self.construct_lattice_weigths
        self.opp_indices: ndarray = self.construct_opposite_indices
        self.main_indices: ndarray = self.construct_main_indices
        self.right_indices: ndarray = self.construct_right_indices
        self.left_indices: ndarray = self.construct_left_indices
        self.top_indices: ndarray = self.construct_top_indices
        self.bottom_indices: ndarray = self.construct_bottom_indices
        if self.d == 3:
            self.front_indices: ndarray = self.construct_front_indices
            self.back_indices: ndarray = self.construct_back_indices

    @property
    def construct_lattice_velocities(self) -> ndarray:
        if self.name == "D2Q9":
            cx = [0, 1, 0, -1, 0, 1, -1, -1, 1]
            cy = [0, 0, 1, 0, -1, 1, 1, -1, -1]
            c = np.array(tuple(zip(cx, cy)))
        else:
            raise ValueError(
                "Lattice not supported, D2Q9 is currently the only supported lattice."
            )

        return c.T

    @property
    def construct_lattice_weigths(self) -> ndarray:
        if self.name == "D2Q9":
            w = np.array(
                [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
            )
        elif self.name == "D3Q15":
            raise NotImplementedError("Dimension larger than 2 not supported.")
        else:
            raise ValueError(
                "Lattice not supported, D2Q9 is currently the only supported lattice."
            )

        return w

    @property
    def construct_opposite_indices(self) -> ndarray | None:
        c = self.c.T
        if self.d == 2:
            return np.array([c.tolist().index((-c[i]).tolist()) for i in range(self.q)])
        if self.d == 3:
            raise NotImplementedError("Dimension larger than 2 not supported.")

    @property
    def construct_main_indices(self) -> ndarray | None:
        c = self.c.T
        if self.d == 2:
            return np.nonzero((np.abs(c[:, 0]) + np.abs(c[:, 1]) == 1))[0]
        if self.d == 3:
            raise NotImplementedError("Dimension larger than 2 not supported.")

    @property
    def construct_right_indices(self) -> ndarray | None:
        c = self.c.T
        if self.d == 2:
            return np.nonzero(np.array(c[:, 0] == 1))[0]
        if self.d == 3:
            raise NotImplementedError("Dimension larger than 2 not supported.")

    @property
    def construct_left_indices(self) -> ndarray | None:
        c = self.c.T
        if self.d == 2:
            return np.nonzero(np.array(c[:, 0] == -1))[0]
        if self.d == 3:
            raise NotImplementedError("Dimension larger than 2 not supported.")

    @property
    def construct_top_indices(self) -> ndarray | None:
        c = self.c.T
        if self.d == 2:
            return np.nonzero(np.array(c[:, 1] == 1))[0]
        if self.d == 3:
            raise NotImplementedError("Dimension larger than 2 not supported.")

    @property
    def construct_bottom_indices(self) -> ndarray | None:
        c = self.c.T
        if self.d == 2:
            return np.nonzero(np.array(c[:, 1] == -1))[0]
        if self.d == 3:
            raise NotImplementedError("Dimension larger than 2 not supported.")

    @property
    def construct_front_indices(self) -> ndarray | None:
        c = self.c.T
        if self.d == 2:
            raise ValueError("Only have front indices in 3D.")
        if self.d == 3:
            raise NotImplementedError("Dimension larger than 2 not supported.")

    @property
    def construct_back_indices(self) -> ndarray | None:
        c = self.c.T
        if self.d == 2:
            raise ValueError("Only have front indices in 3D.")
        if self.d == 3:
            raise NotImplementedError("Dimension larger than 2 not supported.")
