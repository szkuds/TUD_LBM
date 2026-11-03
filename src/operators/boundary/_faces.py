"""Which cells and directions make up each domain face; registers nothing.

``front`` is ``z = nz - 1`` and ``back`` is ``z = 0``, matching the lattice's
``front_indices`` (positive ``cz``) the way ``top`` matches ``top_indices``.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice

#: Edge -> ``(axis, index along it)``.
_FACES: dict[str, tuple[int, int]] = {
    "bottom": (1, 0),
    "top": (1, -1),
    "left": (0, 0),
    "right": (0, -1),
    "back": (2, 0),
    "front": (2, -1),
}
#: Edge -> the lattice field listing the directions that point into the domain there.
_INCOMING: dict[str, str] = {
    "bottom": "top_indices",
    "top": "bottom_indices",
    "left": "right_indices",
    "right": "left_indices",
    "back": "front_indices",
    "front": "back_indices",
}


def face_axis(edge: str) -> int:
    """The spatial axis normal to *edge*.

    Raises:
        ValueError: If *edge* is not a domain face.
    """
    if edge not in _FACES:
        msg = f"Unknown edge '{edge}'"
        raise ValueError(msg)
    return _FACES[edge][0]


def face_cells(edge: str) -> tuple[slice | int, ...]:
    """The ``(x, y, z)`` index selecting the cells of *edge*."""
    axis = face_axis(edge)
    cells: list[slice | int] = [slice(None)] * 3
    cells[axis] = _FACES[edge][1]
    return tuple(cells)


def incoming_directions(lattice: Lattice, edge: str) -> list[int]:
    """The direction indices pointing into the domain at *edge*, as static Python ints."""
    face_axis(edge)
    return [int(i) for i in np.array(getattr(lattice, _INCOMING[edge]))]
