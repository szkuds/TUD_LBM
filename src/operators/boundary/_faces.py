"""Which cells and directions make up each domain face; registers nothing.

``front`` is ``z = nz - 1`` and ``back`` is ``z = 0``, matching the lattice's
``front_indices`` (positive ``cz``) the way ``top`` matches ``top_indices``.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from src.config.boundary import Edge

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice

#: Edge -> the lattice field listing the directions that point into the domain there.
_INCOMING: dict[Edge, str] = {
    Edge.BOTTOM: "top_indices",
    Edge.TOP: "bottom_indices",
    Edge.LEFT: "right_indices",
    Edge.RIGHT: "left_indices",
    Edge.BACK: "front_indices",
    Edge.FRONT: "back_indices",
}


def _face(edge: str) -> Edge:
    """*edge* as an :class:`~src.config.boundary.Edge`.

    The boundary-condition protocol passes the edge as a plain string.

    Raises:
        ValueError: If *edge* is not a domain face.
    """
    try:
        return Edge(edge)
    except ValueError:
        msg = f"Unknown edge '{edge}'"
        raise ValueError(msg) from None


def face_axis(edge: str) -> int:
    """The spatial axis normal to *edge*.

    Raises:
        ValueError: If *edge* is not a domain face.
    """
    return _face(edge).normal_axis


def face_cells(edge: str) -> tuple[slice | int, ...]:
    """The ``(x, y, z)`` index selecting the cells of *edge*."""
    face = _face(edge)
    cells: list[slice | int] = [slice(None)] * 3
    cells[face.normal_axis] = -1 if face.at_far_end else 0
    return tuple(cells)


def incoming_directions(lattice: Lattice, edge: str) -> list[int]:
    """The direction indices pointing into the domain at *edge*, as static Python ints."""
    return [int(i) for i in np.array(getattr(lattice, _INCOMING[_face(edge)]))]
