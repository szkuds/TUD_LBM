"""``[boundary_conditions]``: which BC acts on each edge, with which parameters, in which order.

A BC's parameters live in the section's key named :func:`parameter_section_key`,
e.g. ``["left_velocity_inlet"]`` for a ``"velocity-inlet"`` left edge, completed
from :data:`~src.config.defaults.BOUNDARY_PARAMETERS`. Static for a whole run, so resolved
here; the boundary operator package only looks up and binds.
"""

from __future__ import annotations
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from src.config._checks import registered_names
from src.config.defaults import BOUNDARY_PARAMETERS
from src.config.defaults import DEFAULT_BOUNDARY_CONDITION
from src.registry import get_operators

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig


class Edge(StrEnum):
    """A face of the domain, declared in the order the boundary conditions are applied.

    A member equals its string (``Edge.LEFT == "left"``), so it indexes the
    ``[boundary_conditions]`` table and builds its keys unchanged.
    """

    BOTTOM = "bottom"
    TOP = "top"
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"

    @property
    def normal_axis(self) -> int:
        """The spatial axis normal to this face: 0 (x), 1 (y) or 2 (z)."""
        return _FACE[self][0]

    @property
    def at_far_end(self) -> bool:
        """Whether this face is the last index of its normal axis (``top``, ``right``, ``front``)."""
        return _FACE[self][1]

    @property
    def cells(self) -> tuple[slice | int, ...]:
        """The ``(x, y, z)`` index selecting the cells of this face."""
        cells: list[slice | int] = [slice(None)] * 3
        cells[self.normal_axis] = -1 if self.at_far_end else 0
        return tuple(cells)

    @property
    def tangential_ends(self) -> tuple[Edge, Edge]:
        """The two in-plane faces that bound this in-plane face, at its start and its end."""
        return (Edge.BOTTOM, Edge.TOP) if self.normal_axis == 0 else (Edge.LEFT, Edge.RIGHT)


#: Edge -> (normal axis, at the far end of it). ``front`` is ``z = nz - 1``.
_FACE: dict[Edge, tuple[int, bool]] = {
    Edge.BOTTOM: (1, False),
    Edge.TOP: (1, True),
    Edge.LEFT: (0, False),
    Edge.RIGHT: (0, True),
    Edge.BACK: (2, False),
    Edge.FRONT: (2, True),
}


def edges_of(dim: int) -> tuple[Edge, ...]:
    """The faces of a *dim*-dimensional domain, in application order."""
    return tuple(edge for edge in Edge if edge.normal_axis < dim)


class BoundaryEdge(NamedTuple):
    """One edge's boundary condition, resolved from ``[boundary_conditions]``.

    Attributes:
        edge: The face this boundary condition acts on.
        name: Registered boundary-condition name, e.g. ``"bounce-back"``.
        params: Keyword arguments bound into the BC operator; empty for a BC
            that takes none.
    """

    edge: Edge
    name: str
    params: dict[str, Any]


def parameter_section_key(edge: str, name: str) -> str:
    """Return the key holding *edge*'s parameters for BC *name*.

    Example: ``parameter_section_key("left", "velocity-inlet")`` is
    ``"left_velocity_inlet"``.
    """
    return f"{edge}_{name.replace('-', '_')}"


def lattice_dim(lattice_type: str) -> int:
    """The number of spatial dimensions *lattice_type* declares at its registration.

    An unregistered name counts as three-dimensional, so every edge is kept;
    :class:`~src.config.simulation_config.SimulationConfig` rejects the name itself.
    """
    # Imported for its registration side effect, as in pad_modes.
    import src.lattice.lattice  # noqa: F401

    entry = get_operators("lattice").get(lattice_type)
    dim = (entry.metadata or {}).get("dim") if entry is not None else None
    return dim if isinstance(dim, int) else len(Edge) // 2


def complete_boundary_conditions(section: dict[str, Any] | None, lattice_type: str) -> dict[str, Any]:
    """*section* naming every edge of the lattice, so the boundary builder only looks up and binds.

    A missing edge takes the default BC. A two-dimensional lattice has no
    ``front``/``back`` face, so a periodic entry for one is dropped rather than
    carried into the overview and the saved ``config.toml``; run directories
    written before this still load. Any other BC there is left for
    :func:`validate_boundary_conditions` to reject. An edge whose BC takes
    parameters gets its ``{edge}_{name}`` key, the defaults under what it sets.
    """
    completed = dict(section) if section is not None else {}
    edges = edges_of(lattice_dim(lattice_type))
    for edge in set(Edge) - set(edges):
        if completed.get(edge) == "periodic":
            del completed[edge]
    for edge in edges:
        # A plain-string key, so the table saved to ``config.toml`` holds no enum members.
        name = completed.setdefault(edge.value, DEFAULT_BOUNDARY_CONDITION)
        if isinstance(name, str) and name in BOUNDARY_PARAMETERS:
            key = parameter_section_key(edge, name)
            completed[key] = BOUNDARY_PARAMETERS[name] | dict(completed.get(key, {}))
    return completed


def validate_boundary_conditions(config: SimulationConfig) -> None:
    """Reject an unregistered BC type, and a parameter key no edge's BC reads."""
    section = _section(config)
    valid_bcs = registered_names("src.operators.boundary", "boundary_condition")
    edges = edges_of(lattice_dim(config.lattice_type))
    for edge in Edge:
        if edge not in edges and edge in section:
            msg = (
                f"boundary_conditions['{edge}'] has no face on the {config.lattice_type} lattice, got '{section[edge]}'"
            )
            raise ValueError(msg)
    for edge in edges:
        if section[edge] not in valid_bcs:
            msg = f"boundary_conditions['{edge}'] must be one of {sorted(valid_bcs)}, got '{section[edge]}'"
            raise ValueError(msg)
    parameter_keys = {parameter_section_key(edge, section[edge]) for edge in edges}
    for key in sorted(section.keys() - set(edges) - parameter_keys):
        msg = f"boundary_conditions['{key}'] is not the parameter section of any edge's boundary condition"
        raise ValueError(msg)


def _section(config: SimulationConfig) -> dict[str, Any]:
    """The completed section; ``SimulationConfig.__post_init__`` always fills it in."""
    section = config.boundary_conditions
    assert section is not None  # noqa: S101 - complete_boundary_conditions fills it in
    return section


def periodic_axes(config: SimulationConfig) -> tuple[bool, bool, bool]:
    """Per-axis periodicity ``(x, y, z)``: an axis is periodic only if both its edges are.

    A 2D lattice has no ``front``/``back`` faces, so nothing bounds its z axis.
    """
    section = _section(config)
    return (
        section["left"] == "periodic" and section["right"] == "periodic",
        section["bottom"] == "periodic" and section["top"] == "periodic",
        section.get("front", "periodic") == "periodic" and section.get("back", "periodic") == "periodic",
    )


def pad_modes(config: SimulationConfig) -> tuple[str, ...]:
    """The stencil pad mode of each edge.

    Each edge's BC is looked up in the ``"boundary_condition"`` registry and its
    ``pad_edge_mode`` metadata is used, ``"edge"`` when a BC declares none. The
    fallback does not raise on an unpopulated registry, it pads a periodic run as
    walled, so the package is imported for its registration side effect.

    The order is ``(top, bottom, right, left)``, i.e.
    ``(y_end, y_start, x_end, x_start)``, which is what
    :func:`~src.operators.differential._pad_utils._apply_stencil_padding` expects;
    a 3D lattice appends ``(front, back)``, i.e. ``(z_end, z_start)``.

    Returns:
        Four ``jnp.pad`` mode strings, six on a 3D lattice.
    """
    import src.operators.boundary  # noqa: F401

    section = _section(config)
    bc_ops = get_operators("boundary_condition")

    def _mode(edge: Edge) -> str:
        metadata = bc_ops[section[edge]].metadata or {}
        return str(metadata.get("pad_edge_mode", "edge"))

    modes = _mode(Edge.TOP), _mode(Edge.BOTTOM), _mode(Edge.RIGHT), _mode(Edge.LEFT)
    if Edge.FRONT in section:
        return (*modes, _mode(Edge.FRONT), _mode(Edge.BACK))
    return modes


def boundary_edges(config: SimulationConfig) -> tuple[BoundaryEdge, ...]:
    """Each edge's boundary condition and its parameters, in application order."""
    section = _section(config)
    return tuple(
        BoundaryEdge(
            edge=edge,
            name=section[edge],
            params=dict(section.get(parameter_section_key(edge, section[edge]), {})),
        )
        for edge in Edge
        if edge in section
    )
