"""``[boundary_conditions]``: which BC acts on each edge, with which parameters, in which order.

A BC's parameters live in the section's key named :func:`parameter_section_key`,
e.g. ``["left_velocity_inlet"]`` for a ``"velocity-inlet"`` left edge, completed
from :data:`~src.config.defaults.BOUNDARY_PARAMETERS`. Static for a whole run, so resolved
here; the boundary operator package only looks up and binds.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from src.config._checks import registered_names
from src.config.defaults import BOUNDARY_PARAMETERS
from src.config.defaults import DEFAULT_BOUNDARY_CONDITION
from src.registry import get_operators

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig

BC_APPLICATION_ORDER: tuple[str, ...] = ("bottom", "top", "left", "right", "front", "back")
IN_PLANE_EDGES: tuple[str, ...] = ("top", "bottom", "left", "right")
OUT_OF_PLANE_EDGES: tuple[str, ...] = ("front", "back")
_PLANAR_DIM: int = 2


class BoundaryEdge(NamedTuple):
    """One edge's boundary condition, resolved from ``[boundary_conditions]``.

    Attributes:
        edge: Edge name, one of :data:`BC_APPLICATION_ORDER`.
        name: Registered boundary-condition name, e.g. ``"bounce-back"``.
        params: Keyword arguments bound into the BC operator; empty for a BC
            that takes none.
    """

    edge: str
    name: str
    params: dict[str, Any]


def parameter_section_key(edge: str, name: str) -> str:
    """Return the key holding *edge*'s parameters for BC *name*.

    Example: ``parameter_section_key("left", "velocity-inlet")`` is
    ``"left_velocity_inlet"``.
    """
    return f"{edge}_{name.replace('-', '_')}"


def lattice_edges(lattice_type: str) -> tuple[str, ...]:
    """The edges a domain on *lattice_type* has.

    A two-dimensional lattice has no ``front``/``back`` face, so those edges are
    neither completed, validated, printed nor saved for it. The dimension is the
    ``dim`` the lattice declares at its registration.

    Args:
        lattice_type: A lattice name. An unregistered one gets every edge;
            :class:`~src.config.simulation_config.SimulationConfig` rejects the
            name itself.

    Returns:
        The in-plane edges, followed by ``front``/``back`` for a 3D lattice.
    """
    # Imported for its registration side effect, as in pad_modes.
    import src.lattice.lattice  # noqa: F401

    entry = get_operators("lattice").get(lattice_type)
    if entry is not None and (entry.metadata or {}).get("dim") == _PLANAR_DIM:
        return IN_PLANE_EDGES
    return IN_PLANE_EDGES + OUT_OF_PLANE_EDGES


def is_three_dimensional(lattice_type: str) -> bool:
    """Whether *lattice_type* has ``front``/``back`` faces."""
    return set(OUT_OF_PLANE_EDGES) <= set(lattice_edges(lattice_type))


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
    edges = lattice_edges(lattice_type)
    for edge in OUT_OF_PLANE_EDGES:
        if edge not in edges and completed.get(edge) == "periodic":
            del completed[edge]
    for edge in edges:
        name = completed.setdefault(edge, DEFAULT_BOUNDARY_CONDITION)
        if isinstance(name, str) and name in BOUNDARY_PARAMETERS:
            key = parameter_section_key(edge, name)
            completed[key] = BOUNDARY_PARAMETERS[name] | dict(completed.get(key, {}))
    return completed


def validate_boundary_conditions(config: SimulationConfig) -> None:
    """Reject an unregistered BC type, and a parameter key no edge's BC reads."""
    section = _section(config)
    valid_bcs = registered_names("src.operators.boundary", "boundary_condition")
    edges = lattice_edges(config.lattice_type)
    for edge in OUT_OF_PLANE_EDGES:
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

    def _mode(edge: str) -> str:
        metadata = bc_ops[section[edge]].metadata or {}
        return str(metadata.get("pad_edge_mode", "edge"))

    modes = _mode("top"), _mode("bottom"), _mode("right"), _mode("left")
    if is_three_dimensional(config.lattice_type):
        return (*modes, _mode("front"), _mode("back"))
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
        for edge in BC_APPLICATION_ORDER
        if edge in section
    )
