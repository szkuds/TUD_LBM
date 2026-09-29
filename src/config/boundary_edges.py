"""Boundary edges — the per-edge boundary conditions of a configuration.

Built only by :attr:`SimulationConfig.boundary_edges
<src.config.simulation_config.SimulationConfig.boundary_edges>`: which BC acts
on each edge, with which parameters and in which order, is fixed by the
configuration, so the boundary operator package only looks up and binds.

A BC's parameters live in the ``bc_config`` section named
:func:`parameter_section_key`, e.g. ``bc_config["left_velocity_inlet"]`` for a
``"velocity-inlet"`` left edge; a BC without such a section runs on its own
defaults.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from src.registry import get_operators

if TYPE_CHECKING:
    from collections.abc import Mapping

BC_APPLICATION_ORDER: tuple[str, ...] = ("bottom", "top", "left", "right")


class BoundaryEdge(NamedTuple):
    """One edge's boundary condition, resolved from ``bc_config``.

    Attributes:
        edge: Edge name, one of :data:`BC_APPLICATION_ORDER`.
        name: Registered boundary-condition name, e.g. ``"bounce-back"``.
        params: Keyword arguments bound into the BC operator; empty when the
            configuration has no ``{edge}_{name}`` section.
    """

    edge: str
    name: str
    params: dict[str, Any]


def parameter_section_key(edge: str, name: str) -> str:
    """Return the ``bc_config`` key holding *edge*'s parameters for BC *name*.

    Example: ``parameter_section_key("left", "velocity-inlet")`` is
    ``"left_velocity_inlet"``.
    """
    return f"{edge}_{name.replace('-', '_')}"


def periodic_axes(bc_config: Mapping[str, Any]) -> tuple[bool, bool]:
    """Per-axis periodicity ``(x, y)`` of a completed *bc_config*.

    An axis is periodic only if both its edges are ``"periodic"``.
    """
    return (
        bc_config["left"] == "periodic" and bc_config["right"] == "periodic",
        bc_config["bottom"] == "periodic" and bc_config["top"] == "periodic",
    )


def pad_modes(bc_config: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """The stencil pad mode of each edge of a completed *bc_config*.

    Each edge's BC is looked up in the ``"boundary_condition"`` registry and its
    ``pad_edge_mode`` metadata is used, ``"edge"`` when a BC declares none.

    The order is ``(top, bottom, right, left)``, i.e.
    ``(y_end, y_start, x_end, x_start)``, which is what
    :func:`~src.operators.differential._pad_utils._apply_stencil_padding` expects.

    Args:
        bc_config: A ``bc_config`` that names every edge.

    Returns:
        Four ``jnp.pad`` mode strings.
    """
    # The per-edge fallback is "edge", so an unpopulated registry would not raise
    # — it would silently pad a periodic run as if it were walled. Import the
    # package for its registration side effect first, as the eos/pressure kinds
    # do. src.simulation_io.analysis.wetting_overlay reaches here without it, and
    # would otherwise draw a band the solver never applied.
    import src.operators.boundary  # noqa: F401

    bc_ops = get_operators("boundary_condition")

    def _mode(edge: str) -> str:
        metadata = bc_ops[bc_config[edge]].metadata or {}
        return str(metadata.get("pad_edge_mode", "edge"))

    return (_mode("top"), _mode("bottom"), _mode("right"), _mode("left"))


def build_boundary_edges(bc_config: Mapping[str, Any]) -> tuple[BoundaryEdge, ...]:
    """Resolve a completed *bc_config* into its edges, in application order.

    Args:
        bc_config: A ``bc_config`` that names every edge, as
            :class:`~src.config.simulation_config.SimulationConfig` guarantees.

    Returns:
        One :class:`BoundaryEdge` per edge of :data:`BC_APPLICATION_ORDER`.
    """
    return tuple(
        BoundaryEdge(
            edge=edge,
            name=bc_config[edge],
            params=dict(bc_config.get(parameter_section_key(edge, bc_config[edge]), {})),
        )
        for edge in BC_APPLICATION_ORDER
    )
