"""Build-time edge configuration for wetting boundaries.

Determines which edges need wetting and whether the edges bounding them are periodic.
"""

from __future__ import annotations
from src.config.boundary import Edge
from src.config.boundary import edges_of


def _resolve_wetting_edges(
    boundary_conditions: dict[str, str],
) -> list[tuple[Edge, bool, bool]]:
    """Determine wetting edges and whether their perpendicular BCs are periodic.

    Returns:
        List of ``(edge, perp_start_periodic, perp_end_periodic)``.
    """
    edges = []
    # The in-plane edges, in application order; the first wetting one is measured.
    for edge in edges_of(2):
        if boundary_conditions.get(edge) != "wetting":
            continue
        perp_start, perp_end = edge.tangential_ends
        edges.append(
            (
                edge,
                boundary_conditions.get(perp_start, "periodic") == "periodic",
                boundary_conditions.get(perp_end, "periodic") == "periodic",
            ),
        )
    return edges


def first_wetting_edge(boundary_conditions: dict[str, str] | None) -> Edge | None:
    """The wall contact angles are measured at: the first ``"wetting"`` edge, or ``None``.

    Wetting BCs are applied to every ``"wetting"`` edge, but contact angles are
    read only at this one.
    """
    edges = _resolve_wetting_edges(boundary_conditions) if boundary_conditions else []
    return edges[0][0] if edges else None
