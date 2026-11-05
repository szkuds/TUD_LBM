"""Build-time edge configuration for wetting boundaries.

Determines which edges need wetting and whether the edges bounding them are periodic.
"""

from __future__ import annotations
from src.config.boundary import Edge

#: Order in which ``[boundary_conditions]`` is scanned for wetting walls; the first is measured.
_WETTING_SCAN_ORDER = (Edge.BOTTOM, Edge.TOP, Edge.LEFT, Edge.RIGHT)


def _resolve_wetting_edges(
    boundary_conditions: dict[str, str],
) -> list[tuple[Edge, bool, bool]]:
    """Determine wetting edges and whether their perpendicular BCs are periodic.

    Returns:
        List of ``(edge, perp_start_periodic, perp_end_periodic)``.
    """
    edges = []
    for edge in _WETTING_SCAN_ORDER:
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
