"""Build-time edge configuration for wetting boundaries.

Determines which edges need wetting and their perpendicular BC
relationships.
"""

from __future__ import annotations

# Maps each wetting edge to its two perpendicular edges (start, end along wall).
_PERPENDICULAR = {
    "bottom": ("left", "right"),
    "top": ("left", "right"),
    "left": ("bottom", "top"),
    "right": ("bottom", "top"),
}


def _resolve_wetting_edges(
    boundary_conditions: dict[str, str],
) -> list[tuple[str, bool, bool]]:
    """Determine wetting edges and whether their perpendicular BCs are periodic.

    Returns:
        List of ``(edge, perp_start_periodic, perp_end_periodic)``.
    """
    edges = []
    for edge in ("bottom", "top", "left", "right"):
        if boundary_conditions.get(edge) != "wetting":
            continue
        perp_start, perp_end = _PERPENDICULAR[edge]
        edges.append(
            (
                edge,
                boundary_conditions.get(perp_start, "periodic") == "periodic",
                boundary_conditions.get(perp_end, "periodic") == "periodic",
            ),
        )
    return edges


def first_wetting_edge(boundary_conditions: dict[str, str] | None) -> str | None:
    """The wall contact angles are measured at: the first ``"wetting"`` edge, or ``None``.

    Wetting BCs are applied to every ``"wetting"`` edge, but contact angles are
    read only at this one.
    """
    edges = _resolve_wetting_edges(boundary_conditions) if boundary_conditions else []
    return edges[0][0] if edges else None
