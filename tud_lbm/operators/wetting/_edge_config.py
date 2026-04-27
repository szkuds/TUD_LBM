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
    bc_config: dict[str, str],
) -> list[tuple[str, bool, bool]]:
    """Determine wetting edges and whether their perpendicular BCs are periodic.

    Returns:
        List of ``(edge, perp_start_periodic, perp_end_periodic)``.
    """
    edges = []
    for edge in ("bottom", "top", "left", "right"):
        if bc_config.get(edge) != "wetting":
            continue
        perp_start, perp_end = _PERPENDICULAR[edge]
        edges.append(
            (
                edge,
                bc_config.get(perp_start, "periodic") == "periodic",
                bc_config.get(perp_end, "periodic") == "periodic",
            )
        )
    return edges
