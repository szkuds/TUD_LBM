"""Shared stencil-padding utility for D2Q9 differential operators.

Includes pad-mode resolution from the boundary-condition registry.
Reads ``pad_edge_mode`` metadata from each registered boundary condition
and maps the per-edge BC configuration to the four ``jnp.pad`` mode
strings expected by :func:`compute_gradient` and :func:`compute_laplacian`.

The ordering convention is:
``[right_y, left_y, bottom_x, top_x]``
which matches the padding order in ``gradient.py`` / ``laplacian.py``.
"""

from __future__ import annotations
from typing import Any
import jax.numpy as jnp
from tud_lbm.registry import get_operators


def _apply_stencil_padding(
    grid_2d: jnp.ndarray,
    pad_mode: tuple[str, ...],
) -> jnp.ndarray:
    """Pad a 2-D field with one ghost cell per edge.

    Args:
        grid_2d: Shape ``(nx, ny)``.
        pad_mode: ``(right_y, left_y, bottom_x, top_x)``.

    Returns:
        Shape ``(nx + 2, ny + 2)``.
    """
    gp = jnp.pad(grid_2d, ((0, 0), (0, 1)), mode=pad_mode[0])
    gp = jnp.pad(gp, ((0, 0), (1, 0)), mode=pad_mode[1])
    gp = jnp.pad(gp, ((0, 1), (0, 0)), mode=pad_mode[2])
    return jnp.pad(gp, ((1, 0), (0, 0)), mode=pad_mode[3])


def to_2d(grid: jnp.ndarray) -> jnp.ndarray:
    """Squeeze ``(nx, ny, 1, 1)`` → ``(nx, ny)``; no-op if already 2-D."""
    return grid[:, :, 0, 0] if grid.ndim == 4 else grid


def determine_pad_modes(bc_config: dict[str, Any] | None) -> list[str]:
    """Derive the four pad-mode strings from a *bc_config* dict.

    Each edge's BC name is looked up in the global ``"boundary_condition"``
    registry, and its ``pad_edge_mode`` metadata value is used.  If the
    metadata is missing, ``"edge"`` is used as a safe fallback.

    Args:
        bc_config: Mapping of edge names to BC types, e.g.
            ``{"top": "symmetry", "bottom": "bounce-back", "left": "periodic", "right": "periodic"}``.
            ``None`` means all-periodic (→ all ``"wrap"``).

    Returns:
        Four padding-mode strings ``[top, bottom, left, right]``.
    """
    # Build lookup: bc_name -> pad_edge_mode from registry metadata
    bc_ops = get_operators("boundary_condition")
    pad_for: dict[str, str] = {
        name: entry.metadata.get("pad_edge_mode", "edge") if entry.metadata else "edge"
        for name, entry in bc_ops.items()
    }

    if bc_config is None:
        return ["wrap", "wrap", "wrap", "wrap"]  # all-periodic default

    def _mode(edge: str) -> str:
        bc_type = bc_config.get(edge, "periodic")
        return pad_for.get(bc_type, "edge")

    return [
        _mode("top"),
        _mode("bottom"),
        _mode("right"),
        _mode("left"),
    ]
