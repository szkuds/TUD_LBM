"""Shared stencil-padding utility for D2Q9 differential operators.

Includes pad-mode resolution from the boundary-condition registry.
Reads ``pad_edge_mode`` metadata from each registered boundary condition
and maps the per-edge BC configuration to the four ``jnp.pad`` mode
strings expected by :func:`compute_gradient` and :func:`compute_laplacian`.

The ordering convention is ``(y_end, y_start, x_end, x_start)`` — that is,
``(top, bottom, right, left)`` — which matches the order
:func:`determine_pad_modes` returns and the padding order in ``_gradient.py`` /
``_laplacian.py``.

**Both ghost layers of an axis are derived from that axis's original extent.**
The predecessor padded the two sides in sequence, so the start-side ``jnp.pad``
ran against an array already extended on the end side. On a fully periodic axis
the end pad had appended a copy of row 0, so ``mode="wrap"`` on the start side
copied *that* — the start ghost became row 0 instead of row ``n - 1``, and the
field the stencil saw was not periodic at all but carried a zero-gradient seam
at index 0.

That seam is pinned to the array origin, so it breaks translation invariance
along the axis: measured on a 201-cell periodic wall, a bubble held its centroid
at ``100.00000`` for 2000 steps while *its own 40-cell roll* drifted to
``139.147``, and ``gradient``/``laplacian`` failed to commute with ``jnp.roll``
by 8.0e-04 / 1.8e-03. In a periodic domain there is no restoring force for
translation, so a spurious seam force integrates without bound — the same
argument the D4 alignment invariant rests on.

Only a **fully periodic** axis was affected: ``bounce-back``, ``symmetry`` and
``wetting`` all resolve to ``"edge"``, and an edge pad after an edge pad happens
to be correct.
"""

from __future__ import annotations
from typing import Any
from typing import cast
import jax.numpy as jnp
from src.registry import get_operators


def _pad_axis(grid: jnp.ndarray, axis: int, mode_start: str, mode_end: str) -> jnp.ndarray:
    """Add one ghost layer to each side of *axis*, both read off *grid* itself.

    ``jnp.pad`` stays the authority on what a mode means, so any mode it
    supports keeps working: each side is padded from *grid* and only its own
    new layer is kept. Equal modes take the single-call fast path.
    """
    if mode_start == mode_end:
        width = [(0, 0)] * grid.ndim
        width[axis] = (1, 1)
        return jnp.pad(grid, tuple(width), mode=mode_start)

    start_width, end_width = [(0, 0)] * grid.ndim, [(0, 0)] * grid.ndim
    start_width[axis], end_width[axis] = (1, 0), (0, 1)
    start_slice, end_slice = [slice(None)] * grid.ndim, [slice(None)] * grid.ndim
    start_slice[axis], end_slice[axis] = slice(0, 1), slice(-1, None)

    start = jnp.pad(grid, tuple(start_width), mode=mode_start)[tuple(start_slice)]
    end = jnp.pad(grid, tuple(end_width), mode=mode_end)[tuple(end_slice)]
    return jnp.concatenate([start, grid, end], axis=axis)


def _apply_stencil_padding(
    grid_2d: jnp.ndarray,
    pad_mode: tuple[str, ...],
) -> jnp.ndarray:
    """Pad a 2-D field with one ghost cell per edge.

    Args:
        grid_2d: Shape ``(nx, ny)``.
        pad_mode: ``(y_end, y_start, x_end, x_start)``, i.e.
            ``(top, bottom, right, left)`` — the order
            :func:`determine_pad_modes` returns.

    Returns:
        Shape ``(nx + 2, ny + 2)``.
    """
    # y first, then x, so the corner cells are read off the y-padded field.
    padded = _pad_axis(grid_2d, 1, pad_mode[1], pad_mode[0])
    return _pad_axis(padded, 0, pad_mode[3], pad_mode[2])


def to_2d(grid: jnp.ndarray) -> jnp.ndarray:
    """Squeeze ``(nx, ny, nz, 1, 1)`` → ``(nx, ny)``; no-op if already 2-D."""
    _grid_ndim_5d = 5
    if grid.ndim != _grid_ndim_5d:
        msg = f"Expected 5-D grid, got shape {grid.shape}"
        raise ValueError(msg)
    if grid.shape[2] != 1:
        msg = f"Expected singleton nz dimension, got shape {grid.shape}"
        raise ValueError(msg)
    return grid[:, :, 0, 0, 0]


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
        Four padding-mode strings ``[top, bottom, right, left]``, the order
        :func:`_apply_stencil_padding` expects.
    """
    # The per-edge fallback is "edge", so an unpopulated registry would not raise
    # — it would silently pad a periodic run as if it were walled. Import the
    # package for its registration side effect first, as the eos/pressure kinds
    # do. src.simulation_io.analysis.wetting_overlay reaches here without it, and
    # would otherwise draw a band the solver never applied.
    import src.operators.boundary  # noqa: F401

    # Build lookup: bc_name -> pad_edge_mode from registry metadata
    bc_ops = get_operators("boundary_condition")
    pad_for: dict[str, str] = cast(
        "dict[str, str]",
        {
            name: entry.metadata.get("pad_edge_mode", "edge") if entry.metadata else "edge"
            for name, entry in bc_ops.items()
        },
    )

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
