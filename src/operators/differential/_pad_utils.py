"""Shared stencil-padding utility for D2Q9 differential operators.

The four ``jnp.pad`` mode strings come from the configuration,
:attr:`SimulationConfig.pad_modes
<src.config.simulation_config.SimulationConfig.pad_modes>`, which reads each
boundary condition's ``pad_edge_mode`` registration.

The ordering convention is ``(y_end, y_start, x_end, x_start)`` — that is,
``(top, bottom, right, left)`` — which matches the order
``SimulationConfig.pad_modes`` returns and the padding order in ``_gradient.py`` /
``_laplacian.py``.

**Both ghost layers of an axis are derived from that axis's original extent.**
Padding the sides in sequence would let ``mode="wrap"`` on the start side copy
the end side's freshly appended row 0 instead of row ``n - 1``: a zero-gradient
seam pinned to the array origin, which breaks translation invariance on a fully
periodic axis (``tests/pipeline/test_translation_invariance.py``).
"""

from __future__ import annotations
import jax.numpy as jnp


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
            ``SimulationConfig.pad_modes`` returns.

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
