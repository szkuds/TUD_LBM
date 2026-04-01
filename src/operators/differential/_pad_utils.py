"""Shared stencil-padding utility for D2Q9 differential operators."""

from __future__ import annotations
import jax.numpy as jnp


def apply_stencil_padding(
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
