"""Boundary condition operators — composite builder.

Public API: build_bc(), BCMasks, build_bc_masks

Implementation modules (_bounce_back.py, _periodic.py, _symmetry.py) are internal.

Example:
    from operators.boundary import build_bc, build_bc_masks

    bc_fn = build_bc({"top": "symmetry", "bottom": "bounce-back"}, lattice)
    masks = build_bc_masks((64, 64))
    f = bc_fn(f_stream, f_col, masks)
"""

from __future__ import annotations
from collections.abc import Callable
from typing import Any
from typing import NamedTuple
import jax.numpy as jnp
from operators.boundary import _bounce_back as _bb  # noqa: F401
from operators.boundary import _periodic as _per  # noqa: F401
from operators.boundary import _symmetry as _sym  # noqa: F401
from operators.protocols import BoundaryOperator
from registry import get_operators
from setup.lattice import Lattice


class BCMasks(NamedTuple):
    """Pre-computed boundary-condition masks — valid JAX pytree.

    Each mask is a boolean ``jax.Array`` of shape ``(nx, ny, 1, 1)``
    that is ``True`` on the corresponding edge row/column.

    Attributes:
        top: Mask for the top boundary (y = ny-1).
        bottom: Mask for the bottom boundary (y = 0).
        left: Mask for the left boundary (x = 0).
        right: Mask for the right boundary (x = nx-1).
    """

    top: jnp.ndarray
    bottom: jnp.ndarray
    left: jnp.ndarray
    right: jnp.ndarray


def build_bc_masks(
    grid_shape: tuple[int, ...],
) -> BCMasks:
    """Construct pre-computed boundary-condition masks.

    Each mask is a boolean array of shape ``(nx, ny, 1, 1)`` that is
    ``True`` on the corresponding edge row/column.

    Args:
        grid_shape: Spatial dimensions ``(nx, ny, ...)``.
        bc_config: Boundary-condition mapping (unused for now; reserved
            for future edge-type encoding).

    Returns:
        A :class:`BCMasks` NamedTuple.
    """
    nx, ny = grid_shape[:2]
    top = jnp.zeros((nx, ny, 1, 1), dtype=bool).at[:, -1].set(True)
    bottom = jnp.zeros((nx, ny, 1, 1), dtype=bool).at[:, 0].set(True)
    left = jnp.zeros((nx, ny, 1, 1), dtype=bool).at[0, :].set(True)
    right = jnp.zeros((nx, ny, 1, 1), dtype=bool).at[-1, :].set(True)
    return BCMasks(top=top, bottom=bottom, left=left, right=right)


def _get_bc_dispatch() -> dict[str, Callable]:
    """Build the BC dispatch dict from the global registry."""
    bc_ops = get_operators("boundary_condition")
    return {name: entry.target for name, entry in bc_ops.items()}


def build_bc(
    bc_config: dict[str, Any] | None,
    lattice: Lattice,
) -> BoundaryOperator:
    """Build a composite BC closure from a bc_config dict.

    The returned function applies boundary conditions in order:
    bottom → top → left → right.  Edges not present in *bc_config*
    or mapped to ``"periodic"`` are no-ops.

    Args:
        bc_config: Mapping ``{_edge: bc_type, ...}``, e.g.
            ``{"top": "symmetry", "bottom": "bounce-back", "left": "periodic", "right": "periodic"}``.
            ``None`` means all-periodic.
        lattice: :class:`~setup.lattice.Lattice`.

    Returns:
        A callable satisfying the BoundaryOperator protocol,
        ``bc_fn(f_stream, f_col, bc_masks) → f``.
    """
    if bc_config is None:
        bc_config = {}

    bc_dispatch = _get_bc_dispatch()

    # Pre-compute the list of (_edge, bc_fn) pairs at build time.
    # Only include edges that need an operation (skip periodic / unknown).
    _edge_order = ("bottom", "top", "left", "right")
    ops = []
    for edge in _edge_order:
        bc_type = bc_config.get(edge, "periodic")
        fn = bc_dispatch.get(bc_type)
        if fn is not None and bc_type != "periodic":
            ops.append((edge, fn))

    def bc_fn(
        f_streamed: jnp.ndarray,
        f_collision: jnp.ndarray,
        bc_masks: Any,
    ) -> jnp.ndarray:
        """Apply all boundary conditions in sequence.

        Args:
            f_streamed: Post-streaming populations.
            f_collision: Post-collision populations.
            bc_masks: :class:`~setup.simulation_setup.BCMasks` (currently
                unused — the per-_edge functions slice by index; masks are
                reserved for a future vectorised implementation).

        Returns:
            Populations with boundary conditions applied.
        """
        f = f_streamed
        for _edge, _fn in ops:
            f = _fn(f, f_collision, lattice, _edge)
        return f

    return bc_fn


__all__ = ["BCMasks", "build_bc", "build_bc_masks"]
