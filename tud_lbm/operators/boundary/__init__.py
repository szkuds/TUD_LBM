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
import functools
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import cast
import jax.numpy as jnp
from tud_lbm.operators.boundary import _bounce_back as _bb  # noqa: F401
from tud_lbm.operators.boundary import _outlet as _out  # noqa: F401
from tud_lbm.operators.boundary import _periodic as _per  # noqa: F401
from tud_lbm.operators.boundary import _symmetry as _sym  # noqa: F401
from tud_lbm.operators.boundary import _velocity_inlet as _vin  # noqa: F401
from tud_lbm.registry import get_operators

if TYPE_CHECKING:
    from collections.abc import Callable
    from tud_lbm.lattice.lattice import Lattice
    from tud_lbm.operators.protocols import BoundaryOperator


class BCMasks(NamedTuple):
    """Pre-computed boundary-condition masks — valid JAX pytree.

    Each mask is a boolean ``jax.Array`` of shape ``(nx, ny, nz, 1, 1)``
    that is ``True`` on the corresponding edge row/column/plane.

    Attributes:
        top: Mask for the top boundary (y = ny-1).
        bottom: Mask for the bottom boundary (y = 0).
        left: Mask for the left boundary (x = 0).
        right: Mask for the right boundary (x = nx-1).
        front: Mask for the front boundary (z = nz-1).
        back: Mask for the back boundary (z = 0).
    """

    top: jnp.ndarray
    bottom: jnp.ndarray
    left: jnp.ndarray
    right: jnp.ndarray
    front: jnp.ndarray | None = None
    back: jnp.ndarray | None = None


def build_bc_masks(
    grid_shape: tuple[int, ...],
) -> BCMasks:
    """Construct pre-computed boundary-condition masks.

    Each mask is a boolean array of shape ``(nx, ny, nz, 1, 1)`` that is
    ``True`` on the corresponding edge row/column/plane.

    Args:
        grid_shape: Spatial dimensions ``(nx, ny, nz)``.

    Returns:
        A :class:`BCMasks` NamedTuple with 3D-shaped masks for all 6 faces.
    """
    nx, ny, nz = grid_shape[:3] if len(grid_shape) >= 3 else (*grid_shape[:2], 1)  # noqa: PLR2004

    # Create 3D masks for x-faces (left/right)
    top = jnp.zeros((nx, ny, nz, 1, 1), dtype=bool).at[:, -1, :].set(True)
    bottom = jnp.zeros((nx, ny, nz, 1, 1), dtype=bool).at[:, 0, :].set(True)
    left = jnp.zeros((nx, ny, nz, 1, 1), dtype=bool).at[0, :, :].set(True)
    right = jnp.zeros((nx, ny, nz, 1, 1), dtype=bool).at[-1, :, :].set(True)

    # Create 3D masks for z-faces (front/back) — optional for 3D
    front = jnp.zeros((nx, ny, nz, 1, 1), dtype=bool).at[:, :, -1].set(True) if nz > 1 else None
    back = jnp.zeros((nx, ny, nz, 1, 1), dtype=bool).at[:, :, 0].set(True) if nz > 1 else None

    return BCMasks(top=top, bottom=bottom, left=left, right=right, front=front, back=back)


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
        fn = bc_dispatch.get("bounce-back") if bc_type == "wetting" else bc_dispatch.get(bc_type)
        if fn is not None and bc_type == "velocity-inlet":
            inlet_params = bc_config.get(f"{edge}_velocity_inlet", {})
            fn = functools.partial(fn, u0=inlet_params.get("u0", 0.05))
        if fn is not None and bc_type != "periodic":
            ops.append((edge, fn))

    def bc_fn(
        f_streamed: jnp.ndarray,
        f_collision: jnp.ndarray,
        bc_masks: Any,  # noqa: ANN401, ARG001
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

    return cast("BoundaryOperator", bc_fn)


__all__ = ["BCMasks", "build_bc", "build_bc_masks"]
