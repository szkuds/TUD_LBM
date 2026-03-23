"""Composite boundary-condition builder — chains per-edge BC functions.

Extracted from
:class:`simulation_operators.boundary_condition.BoundaryCondition`.

``build_composite_bc`` inspects the ``bc_config`` dict at *setup time*
(outside JIT) and returns a single closure
``bc_fn(f_stream, f_col, bc_masks) → f`` that chains the appropriate
per-edge pure functions.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import Any
import jax.numpy as jnp

# Ensure BC modules are imported so decorators fire
from operators.boundary import bounce_back as _bb  # noqa: F401
from operators.boundary import periodic as _per  # noqa: F401
from operators.boundary import symmetry as _sym  # noqa: F401
from registry import get_operators
from setup.lattice import Lattice


def _get_bc_dispatch() -> dict[str, Callable]:
    """Build the BC dispatch dict from the global registry."""
    bc_ops = get_operators("boundary_condition")
    return {name: entry.target for name, entry in bc_ops.items()}


def build_composite_bc(
    bc_config: dict[str, Any] | None,
    lattice: Lattice,
) -> Callable:
    """Build a composite BC closure from a bc_config dict.

    The returned function applies boundary conditions in order:
    bottom → top → left → right.  Edges not present in *bc_config*
    or mapped to ``"periodic"`` are no-ops.

    Args:
        bc_config: Mapping ``{edge: bc_type, ...}``, e.g.
            ``{"top": "symmetry", "bottom": "bounce-back",
              "left": "periodic", "right": "periodic"}``.
            ``None`` means all-periodic.
        lattice: :class:`~setup.lattice.Lattice`.

    Returns:
        ``bc_fn(f_stream, f_col, bc_masks) → f``.
    """
    if bc_config is None:
        bc_config = {}

    bc_dispatch = _get_bc_dispatch()

    # Pre-compute the list of (edge, bc_fn) pairs at build time.
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
                unused — the per-edge functions slice by index; masks are
                reserved for a future vectorised implementation).

        Returns:
            Populations with boundary conditions applied.
        """
        f = f_streamed
        for edge, fn in ops:
            f = fn(f, f_collision, lattice, edge)
        return f

    return bc_fn
