"""Canonical wall-aligned view of a 2D density field.

The contact-angle and contact-line formulas were originally written for a
droplet sitting on the **bottom** wall: they sample fixed rows counting up
from ``y=0`` and use a ``π/2 + arctan(Δx)`` sign convention that only holds
when the outward wall normal points in ``−y``.

Rather than re-parameterise axes, offsets, and signs at every measurement
site, :func:`to_canonical` maps the field into a single **canonical frame**
where the wetting wall is always at column ``0`` with liquid at increasing
column index — geometrically identical to a sessile drop on the bottom. Every
downstream formula then applies unchanged for all four edges.

The transform is exactly the transpose-plus-flip that the wetting *applicator*
already uses in :mod:`src.operators.wetting._apply_edge` (``arr = gp.T`` for
left/right; ghost index ``0`` vs ``-1``). Keeping the two in lock-step is what
guarantees the optimiser's ``phi_left`` / ``d_rho_left`` parameters address the
same physical contact line the measurement reports.

Both ``.T`` and ``[:, ::-1]`` lower to ``lax.transpose`` / ``lax.rev``:
static-shape, jit-traceable, and differentiable (the reversal only permutes
the wall-normal axis, so gradients route back to the correct original rows).
``edge`` is a plain Python string resolved at trace time, so no
``static_argnums`` is needed.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax.numpy as jnp

#: Physical axis that the canonical *tangential* coordinate (axis 0 of the
#: returned view) corresponds to. Bottom/top scan along x; left/right along y.
#: This is what ``cll_left`` / ``cll_right`` are measured in.
EDGE_TANGENTIAL_AXIS = {
    "bottom": "x",
    "top": "x",
    "left": "y",
    "right": "y",
}


def to_canonical(rho_2d: jnp.ndarray, edge: str) -> jnp.ndarray:
    """Map ``rho_2d`` into the wall-aligned canonical frame for *edge*.

    Args:
        rho_2d: Density slice, shape ``(nx, ny)``.
        edge: One of ``"bottom"``, ``"top"``, ``"left"``, ``"right"``.

    Returns:
        A view of shape ``(tangential, normal)`` where index 0 of the normal
        axis is the wall row and index increases into the fluid. For
        ``"bottom"`` this is the identity.
    """
    arr = rho_2d.T if edge in ("left", "right") else rho_2d
    return arr[:, ::-1] if edge in ("top", "right") else arr
