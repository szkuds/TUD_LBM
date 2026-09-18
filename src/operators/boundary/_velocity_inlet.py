"""Velocity-inlet boundary condition — pure function.

Resets populations at the inlet edge to the equilibrium distribution for
a prescribed parabolic velocity profile. Simplest correct option for a
worked channel-flow example — not a Zou-He pressure-coupled inlet.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.equilibrium._equilibrium_well_balanced import compute_equilibrium
from src.registry import boundary_condition

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice


@boundary_condition(name="velocity-inlet", pad_edge_mode="edge")
def apply_velocity_inlet(
    f_streamed: jnp.ndarray,
    _f_collision: jnp.ndarray,
    lattice: Lattice,
    edge: str,
    *,
    u0: float = 0.05,
) -> jnp.ndarray:
    """Apply an equilibrium-reset velocity inlet on the ``"left"`` edge.

    Sets populations at the inlet column to the equilibrium distribution
    for a parabolic inflow profile ``u_x(y) = u0 * 4*(y/(ny-1))*(1-y/(ny-1))``,
    ``u_y = 0``, at fixed density ``rho_in = 1.0``.

    Args:
        f_streamed: Post-streaming populations, shape ``(nx, ny, nz, q, 1)``.
        _f_collision: Post-collision populations (unused — inlet overwrites
            unconditionally rather than blending with streamed values).
        lattice: :class:`~src.lattice.lattice.Lattice`.
        edge: Edge name; only ``"left"`` is handled, other edges are no-ops.
        u0: Peak inflow velocity (lattice units).

    Returns:
        Populations with the inlet column reset to the prescribed equilibrium.
    """
    if edge != "left":
        return f_streamed

    ny = f_streamed.shape[1]
    y = jnp.arange(ny, dtype=f_streamed.dtype)
    denom = ny - 1 if ny > 1 else 1
    u_x = u0 * 4.0 * (y / denom) * (1.0 - y / denom)

    u_in = jnp.zeros((1, ny, 1, 1, lattice.d), dtype=f_streamed.dtype)
    u_in = u_in.at[0, :, 0, 0, 0].set(u_x)
    rho_in = jnp.ones((1, ny, 1, 1, 1), dtype=f_streamed.dtype)

    feq_inlet = compute_equilibrium(rho_in, u_in, lattice)
    return f_streamed.at[0, :, 0, :, 0].set(feq_inlet[0, :, 0, :, 0])
