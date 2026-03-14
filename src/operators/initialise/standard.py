"""Standard (uniform) initialisation — pure function.

Initialises a uniform density and velocity field, returning
population distributions at equilibrium.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from setup.lattice import Lattice
from operators.equilibrium.equilibrium import compute_equilibrium
from registry import initialise_operator


@initialise_operator(name="standard")
def init_standard(
    nx: int,
    ny: int,
    lattice: Lattice,
    *,
    density: float = 1.0,
    velocity: Tuple[float, float] = (0.0, 0.0),
    **kwargs,
) -> jnp.ndarray:
    """Initialise uniform density and velocity at equilibrium.

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        lattice: :class:`~setup.lattice.Lattice`.
        density: Uniform density value.
        velocity: Uniform velocity ``(ux, uy)``.

    Returns:
        Initial distribution ``f``, shape ``(nx, ny, q, 1)``.
    """
    rho = jnp.full((nx, ny, 1, 1), density)
    u = jnp.broadcast_to(jnp.array(velocity).reshape(1, 1, 1, 2), (nx, ny, 1, 2))
    return compute_equilibrium(rho, u, lattice)
