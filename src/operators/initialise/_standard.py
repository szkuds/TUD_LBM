"""Standard (uniform) initialisation — pure function.

Initialises a uniform density and velocity field, returning
population distributions at equilibrium.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.equilibrium import build_equilibrium_fn
from src.registry import initialise_operator

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice


@initialise_operator(name="standard")
def init_standard(
    nx: int,
    ny: int,
    nz: int,
    lattice: Lattice,
    *,
    density: float = 1.0,
    velocity: tuple[float, ...] = (0.0, 0.0, 0.0),
    **_kwargs: object,
) -> jnp.ndarray:
    """Initialise uniform density and velocity at equilibrium.

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        nz: Grid size in z.
        lattice: :class:`~setup.lattice.Lattice`.
        density: Uniform density value.
        velocity: Uniform velocity ``(ux, uy, uz)``.
        **kwargs: Additional arguments (ignored).

    Returns:
        Initial distribution ``f``, shape ``(nx, ny, nz, q, 1)``.
    """
    equilibrium_fn = build_equilibrium_fn("wb")
    rho = jnp.full((nx, ny, nz, 1, 1), density)
    u = jnp.broadcast_to(jnp.array(velocity[: lattice.d]).reshape(1, 1, 1, 1, lattice.d), (nx, ny, nz, 1, lattice.d))
    return equilibrium_fn(rho, u, lattice)
