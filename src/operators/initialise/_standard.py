"""Standard (uniform) initialisation — pure function.

Initialises a uniform density and velocity field, returning
population distributions at equilibrium.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from operators.equilibrium import build_equilibrium_fn
from registry import initialise_operator

if TYPE_CHECKING:
    from setup.lattice import Lattice


@initialise_operator(name="standard")
def init_standard(
    nx: int,
    ny: int,
    lattice: Lattice,
    *,
    density: float = 1.0,
    velocity: tuple[float, float] = (0.0, 0.0),
    **_kwargs: object,
) -> jnp.ndarray:
    """Initialise uniform density and velocity at equilibrium.

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        lattice: :class:`~setup.lattice.Lattice`.
        density: Uniform density value.
        velocity: Uniform velocity ``(ux, uy)``.
        **kwargs: Additional arguments (ignored).

    Returns:
        Initial distribution ``f``, shape ``(nx, ny, q, 1)``.
    """
    equilibrium_fn = build_equilibrium_fn("wb")
    rho = jnp.full((nx, ny, 1, 1), density)
    u = jnp.broadcast_to(jnp.array(velocity).reshape(1, 1, 1, 2), (nx, ny, 1, 2))
    return equilibrium_fn(rho, u, lattice)
