"""Multiphase bubble-at-bottom initialisation — pure function.

Places a vapour bubble near the bottom of the domain.
"""

from __future__ import annotations

import jax.numpy as jnp

from setup.lattice import Lattice
from operators.equilibrium.equilibrium import compute_equilibrium
from registry import initialise_operator


@initialise_operator(name="multiphase_bubble_bot")
def init_multiphase_bubble_bot(
    nx: int,
    ny: int,
    lattice: Lattice,
    *,
    rho_l: float = 1.0,
    rho_v: float = 0.33,
    interface_width: int = 4,
    **kwargs,
) -> jnp.ndarray:
    """Initialise a vapour bubble centred near the bottom of the domain.

    Centre: ``(nx/2, ny/6)``, radius: ``min(nx, ny) / 4``.

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        lattice: :class:`~setup.lattice.Lattice`.
        rho_l: Liquid density.
        rho_v: Vapour density.
        interface_width: Diffuse-interface thickness.

    Returns:
        Initial distribution ``f``, shape ``(nx, ny, q, 1)``.
    """
    x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    cx, cy = nx // 2, ny // 6
    radius = min(nx, ny) // 4
    distance = jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    rho_2d = (rho_l + rho_v) / 2.0 + (rho_l - rho_v) / 2.0 * jnp.tanh(
        (distance - radius) / interface_width
    )
    rho = rho_2d.reshape(nx, ny, 1, 1)
    u = jnp.zeros((nx, ny, 1, 2))
    return compute_equilibrium(rho, u, lattice)
