"""Multiphase double-bubble initialisation — pure function.

Places two vapour bubbles side by side in the domain.
"""

from __future__ import annotations
import jax.numpy as jnp
from operators.equilibrium._equilibrium import compute_equilibrium
from registry import initialise_operator
from setup.lattice import Lattice


@initialise_operator(name="multiphase_bubble_bubble")
def init_multiphase_bubble_bubble(
    nx: int,
    ny: int,
    lattice: Lattice,
    *,
    rho_l: float = 1.0,
    rho_v: float = 0.33,
    interface_width: int = 4,
    **kwargs,
) -> jnp.ndarray:
    """Initialise two vapour bubbles side by side.

    Centres at ``(nx/4, ny/2)`` and ``(nx*2.4/4, ny/2)``,
    radius ``min(nx, ny) / 5``.  The minimum of the two tanh
    profiles is used to produce two distinct bubbles.

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
    radius = min(nx, ny) / 5.0

    # Bubble 1
    cx1, cy1 = nx / 4.0, ny / 2.0
    d1 = jnp.sqrt((x - cx1) ** 2 + (y - cy1) ** 2)
    rho1 = (rho_l + rho_v) / 2.0 + (rho_l - rho_v) / 2.0 * jnp.tanh(
        (d1 - radius) / interface_width,
    )

    # Bubble 2
    cx2, cy2 = nx * 2.4 / 4.0, ny / 2.0
    d2 = jnp.sqrt((x - cx2) ** 2 + (y - cy2) ** 2)
    rho2 = (rho_l + rho_v) / 2.0 + (rho_l - rho_v) / 2.0 * jnp.tanh(
        (d2 - radius) / interface_width,
    )

    # Take the minimum to produce two separate low-density regions
    rho_2d = jnp.minimum(rho1, rho2)

    rho = rho_2d.reshape(nx, ny, 1, 1)
    u = jnp.zeros((nx, ny, 1, 2))
    return compute_equilibrium(rho, u, lattice)
