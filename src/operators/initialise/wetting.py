"""Wetting initialisation — pure function.

Places a sessile droplet (half-circle) at the bottom wall.
"""

from __future__ import annotations
import jax.numpy as jnp
from operators.equilibrium.equilibrium import compute_equilibrium
from registry import initialise_operator
from setup.lattice import Lattice


@initialise_operator(name="wetting")
def init_wetting(
    nx: int,
    ny: int,
    lattice: Lattice,
    *,
    rho_l: float = 1.0,
    rho_v: float = 0.33,
    interface_width: int = 4,
    **kwargs,
) -> jnp.ndarray:
    """Initialise a sessile droplet resting on the bottom wall.

    The droplet is a half-circle centred at ``(nx/2, 0)`` with
    radius ``ny / 3.33``.

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
    r = ny / 3.33
    x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    xc = nx // 2
    distance = jnp.sqrt((x - xc) ** 2 + y**2)

    rho_2d = (rho_l + rho_v) / 2.0 + (rho_l - rho_v) / 2.0 * jnp.tanh(
        2.0 * (r - distance) / interface_width,
    )
    rho = jnp.zeros((nx, ny, 1, 1)).at[:, :, 0, 0].set(rho_2d)
    u = jnp.zeros((nx, ny, 1, 2))
    return compute_equilibrium(rho, u, lattice)
