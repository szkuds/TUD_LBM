"""Wetting with chemical step initialisation — pure function.

Identical to the standard wetting initialisation but uses a slightly
different radius ratio (``ny / 3.3``) to model a chemical heterogeneity
step on the substrate.
"""

from __future__ import annotations
import jax.numpy as jnp
from operators.equilibrium.equilibrium import compute_equilibrium
from registry import initialise_operator
from setup.lattice import Lattice


@initialise_operator(name="wetting_chem_step")
def init_wetting_chemical_step(
    nx: int,
    ny: int,
    lattice: Lattice,
    *,
    rho_l: float = 1.0,
    rho_v: float = 0.33,
    interface_width: int = 4,
    **kwargs,
) -> jnp.ndarray:
    """Initialise a sessile droplet for a chemical-step wetting study.

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
    r = ny / 3.3
    x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    xc = nx // 2
    distance = jnp.sqrt((x - xc) ** 2 + y**2)

    rho_2d = (rho_l + rho_v) / 2.0 + (rho_l - rho_v) / 2.0 * jnp.tanh(
        2.0 * (r - distance) / interface_width,
    )
    rho = jnp.zeros((nx, ny, 1, 1)).at[:, :, 0, 0].set(rho_2d)
    u = jnp.zeros((nx, ny, 1, 2))
    return compute_equilibrium(rho, u, lattice)
