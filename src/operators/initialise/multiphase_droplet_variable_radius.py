"""Multiphase droplet with variable radius — pure function.

Places a liquid droplet at the domain centre with a user-supplied radius.
"""

from __future__ import annotations
import jax.numpy as jnp
from operators.equilibrium.equilibrium import compute_equilibrium
from registry import initialise_operator
from setup.lattice import Lattice


@initialise_operator(name="multiphase_droplet_variable_radius")
def init_multiphase_droplet_variable_radius(
    nx: int,
    ny: int,
    lattice: Lattice,
    *,
    rho_l: float = 1.0,
    rho_v: float = 0.33,
    interface_width: int = 4,
    radius: float | None = None,
    **kwargs,
) -> jnp.ndarray:
    """Initialise a centred liquid droplet with a user-specified radius.

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        lattice: :class:`~setup.lattice.Lattice`.
        rho_l: Liquid density.
        rho_v: Vapour density.
        interface_width: Diffuse-interface thickness.
        radius: Droplet radius.  Defaults to ``min(nx, ny) / 8``.

    Returns:
        Initial distribution ``f``, shape ``(nx, ny, q, 1)``.
    """
    if radius is None:
        radius = min(nx, ny) / 8.0

    x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    cx, cy = nx / 2.0, ny / 2.0
    distance = jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    rho_2d = (rho_l + rho_v) / 2.0 - (rho_l - rho_v) / 2.0 * jnp.tanh(
        (distance - radius) / interface_width,
    )
    rho = rho_2d.reshape(nx, ny, 1, 1)
    u = jnp.zeros((nx, ny, 1, 2))
    return compute_equilibrium(rho, u, lattice)
