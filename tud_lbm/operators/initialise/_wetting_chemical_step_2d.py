"""Wetting with chemical step initialisation — pure function.

Identical to the standard wetting initialisation but uses a slightly
different radius ratio (``ny / 3.3``) to model a chemical heterogeneity
step on the substrate.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from tud_lbm.operators.equilibrium._equilibrium import compute_equilibrium
from tud_lbm.registry import initialise_operator

if TYPE_CHECKING:
    from tud_lbm.lattice.lattice import Lattice


@initialise_operator(name="wetting_chem_step")
def init_wetting_chemical_step(
    nx: int,
    ny: int,
    nz: int,
    lattice: Lattice,
    *,
    rho_l: float = 1.0,
    rho_v: float = 0.33,
    interface_width: int = 4,
    **_kwargs: object,
) -> jnp.ndarray:
    """Initialise a sessile droplet for a chemical-step wetting study.

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        nz: Grid size in z (must be 1).
        lattice: :class:`~setup.lattice.Lattice`.
        rho_l: Liquid density.
        rho_v: Vapour density.
        interface_width: Diffuse-interface thickness.
        **kwargs: Additional arguments (ignored).
        rho_v: Vapour density.
        interface_width: Diffuse-interface thickness.

    Returns:
        Initial distribution ``f``, shape ``(nx, ny, nz, q, 1)``.
    """
    if nz != 1:
        msg = "Chemical-step wetting initialisation only supports 2D (nz=1)."
        raise ValueError(msg)

    r = ny / 3.3
    x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    xc = nx // 2
    distance = jnp.sqrt((x - xc) ** 2 + y**2)

    rho_2d = (rho_l + rho_v) / 2.0 + (rho_l - rho_v) / 2.0 * jnp.tanh(
        2.0 * (r - distance) / interface_width,
    )
    rho = jnp.zeros((nx, ny, nz, 1, 1)).at[:, :, 0, 0, 0].set(rho_2d)
    u = jnp.zeros((nx, ny, nz, 1, lattice.d))
    return compute_equilibrium(rho, u, lattice)
