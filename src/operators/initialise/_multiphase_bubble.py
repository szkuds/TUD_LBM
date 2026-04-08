"""Multiphase bubble initialisation — pure function.

Places a vapour bubble (low-density centre) in the middle of the domain
using a tanh density profile.
"""

from __future__ import annotations
import jax.numpy as jnp
from operators.equilibrium._equilibrium import compute_equilibrium
from registry import initialise_operator
from setup.lattice import Lattice


@initialise_operator(name="multiphase_bubble")
def init_multiphase_bubble(
    nx: int,
    ny: int,
    lattice: Lattice,
    *,
    rho_l: float = 1.0,
    rho_v: float = 0.33,
    interface_width: int = 4,
    **kwargs,
) -> jnp.ndarray:
    """Initialise a centred vapour bubble in a liquid background.

    The density profile uses a hyperbolic tangent transition:

    .. math::

        \\rho = \\frac{\\rho_l + \\rho_v}{2}
              + \\frac{\\rho_l - \\rho_v}{2}
                \\tanh\\!\\left(\\frac{r - R}{W}\\right)

    where *r* is the distance from the domain centre and *R* is the
    bubble radius (``min(nx, ny) / 4``).

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        lattice: :class:`~setup.lattice.Lattice`.
        rho_l: Liquid density.
        rho_v: Vapour density.
        interface_width: Diffuse-interface thickness *W*.

    Returns:
        Initial distribution ``f``, shape ``(nx, ny, q, 1)``.
    """
    x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    cx, cy = nx // 2, ny // 2
    radius = min(nx, ny) // 4
    distance = jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    rho_2d = (rho_l + rho_v) / 2.0 + (rho_l - rho_v) / 2.0 * jnp.tanh(
        (distance - radius) / interface_width,
    )
    rho = rho_2d.reshape(nx, ny, 1, 1)
    u = jnp.zeros((nx, ny, 1, 2))
    return compute_equilibrium(rho, u, lattice)
