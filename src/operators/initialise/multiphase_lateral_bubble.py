"""Multiphase lateral (vertically stacked) double-bubble initialisation.
Places two vapour bubbles stacked vertically in the domain.
"""

from __future__ import annotations
import jax.numpy as jnp
from operators.equilibrium.equilibrium import compute_equilibrium
from registry import initialise_operator
from setup.lattice import Lattice


@initialise_operator(name="multiphase_lateral_bubble")
def init_multiphase_lateral_bubble(
    nx: int,
    ny: int,
    lattice: Lattice,
    *,
    rho_l: float = 1.0,
    rho_v: float = 0.33,
    interface_width: int = 4,
    **kwargs,
) -> jnp.ndarray:
    """Initialise two vapour bubbles stacked vertically.
    Centres at ``(nx/2, ny/3)`` and ``(nx/2, 2*ny/3)``,
    radius ``min(nx, ny) / 6.5``.  The minimum of two tanh
    profiles yields two distinct bubbles.

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
    radius = min(nx, ny) / 6.5
    # Bubble 1 (lower)
    cx1, cy1 = nx / 2.0, ny / 3.0
    d1 = jnp.sqrt((x - cx1) ** 2 + (y - cy1) ** 2)
    rho1 = (rho_l + rho_v) / 2.0 + (rho_l - rho_v) / 2.0 * jnp.tanh(
        (d1 - radius) / interface_width,
    )
    # Bubble 2 (upper)
    cx2, cy2 = nx / 2.0, 2.0 * ny / 3.0
    d2 = jnp.sqrt((x - cx2) ** 2 + (y - cy2) ** 2)
    rho2 = (rho_l + rho_v) / 2.0 + (rho_l - rho_v) / 2.0 * jnp.tanh(
        (d2 - radius) / interface_width,
    )
    rho_2d = jnp.minimum(rho1, rho2)
    rho = rho_2d.reshape(nx, ny, 1, 1)
    u = jnp.zeros((nx, ny, 1, 2))
    return compute_equilibrium(rho, u, lattice)
