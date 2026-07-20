"""Multiphase bubble-at-bottom initialisation — pure function.

Places a vapour bubble near the bottom of the domain.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.equilibrium._equilibrium import compute_equilibrium
from src.registry import initialise_operator

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice


@initialise_operator(name="multiphase_bubble_top")
def init_multiphase_bubble_top(
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
    """Initialise a vapour bubble centred near the bottom of the domain.

    Centre: ``(nx/2, ny/6)``, radius: ``min(nx, ny) / 4``.

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        nz: Grid size in z (supports nz=1 for pseudo-3D).
        lattice: :class:`~setup.lattice.Lattice`.
        rho_l: Liquid density.
        rho_v: Vapour density.
        interface_width: Diffuse-interface thickness.
        **kwargs: Additional arguments (ignored).

    Returns:
        Initial distribution ``f``, shape ``(nx, ny, nz, q, 1)``.
    """
    # Support pseudo-3D (nz=1) — bubble logic remains 2D
    if nz != 1:
        msg = "Multiphase bubble initialisation supports nz=1 (pseudo-3D) for now."
        raise ValueError(msg)

    x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    cx, _ = nx // 2, ny // 6
    radius = min(nx, ny) // 4
    distance = jnp.sqrt((x - cx) ** 2 + (ny - 1 - y) ** 2)

    rho_2d = (rho_l + rho_v) / 2.0 + (rho_l - rho_v) / 2.0 * jnp.tanh(
        (distance - radius) / interface_width,
    )
    rho = rho_2d.reshape(nx, ny, nz, 1, 1)
    u = jnp.zeros((nx, ny, nz, 1, lattice.d))
    return compute_equilibrium(rho, u, lattice)
