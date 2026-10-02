r"""Two equal droplets on a head-on or off-centre collision course.

Zhang, Guo & Wang (Phys. Fluids 34, 012110, 2022), Eqs. 36-37:

.. math::

    \rho = \frac{\rho_g + \rho_l}{2} + \frac{\rho_l - \rho_g}{2}\,\Phi, \qquad
    \Phi = 1 + \sum_{k=1,2} \tanh\frac{2(r - |x - x_k|)}{W},

and ``u_x = ±U_0 (1 + Phi)/2`` (``+`` left of the domain centre, ``-`` right),
``u_y = 0``. The droplets sit on the horizontal centre line, their surfaces
``gap`` cells apart, offset vertically by ``chi = 2 r B`` so that ``B = chi/(2r)``.
With this ``U_0`` (each droplet's speed), ``We = 8 r rho_l U_0^2 / sigma`` and
``Re = 4 r U_0 / nu_l`` (Eq. 35).
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.equilibrium._equilibrium_well_balanced import compute_equilibrium
from src.registry import initialise_operator

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice


@initialise_operator(
    name="binary_collision",
    requires="[initialisation] radius, speed",
    impact_parameter="B = chi / (2 r), default 0",
    gap="cells between the droplet surfaces, default 20",
)
def init_binary_collision_2d(
    grid_shape: tuple[int, int, int],
    lattice: Lattice,
    *,
    rho_l: float,
    rho_v: float,
    interface_width: int,
    radius: float,
    speed: float,
    impact_parameter: float = 0.0,
    gap: float = 20.0,
    **_kwargs: object,
) -> jnp.ndarray:
    """Initialise two droplets moving towards each other along ``x``.

    Args:
        grid_shape: Spatial dimensions ``(nx, ny, 1)``.
        lattice: :class:`~src.lattice.lattice.Lattice`.
        rho_l: Liquid density.
        rho_v: Vapour density.
        interface_width: Interface width ``W`` of the tanh profile.
        radius: Droplet radius ``r`` in cells.
        speed: Each droplet's speed ``U_0``.
        impact_parameter: ``B``; the centres are offset by ``2 r B`` in ``y``.
        gap: Distance between the two droplet surfaces along ``x``.
        **_kwargs: Other initialisation keys (ignored).

    Returns:
        Initial populations ``f``, shape ``(nx, ny, 1, q, 1)``.
    """
    nx, ny, nz = grid_shape
    x, y = jnp.meshgrid(jnp.arange(nx, dtype=float), jnp.arange(ny, dtype=float), indexing="ij")
    x_mid, y_mid = (nx - 1) / 2.0, (ny - 1) / 2.0
    offset_x = radius + gap / 2.0
    offset_y = radius * impact_parameter  # half of chi = 2 r B

    phi = jnp.ones((nx, ny))
    for sign in (-1.0, 1.0):
        distance = jnp.sqrt((x - (x_mid + sign * offset_x)) ** 2 + (y - (y_mid - sign * offset_y)) ** 2)
        phi = phi + jnp.tanh(2.0 * (radius - distance) / interface_width)

    rho = (rho_v + rho_l) / 2.0 + (rho_l - rho_v) / 2.0 * phi
    # sign(x_mid - x) rather than the paper's x <= Lx/2 split: zero on an odd
    # grid's centre column, so the two halves carry equal and opposite momentum.
    u_x = jnp.sign(x_mid - x) * speed * (1.0 + phi) / 2.0

    u = jnp.zeros((nx, ny, nz, 1, lattice.d)).at[:, :, 0, 0, 0].set(u_x)
    return compute_equilibrium(rho.reshape(nx, ny, nz, 1, 1), u, lattice)
