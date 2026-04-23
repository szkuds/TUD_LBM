"""General multiphase bubble/droplet initialisation.

Defines N circular inclusions from config-provided fractional centres and radii.
"""

from __future__ import annotations

import jax.numpy as jnp

from operators.equilibrium._equilibrium import compute_equilibrium
from registry import initialise_operator
from setup.lattice import Lattice


@initialise_operator(
    name="multiphase_bubbles",
    requires="[initialisation] centres, radii",
    dispersed="'vapour' (default) | 'liquid'",
)
def init_multiphase_bubbles(
    nx: int,
    ny: int,
    lattice: Lattice,
    *,
    rho_l: float = 1.0,
    rho_v: float = 0.33,
    interface_width: int = 4,
    centres,
    radii,
    dispersed: str = "vapour",
    **kwargs,
) -> jnp.ndarray:
    """Initialise multiple diffuse-interface bubbles/droplets.

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        lattice: :class:`~setup.lattice.Lattice`.
        rho_l: Liquid density.
        rho_v: Vapour density.
        interface_width: Diffuse-interface thickness.
        centres: Sequence of ``[fx, fy]`` fractional centres in ``[0, 1]``.
        radii: Sequence of radius fractions of ``min(nx, ny)``.
        dispersed: ``"vapour"`` for low-density inclusions in liquid,
            or ``"liquid"`` for high-density inclusions in vapour.

    Returns:
        Initial distribution ``f``, shape ``(nx, ny, q, 1)``.
    """
    if dispersed not in {"vapour", "liquid"}:
        raise ValueError("'dispersed' must be 'vapour' or 'liquid'.")

    centres_list = list(centres)
    radii_list = list(radii)
    if not centres_list or not radii_list:
        raise ValueError("'centres' and 'radii' must be non-empty.")
    if len(centres_list) != len(radii_list):
        raise ValueError("'centres' and 'radii' must have the same length.")

    x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    avg = (rho_l + rho_v) / 2.0
    amp = (rho_l - rho_v) / 2.0

    if dispersed == "vapour":
        rho_2d = jnp.full((nx, ny), rho_l)
        combine = jnp.minimum
        sign = 1.0
    else:
        rho_2d = jnp.full((nx, ny), rho_v)
        combine = jnp.maximum
        sign = -1.0

    min_dim = float(min(nx, ny))
    for centre, radius_fraction in zip(centres_list, radii_list, strict=False):
        if len(centre) != 2:
            raise ValueError("Each centre must have exactly two coordinates: [fx, fy].")
        fx, fy = float(centre[0]), float(centre[1])
        if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
            raise ValueError("Centre fractions must be in [0, 1].")

        radius = float(radius_fraction) * min_dim
        cx, cy = fx * nx, fy * ny
        distance = jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        profile = avg + sign * amp * jnp.tanh((distance - radius) / interface_width)
        rho_2d = combine(rho_2d, profile)

    rho = rho_2d.reshape(nx, ny, 1, 1)
    u = jnp.zeros((nx, ny, 1, 2))
    return compute_equilibrium(rho, u, lattice)
