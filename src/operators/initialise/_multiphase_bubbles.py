"""General multiphase bubble/droplet initialisation.

Defines N circular inclusions from config-provided fractional centres and radii.
"""

from __future__ import annotations
import functools
import operator
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.equilibrium._equilibrium import compute_equilibrium
from src.registry import initialise_operator

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice

#: How a centre's coordinate count is spelled in its error, by lattice dimension.
_COORDINATE_COUNT = {2: "two coordinates: [fx, fy]", 3: "three coordinates: [fx, fy, fz]"}


@initialise_operator(
    name="multiphase_bubbles",
    requires="[initialisation] centres, radii",
    dispersed="'vapour' (default) | 'liquid'",
)
def init_multiphase_bubbles_2d(
    grid_shape: tuple[int, int, int],
    lattice: Lattice,
    *,
    rho_l: float = 1.0,
    rho_v: float = 0.33,
    interface_width: int = 4,
    centres: list,
    radii: list,
    dispersed: str = "vapour",
    **_kwargs: object,
) -> jnp.ndarray:
    """Initialise multiple diffuse-interface bubbles/droplets.

    Args:
        grid_shape: Spatial dimensions ``(nx, ny, nz)``; nz=1 gives pseudo-3D.
        lattice: :class:`~setup.lattice.Lattice`.
        rho_l: Liquid density.
        rho_v: Vapour density.
        interface_width: Diffuse-interface thickness.
        centres: Sequence of fractional centres in ``[0, 1]``: ``[fx, fy]``, or
            ``[fx, fy, fz]`` on a 3D lattice.
        radii: Sequence of radius fractions of the smallest extent.
        dispersed: ``"vapour"`` for low-density inclusions in liquid,
            or ``"liquid"`` for high-density inclusions in vapour.
        **kwargs: Additional arguments (ignored).

    Returns:
        Initial distribution ``f``, shape ``(nx, ny, nz, q, 1)``.
    """
    nx, ny, nz = grid_shape
    d = lattice.d
    if d == min(_COORDINATE_COUNT) and nz != 1:
        msg = "Multiphase bubble initialisation supports nz=1 (pseudo-3D) for now."
        raise ValueError(msg)

    if dispersed not in {"vapour", "liquid"}:
        msg = "'dispersed' must be 'vapour' or 'liquid'."
        raise ValueError(msg)

    centres_list = list(centres)
    radii_list = list(radii)
    if not centres_list or not radii_list:
        msg = "'centres' and 'radii' must be non-empty."
        raise ValueError(msg)
    if len(centres_list) != len(radii_list):
        msg = "'centres' and 'radii' must have the same length."
        raise ValueError(msg)

    extents = grid_shape[:d]
    axes = jnp.meshgrid(*(jnp.arange(n) for n in extents), indexing="ij")
    avg = (rho_l + rho_v) / 2.0
    amp = (rho_l - rho_v) / 2.0

    if dispersed == "vapour":
        rho_field = jnp.full(extents, rho_l)
        combine = jnp.minimum
        sign = 1.0
    else:
        rho_field = jnp.full(extents, rho_v)
        combine = jnp.maximum
        sign = -1.0

    min_dim = float(min(extents))
    for centre, radius_fraction in zip(centres_list, radii_list, strict=False):
        if len(centre) != d:
            msg = f"Each centre must have exactly {_COORDINATE_COUNT[d]}."
            raise ValueError(msg)
        fractions = [float(coordinate) for coordinate in centre]
        if not all(0.0 <= fraction <= 1.0 for fraction in fractions):
            msg = "Centre fractions must be in [0, 1]."
            raise ValueError(msg)

        radius = float(radius_fraction) * min_dim
        cells = [round(fraction * (n - 1)) for fraction, n in zip(fractions, extents, strict=True)]
        distance = jnp.sqrt(
            functools.reduce(operator.add, ((axis - cell) ** 2 for axis, cell in zip(axes, cells, strict=True)))
        )
        profile = avg + sign * amp * jnp.tanh((distance - radius) / interface_width)
        rho_field = combine(rho_field, profile)

    rho = rho_field.reshape(nx, ny, nz, 1, 1)
    u = jnp.zeros((nx, ny, nz, 1, lattice.d))
    return compute_equilibrium(rho, u, lattice)
