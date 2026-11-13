"""``multiphase_bubbles`` reproduces the initialisers it replaced, exactly.

The expected fields are the removed initialisers' own formulas. The three wall
droplets used a tanh twice as steep as ``multiphase_bubbles``, so they need
``interface_width`` halved in ``[initialisation]``.
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
import pytest
from src.lattice.lattice import build_lattice
from src.operators.equilibrium._equilibrium import compute_equilibrium
from src.operators.initialise import build_initialise_fn

#: ``(nx, ny), rho_l, rho_v, W`` with bits to spare. Grids and densities exist on
#: which the two differ by up to 2e-15: a radius fraction times ``min(nx, ny)``
#: cannot always hit the old radius, and ``avg - amp`` can land an ulp below
#: ``rho_v``, which ``multiphase_bubbles`` clamps.
_PARAMETER_SETS = [
    ((64, 64), 1.0, 0.33, 4),
    ((101, 51), 12.18, 0.015, 5),
    ((65, 33), 2.0, 0.1, 4),
]


def _legacy_rho(nx: int, ny: int, rho_l: float, rho_v: float, width: int, name: str) -> jnp.ndarray:
    x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    avg, amp = (rho_l + rho_v) / 2.0, (rho_l - rho_v) / 2.0
    if name == "multiphase_bubble_top":
        distance = jnp.sqrt((x - nx // 2) ** 2 + (ny - 1 - y) ** 2)
        return avg + amp * jnp.tanh((distance - min(nx, ny) // 4) / width)
    wall = ny - 1 - y if name == "wetting_drop_top" else y
    distance = jnp.sqrt((x - nx // 2) ** 2 + wall**2)
    radius = ny / 3.3 if name == "wetting_chem_step" else ny / 3.33
    return avg + amp * jnp.tanh(2.0 * (radius - distance) / width)


def _replacement(nx: int, ny: int, width: int, name: str) -> dict[str, object]:
    """The ``[initialisation]`` keys that stand in for legacy initialiser *name*."""
    fx = (nx // 2) / (nx - 1)
    if name == "multiphase_bubble_top":
        radius = (min(nx, ny) // 4) / min(nx, ny)
        return {"centres": [[fx, 1.0]], "radii": [radius], "interface_width": width}
    fy = 1.0 if name == "wetting_drop_top" else 0.0
    radius = (ny / 3.3 if name == "wetting_chem_step" else ny / 3.33) / min(nx, ny)
    return {"centres": [[fx, fy]], "radii": [radius], "dispersed": "liquid", "interface_width": width / 2}


@pytest.mark.parametrize("name", ["wetting", "wetting_chem_step", "wetting_drop_top", "multiphase_bubble_top"])
@pytest.mark.parametrize(("shape", "rho_l", "rho_v", "width"), _PARAMETER_SETS)
def test_the_field_is_reproduced_exactly(
    name: str, shape: tuple[int, int], rho_l: float, rho_v: float, width: int
) -> None:
    lattice = build_lattice("D2Q9")
    nx, ny = shape
    rho = _legacy_rho(nx, ny, rho_l, rho_v, width, name).reshape(nx, ny, 1, 1, 1)
    expected = compute_equilibrium(rho, jnp.zeros((nx, ny, 1, 1, lattice.d)), lattice)

    kwargs = _replacement(nx, ny, width, name)
    result = build_initialise_fn("multiphase_bubbles")((nx, ny, 1), lattice, rho_l=rho_l, rho_v=rho_v, **kwargs)

    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
