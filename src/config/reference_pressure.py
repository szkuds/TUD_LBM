"""Hydrostatic reference pressure ``p_g = rho_0 g.x`` of a buoyancy-referenced gravity.

Built only by :attr:`SimulationConfig.reference_pressure
<src.config.simulation_config.SimulationConfig.reference_pressure>`.

Zhang, Guo & Wang (Phys. Fluids 34, 012110, 2022), Eqs. 25-26: with a reference
density ``rho_0`` the gravity force is only the excess ``-(rho - rho_0) g``, and
the ``rho_0 g`` part enters as ``p_g`` in the equilibrium's second moment and as
``-grad p_g`` in the source term's velocity-force product. The field is
measured from the domain centre so ``|p_g|`` stays as small as possible; an
additive constant has no dynamical effect.
"""

from __future__ import annotations
from typing import Any
from typing import NamedTuple
import jax.numpy as jnp
import numpy as np


class ReferencePressure(NamedTuple):
    """``p_g`` and its gradient ``rho_0 (g_x, g_y)`` at every lattice node.

    Attributes:
        field: ``p_g``, shape ``(nx, ny, nz, 1, 1)``.
        gradient: ``grad p_g``, shape ``(nx, ny, nz, 1, 2)``.
    """

    field: jnp.ndarray
    gradient: jnp.ndarray


def build_reference_pressure(gravity: dict[str, Any] | None, grid_shape: tuple[int, ...]) -> ReferencePressure | None:
    """The reference pressure of a ``[gravity_force]`` section with ``reference_density``, else ``None``."""
    if gravity is None or "reference_density" not in gravity:
        return None

    from src.operators.force._gravity import gravity_vector

    g_x, g_y = gravity_vector(gravity)
    rho_0 = float(gravity["reference_density"])
    nx, ny, nz = grid_shape[:3]
    x, y = np.meshgrid(np.arange(nx) - (nx - 1) / 2.0, np.arange(ny) - (ny - 1) / 2.0, indexing="ij")
    field = rho_0 * (g_x * x + g_y * y)
    gradient = np.broadcast_to(np.array([rho_0 * g_x, rho_0 * g_y]), (nx, ny, nz, 1, 2))
    return ReferencePressure(
        field=jnp.asarray(np.broadcast_to(field[:, :, None, None, None], (nx, ny, nz, 1, 1))),
        gradient=jnp.asarray(gradient),
    )
