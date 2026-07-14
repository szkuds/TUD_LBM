"""Single-phase macroscopic field computation — pure function.

Extracted from :class:`simulation_operators.macroscopic.Macroscopic`.
Computes density (zeroth moment) and velocity (first moment / density).
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from tud_lbm.registry import macroscopic_operator

if TYPE_CHECKING:
    from tud_lbm.lattice.lattice import Lattice


@macroscopic_operator(name="standard")
def compute_macroscopic(
    f: jnp.ndarray,
    lattice: Lattice,
    force: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Compute density and velocity from population distributions.

    Args:
        f: Populations, shape ``(nx, ny, nz, q, 1)``.
        lattice: :class:`~setup.lattice.Lattice`.
        force: Optional external force field, shape ``(nx, ny, nz, 1, 2)``.
            When provided the velocity is corrected:
            ``u_eq = u + force / (2 rho)``.

    Returns:
        ``(rho, u, force)``; *force* is echoed back (``None`` when not given)
        so the return shape matches the multiphase operator.

        * ``rho``: shape ``(nx, ny, nz, 1, 1)``
        * ``u``: shape ``(nx, ny, nz, 1, 2)``; force-corrected when *force* is given
    """
    # Density — zeroth moment. Sum over q
    rho: jnp.ndarray = jnp.sum(f, axis=-2, keepdims=True)  # (nx, ny, nz, 1, 1)

    # Momentum — first moment
    u: jnp.ndarray = jnp.sum(f * lattice.c, axis=-2, keepdims=True) / rho  # (nx, ny, nz, 1, d)

    if force is not None:
        u: jnp.ndarray = u + force / (2.0 * rho)

    return rho, u, force
