"""Single-phase macroscopic field computation — pure function.

Extracted from :class:`simulation_operators.macroscopic.Macroscopic`.
Computes density (zeroth moment) and velocity (first moment / density).
"""

from __future__ import annotations
import jax.numpy as jnp
from registry import macroscopic_operator
from setup.lattice import Lattice


@macroscopic_operator(name="standard")
def compute_macroscopic(
    f: jnp.ndarray,
    lattice: Lattice,
    force: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute density and velocity from population distributions.

    Args:
        f: Populations, shape ``(nx, ny, nz, q, 1)``.
        lattice: :class:`~setup.lattice.Lattice`.
        force: Optional external force field, shape ``(nx, ny, nz, 1, 2)``.
            When provided the velocity is corrected:
            ``u_eq = u + force / (2 rho)``.

    Returns:
        ``(rho, u)`` when *force* is ``None``, or
        ``(rho, u_eq, force)`` when *force* is given.

        * ``rho``: shape ``(nx, ny, nz, 1, 1)``
        * ``u`` / ``u_eq``: shape ``(nx, ny, nz, 1, 2)``
    """

    # Density — zeroth moment. Sum over q
    rho = jnp.sum(f, axis=-2, keepdims=True)  # (nx, ny, nz, 1, 1)

    # Momentum — first moment
    u = jnp.sum(f * lattice.c, axis=-2, keepdims=True) / rho  # (nx, ny, nz, 1, D)

    if force is not None:
        u_eq = u + force / (2.0 * rho)
        return rho, u_eq, force

    return rho, u
