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
        f: Populations, shape ``(nx, ny, q, 1)``.
        lattice: :class:`~setup.lattice.Lattice`.
        force: Optional external force field, shape ``(nx, ny, 1, 2)``.
            When provided the velocity is corrected:
            ``u_eq = u + force / (2 rho)``.

    Returns:
        ``(rho, u)`` when *force* is ``None``, or
        ``(rho, u_eq, force)`` when *force* is given.

        * ``rho``: shape ``(nx, ny, 1, 1)``
        * ``u`` / ``u_eq``: shape ``(nx, ny, 1, 2)``
    """
    cx = lattice.c[0]  # (q,)
    cy = lattice.c[1]  # (q,)
    q = lattice.q

    # Density — zeroth moment
    rho = jnp.sum(f, axis=2, keepdims=True)  # (nx, ny, 1, 1)

    # Momentum — first moment
    cx4 = cx.reshape((1, 1, q, 1))
    cy4 = cy.reshape((1, 1, q, 1))
    ux = jnp.sum(f * cx4, axis=2, keepdims=True)
    uy = jnp.sum(f * cy4, axis=2, keepdims=True)
    u = jnp.concatenate([ux, uy], axis=-1) / rho  # (nx, ny, 1, 2)

    if force is not None:
        u_eq = u + force / (2.0 * rho)
        return rho, u_eq, force

    return rho, u
