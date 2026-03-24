"""MRT (Multiple Relaxation Time) collision operator — pure function.

Extracted from :class:`simulation_operators.collision_models.CollisionMRT`.
Uses the D2Q9 moment transformation matrix *M* and its inverse *M_INV*.

The collision step in moment space is:

.. math::

    f_{\\text{col}} = f + M^{-1} K M (f^{\\text{eq}} - f)
                    + M^{-1} (I - K/2) M \\, S

where *K* = ``diag(k_diag)`` is the diagonal relaxation-rate matrix and
*S* is the optional source term.
"""

from __future__ import annotations
import jax.numpy as jnp
from registry import collision_model

# ── D2Q9 moment basis (compile-time constant) ───────────────────────

M = jnp.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
    ],
    dtype=jnp.float32,
)

M_INV = jnp.linalg.inv(M)


@collision_model(name="mrt")
def collide_mrt(
    f: jnp.ndarray,
    feq: jnp.ndarray,
    tau: float,
    source: jnp.ndarray | None = None,
    k_diag: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """MRT collision using the D2Q9 moment transformation matrix.

    Args:
        f: Pre-collision populations, shape ``(nx, ny, q, 1)``.
        feq: Equilibrium populations, same shape.
        tau: Relaxation time (used to build *k_diag* default if not supplied).
        source: Optional source term, same shape.
        k_diag: Diagonal relaxation rates, shape ``(q,)``.
            If ``None``, a default vector with ``1/tau`` on relevant moments
            is used.

    Returns:
        Post-collision populations, same shape.
    """
    if k_diag is None:
        k_diag = jnp.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0 / tau, 1.0 / tau])

    k = jnp.diag(k_diag)
    identity = jnp.eye(len(k_diag))

    # Pre-compute combined matrices
    mat_f_neq = M_INV @ k @ M  # collision matrix
    mat_source = M_INV @ (identity - k / 2) @ M  # source matrix

    # Apply collision in moment space — work on (..., q) by squeezing last dim
    f_neq_post = jnp.einsum("ij,...j->...i", mat_f_neq, (feq - f)[..., 0])

    if source is not None:
        source_post = jnp.einsum("ij,...j->...i", mat_source, source[..., 0])
        f_post = f[..., 0] + f_neq_post + source_post
    else:
        f_post = f[..., 0] + f_neq_post

    return f_post[..., None]
