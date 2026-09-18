r"""MRT (Multiple Relaxation Time) collision operator — pure function.

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
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.registry import collision_model

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    dtype=jnp.float64,
)

M_INV = jnp.linalg.inv(M)

#: Indices of the shear-stress moments ``p_xx`` and ``p_xy`` in the basis above
#: (rows 7 and 8). These two alone set the kinematic viscosity, exactly as
#: ``omega = 1/tau`` does in BGK, so they are the entries of ``k_diag`` that
#: ``tau`` owns. Rows 0, 3 and 5 are conserved (density and momentum) and rows
#: 1, 2, 4 and 6 are free stability knobs that do not affect viscosity.
SHEAR_MOMENTS = (7, 8)

#: Relaxation rates for everything ``tau`` does not own, used when no ``k_diag``
#: is configured. Entries ``SHEAR_MOMENTS`` are set by :func:`couple_shear_to_tau`,
#: so the values carried here for them are never read.
_FREE_RATE_DEFAULTS = (0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)


def couple_shear_to_tau(k_diag: Sequence[float], tau: float) -> tuple[float, ...]:
    """Return *k_diag* with its shear-stress rates replaced by ``1/tau``.

    A configured ``k_diag`` otherwise sets the viscosity independently of
    ``tau``, so ``nu = cs2*(tau - 0.5)`` — which every reported ``Oh``, ``La``,
    ``Re`` and ``Ar`` is computed from — would describe a run that relaxed at
    some other rate. Applied by :class:`~src.config.SimulationConfig`, so the
    coupling holds for the saved config as well as the run.

    Args:
        k_diag: Relaxation rates, length ``q``.
        tau: Relaxation time.

    Raises:
        ValueError: If *k_diag* is too short to hold the shear moments.
    """
    rates = [float(rate) for rate in k_diag]
    if len(rates) <= max(SHEAR_MOMENTS):
        msg = f"k_diag must have at least {max(SHEAR_MOMENTS) + 1} entries, got {len(rates)}"
        raise ValueError(msg)
    for index in SHEAR_MOMENTS:
        rates[index] = 1.0 / tau
    return tuple(rates)


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
        f: Pre-collision populations, shape ``(nx, ny, nz, q, 1)``.
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
        # The same call the config applies to a configured k_diag, so the two
        # cannot express the coupling differently.
        k_diag = jnp.asarray(couple_shear_to_tau(_FREE_RATE_DEFAULTS, tau))

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
