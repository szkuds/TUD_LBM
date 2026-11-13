"""Well-balanced forcing source term — pure function.

Couples the total force into the populations as the collision source term; it
is not itself a force, so it lives in its own package rather than under
``operators/force``.

Uses the density gradient operator for computing gravity corrections.
"""

from __future__ import annotations
import functools
import operator
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from src.registry import source_term_operator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from src.lattice.lattice import Lattice
    from src.protocols import DifferentialOperator


def _dot(a: Sequence, b: Sequence) -> jnp.ndarray:
    """``a[0] * b[0] + a[1] * b[1] + ...``, in that order."""
    return functools.reduce(operator.add, (x * y for x, y in zip(a, b, strict=True)))


@source_term_operator(name="wb")
def compute_source(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    force: jnp.ndarray,
    lattice: Lattice,
    *,
    gradient: DifferentialOperator,
) -> jnp.ndarray:
    """Compute the well-balanced forcing source term.

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, nz, 1, d)``.
        force: Force field, shape ``(nx, ny, nz, 1, d)``.
        lattice: :class:`~setup.lattice.Lattice`.
        gradient: Standard LBM-stencil gradient callable
            (grid) → gradient. Used for density gradient.

    Returns:
        Source term, shape ``(nx, ny, nz, q, 1)``.
    """
    q = lattice.q
    d = lattice.d

    # Pre-extract lattice data as numpy for JIT safety
    w_np = np.array(lattice.w)[0, 0, 0, :, 0]  # (q,) as numpy — avoids tracing issues
    c_np = np.array(lattice.c)[0, 0, 0]  # (q, d) as numpy — avoids tracing issues

    # Fields without their singleton axis, one array per component
    u_d = [u[:, :, :, 0, a] for a in range(d)]
    force_d = [force[:, :, :, 0, a] for a in range(d)]

    # Density gradient via LBM-stencil operator
    grad_rho_5d = gradient(rho)  # (nx, ny, nz, 1, d)
    grad_rho_d = [grad_rho_5d[:, :, :, 0, a] for a in range(d)]

    # Corrected force
    force_cor = [force_d[a] + grad_rho_d[a] / 3.0 for a in range(d)]

    source = jnp.zeros((*rho.shape[:3], q))

    # Every term stays inside the loop: XLA fuses this form to the same bits it always gave.
    for i in range(q):
        ci = [float(component) for component in c_np[i]]
        wi = float(w_np[i])

        cu = _dot(ci, u_d)
        cf = _dot(ci, force_d)
        cf_cor = _dot(ci, force_cor)
        uf_cor = _dot(u_d, force_cor)
        u_grad_rho = _dot(u_d, grad_rho_d)

        source = source.at[..., i].set(
            wi * (3.0 * cf + 9.0 * cf_cor * cu - 3.0 * uf_cor + 0.5 * (3.0 * _dot(ci, ci) - d) * u_grad_rho),
        )

    return source[..., None]
