"""Well-balanced source term with a hydrostatic reference pressure (Zhang, Guo & Wang 2022).

Registered as ``"wb_referenced"``. ``reference_gradient`` is a keyword-only parameter
the ``SourceTermOperator`` protocol lacks, so ``build_setup`` binds it with
``functools.partial``.

Eq. 26 puts ``F - grad p_g + cs2 grad rho`` in the velocity-force product
``u (.) : (c_i c_i - cs2 I) / cs4`` while the first-moment term keeps ``F``. The
``wb`` source (:mod:`._source_well_balanced`, called, never edited) supplies every
term without ``p_g``; this adds the ``-grad p_g`` part,

    ``-w_i (9 (c_i . u)(c_i . grad p_g) - 3 u . grad p_g)``,

which carries no mass and no momentum and shifts the second moment by
``-(u grad p_g + grad p_g u)``.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.source_term._source_well_balanced import compute_source as compute_source_wb
from src.registry import source_term_operator

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice
    from src.operators.protocols import DifferentialOperator


@source_term_operator(name="wb_referenced")
def compute_source(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    force: jnp.ndarray,
    lattice: Lattice,
    *,
    gradient: DifferentialOperator,
    reference_gradient: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the well-balanced source term with the ``-grad p_g`` correction.

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, nz, 1, 2)``.
        force: Total force, shape ``(nx, ny, nz, 1, 2)``.
        lattice: :class:`~src.lattice.lattice.Lattice`.
        gradient: Density gradient operator, as for ``wb``.
        reference_gradient: ``grad p_g``, shape ``(nx, ny, nz, 1, 2)``.

    Returns:
        Source term, shape ``(nx, ny, nz, q, 1)``.
    """
    cu = jnp.sum(lattice.c * u, axis=-1, keepdims=True)  # (nx, ny, nz, q, 1)
    cg = jnp.sum(lattice.c * reference_gradient, axis=-1, keepdims=True)  # (nx, ny, nz, q, 1)
    ug = jnp.sum(u * reference_gradient, axis=-1, keepdims=True)  # (nx, ny, nz, 1, 1)
    correction = -lattice.w * (9.0 * cu * cg - 3.0 * ug)
    return compute_source_wb(rho, u, force, lattice, gradient=gradient) + correction
