"""BGK (Bhatnagar–Gross–Krook) collision operator — pure function.

Extracted from :class:`simulation_operators.collision_models.CollisionBGK`.
The formula matches the legacy class exactly:

.. math::

    f_{\\text{col}} = \\left(1 - \\frac{1}{\\tau}\\right) f
                    + \\frac{1}{\\tau} f^{\\text{eq}}
                    + \\left(1 - \\frac{1}{2\\tau}\\right) S

where *S* is the optional source term.
"""

from __future__ import annotations

import jax.numpy as jnp

from registry import collision_model


@collision_model(name="bgk")
def collide_bgk(
    f: jnp.ndarray,
    feq: jnp.ndarray,
    tau: float,
    source: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """BGK collision: f_col = (1 − 1/τ) f + (1/τ) feq [+ (1 − 1/(2τ)) source].

    Args:
        f: Pre-collision populations, shape ``(nx, ny, q, 1)``.
        feq: Equilibrium populations, same shape.
        tau: Relaxation time (> 0.5).
        source: Optional source term, same shape.

    Returns:
        Post-collision populations, same shape.
    """
    omega = 1.0 / tau
    f_col = (1.0 - omega) * f + omega * feq
    if source is not None:
        f_col = f_col + (1.0 - 0.5 * omega) * source
    return f_col
