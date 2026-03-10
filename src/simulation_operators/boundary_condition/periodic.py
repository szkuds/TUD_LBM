"""Periodic boundary condition operator.

For periodic boundaries, the streaming step already handles periodicity
via the array roll; no additional post-streaming transformation is required.
This operator is therefore a no-op, but it is registered in the operator
registry so that users can explicitly request ``"periodic"`` in config
files and the boundary-condition dispatch table.
"""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import jit

from app_setup.registry import register_operator
from .base import BoundaryConditionBase


@register_operator("boundary_condition")
class PeriodicBoundaryCondition(BoundaryConditionBase):
    """No-op periodic boundary condition.

    The streaming operator already wraps arrays periodically, so this
    class simply returns ``f_streamed`` unchanged.
    """

    name = "periodic"

    @partial(jit, static_argnums=(0,))
    def __call__(
        self, f_streamed: jnp.ndarray, f_collision: jnp.ndarray
    ) -> jnp.ndarray:
        """Return *f_streamed* unchanged (streaming handles periodicity)."""
        return f_streamed

