"""Stored result of the differential-operators factory.

:class:`DifferentialOperators` is the NamedTuple of pre-built callables
that lives on :class:`~setup.simulation_setup.SimulationSetup`.  All three
callables share the signature ``f(grid: jnp.ndarray) -> jnp.ndarray`` so
the simulation loop is completely uniform.

.. note::

   This NamedTuple contains Python *callables*, not JAX array leaves.
   Because :class:`SimulationSetup` is closed over (not passed through
   ``lax.scan``), this is correct and intentional.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import NamedTuple
import jax.numpy as jnp


class DifferentialOperators(NamedTuple):
    """Pre-built, jitted differential operators ready for the simulation loop.

    All three callables have signature
    ``f(grid: jnp.ndarray) -> jnp.ndarray`` so the loop never needs to
    know about pad_modes, wetting, or BCs.

    Attributes:
        grad_standard: Standard LBM-stencil gradient.  **Always** built
            from pad_modes only, independent of wetting.
            Use this for chemical_potential.
        grad_field: Gradient for density / order-parameter fields.
            Identical to *grad_standard* when wetting is off; includes
            ghost-cell correction when wetting is on.
        laplacian: LBM-stencil Laplacian.
            Currently wetting-independent; add a wetting variant in the
            factory if needed.
    """

    grad_standard: Callable[[jnp.ndarray], jnp.ndarray]
    grad_field: Callable[[jnp.ndarray], jnp.ndarray]
    laplacian: Callable[[jnp.ndarray], jnp.ndarray]
