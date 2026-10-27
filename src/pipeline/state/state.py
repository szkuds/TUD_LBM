"""Dynamic simulation state — a JAX pytree.

:class:`State` is a :class:`typing.NamedTuple` and therefore automatically a
valid JAX pytree. It is the *carry* in the ``jax.lax.scan`` time-stepping loop.

Every field is a ``jax.Array`` or, for ``ext``, a dict of feature substates.
No mutable Python objects are stored here.

Usage::

    import jax.numpy as jnp
    from src.pipeline.state import State

    state = State(f=jnp.zeros((64, 64, 1, 9, 1)), t=jnp.array(0))
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

if TYPE_CHECKING:
    import jax.numpy as jnp


class State(NamedTuple):
    """Complete dynamic simulation state — the *carry* in ``lax.scan``.

    Only what a next step cannot recompute is carried. Density, velocity,
    forces and pressure are functions of this state, evaluated by
    :func:`~src.pipeline.step.observe` where they are wanted; the physics
    itself reads only ``f``, ``t`` and ``ext``.

    Attributes:
        f: Population distributions, shape ``(nx, ny, nz, q, 1)``.
        t: Current timestep — scalar ``jax.Array``. Not derivable: the gravity
            ramps and restarts read it.
        ext: Feature substates keyed by feature name, e.g. ``ext["wetting"]``
            (a :class:`~src.operators.wetting._params.WettingState`) or
            ``ext["electric"]`` (``{"h": ...}``). Not derivable: they carry their
            own history. Each feature writes only its own key; empty for a run
            with no active feature. The keys are fixed by
            :func:`~src.pipeline.runner.init_state`, so the carry structure stays
            constant across ``lax.scan`` iterations.
    """

    f: jnp.ndarray
    t: jnp.ndarray
    ext: dict[str, Any] = {}  # noqa: RUF012 - never mutated: features replace it with `ext | {...}`
