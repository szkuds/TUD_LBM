"""Dynamic simulation state — JAX pytrees.

Both :class:`State` and :class:`WettingState` are
:class:`typing.NamedTuple` subclasses and therefore automatically
valid JAX pytrees.  They are designed to be the *carry* in a
``jax.lax.scan`` time-stepping loop.

Every field is either a ``jax.Array`` or ``None`` (which JAX treats
as a static leaf).  No mutable Python objects are stored here.

Usage::

    import jax.numpy as jnp
    from state.state import State

    state = State(
        f=jnp.zeros((64, 64, 9, 1)),
        rho=jnp.ones((64, 64, 1, 1)),
        u=jnp.zeros((64, 64, 1, 2)),
        t=jnp.array(0),
    )
"""

from __future__ import annotations
from typing import Any
from typing import NamedTuple
import jax.numpy as jnp


class WettingState(NamedTuple):
    """Dynamic wetting / hysteresis parameters — updated every step.

    All fields are ``jax.Array`` scalars or small arrays so that the
    state can flow through ``jax.lax.scan`` as part of the carry.

    Attributes:
        d_rho_left: Density offset parameter (left contact line).
        d_rho_right: Density offset parameter (right contact line).
        phi_left: Wetting potential parameter (left).
        phi_right: Wetting potential parameter (right).
        ca_left: Current contact angle — left.
        ca_right: Current contact angle — right.
        cll_left: Contact-line location — left.
        cll_right: Contact-line location — right.
        opt_state_left: ``optax`` optimizer state (left) — a valid pytree.
        opt_state_right: ``optax`` optimizer state (right) — a valid pytree.
    """

    d_rho_left: jnp.ndarray
    d_rho_right: jnp.ndarray
    phi_left: jnp.ndarray
    phi_right: jnp.ndarray
    ca_left: jnp.ndarray
    ca_right: jnp.ndarray
    cll_left: jnp.ndarray
    cll_right: jnp.ndarray
    opt_state_left: Any = None  # optax optimizer state (pytree)
    opt_state_right: Any = None  # optax optimizer state (pytree)


class State(NamedTuple):
    """Complete dynamic simulation state — the *carry* in ``lax.scan``.

    Attributes:
        f: Population distributions, shape ``(nx, ny, q, 1)``.
        rho: Density field, shape ``(nx, ny, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, 1, d)``.
        t: Current timestep — scalar ``jax.Array``.
        force: Interaction / body force field (optional).
        force_ext: External force field (optional).
        h: Electric potential field (optional).
        wetting: Dynamic wetting state (``None`` for non-wetting runs).
    """

    f: jnp.ndarray
    rho: jnp.ndarray
    u: jnp.ndarray
    t: jnp.ndarray
    force: jnp.ndarray | None = None
    force_ext: jnp.ndarray | None = None
    h: jnp.ndarray | None = None
    wetting: WettingState | None = None
