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
        f=jnp.zeros((64, 64, 1, 9, 1)),
        rho=jnp.ones((64, 64, 1, 1, 1)),
        u=jnp.zeros((64, 64, 1, 1, 2)),
        t=jnp.array(0),
    )
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple

if TYPE_CHECKING:
    import jax.numpy as jnp


class WettingState(NamedTuple):
    """Dynamic wetting / hysteresis parameters — updated every step.

    All fields are ``jax.Array`` scalars or small arrays so that the
    state can flow through ``jax.lax.scan`` as part of the carry.

    Attributes:
        phi_left: Wetting potential parameter (left).
        phi_right: Wetting potential parameter (right).
        d_rho_left: Density offset parameter (left contact line).
        d_rho_right: Density offset parameter (right contact line).
        ca_left: Current contact angle — left.
        ca_right: Current contact angle — right.
        cll_left: Contact-line location — left.
        cll_right: Contact-line location — right.
    """

    phi_left: jnp.ndarray
    phi_right: jnp.ndarray
    d_rho_left: jnp.ndarray
    d_rho_right: jnp.ndarray
    ca_left: jnp.ndarray
    ca_right: jnp.ndarray
    cll_left: jnp.ndarray
    cll_right: jnp.ndarray


class State(NamedTuple):
    """Complete dynamic simulation state — the *carry* in ``lax.scan``.

    Attributes:
        f: Population distributions, shape ``(nx, ny, nz, q, 1)``.
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, nz, 1, d)``.
        t: Current timestep — scalar ``jax.Array``.
        force: Interaction / body force field (optional), shape ``(nx, ny, nz, 1, d)``.
        force_ext: External force field (optional), shape ``(nx, ny, nz, 1, d)``.
        h: Electric potential field (optional), shape ``(nx, ny, nz, q, 1)``.
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
