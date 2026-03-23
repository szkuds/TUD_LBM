"""Dynamic simulation state for TUD-LBM.

Provides :class:`State` and :class:`WettingState` — the pytree-compatible
carry objects used inside ``jax.lax.scan``.

Public API::

    from state import State, WettingState
"""

from state.state import State, WettingState

__all__ = [
    "State",
    "WettingState",
]
