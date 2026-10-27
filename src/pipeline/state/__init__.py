"""Dynamic simulation state for TUD-LBM.

Provides :class:`State`, the pytree carry used inside ``jax.lax.scan``: the
populations, the time and the feature substates (``State.ext``), which the
features initialise through their ``init_state`` slot (:mod:`src.features`).
Every other field is derived by :func:`src.pipeline.step.observe`.

Public API::

    from src.pipeline.state import State, WettingState
"""

from src.operators.wetting._params import WettingState
from src.pipeline.state.state import State

__all__ = [
    "State",
    "WettingState",
]
