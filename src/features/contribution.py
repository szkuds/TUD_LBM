"""What a feature contributes to a run: four fixed, typed slots.

A *feature* is behaviour that touches several parts of the simulation at once
(the wetting wall, the electric potential). It cannot be a plain operator,
because no single slot of the step holds it. Instead it returns a
:class:`Contribution` that fills some of four fixed slots, and
:func:`~src.pipeline.setup.build_setup` merges every active feature's
contribution into them:

==============  =======================================  ==========================
Slot            Shape                                    Merge rule
==============  =======================================  ==========================
``ghost_fill``  ``(padded, ext) -> padded``              composed, name order
``forces``      tuple of :class:`ForceOperator`          concatenated
``post_step``   ``(setup, prev, new) -> new``            run in name order
``init_state``  ``(setup) -> substate``                  ``ext[name] = substate``
==============  =======================================  ==========================

Features must be independent: a feature reads and writes only its own
``State.ext[name]``. One that needs another's output is a real coupling and
needs a new slot, which is the one edit to ``build_setup`` this design allows.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    import jax.numpy as jnp
    from src.operators.protocols import ForceOperator
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State

    #: Rewrites the ghost layer of a padded scalar field, between padding and the stencil.
    GhostFill = Callable[[jnp.ndarray, Mapping[str, Any]], jnp.ndarray]
    #: Runs after the kernel: ``(setup, state before the step, state after it) -> state``.
    PostStepHook = Callable[[SimulationSetup, State, State], State]
    #: Builds the feature's initial substate, ``State.ext[name]``.
    InitStateHook = Callable[[SimulationSetup], Any]


@dataclass(frozen=True, eq=False)
class Contribution:
    """The slots one feature fills; any slot may be left empty.

    Attributes:
        ghost_fill: Applied to the density gradient and Laplacian after padding;
            never to the chemical-potential gradient.
        forces: Forces that exist only with this feature, appended to the
            configured ``*_force`` forces.
        post_step: Hook run after the kernel, receiving the state before and
            after the step.
        init_state: Builds the feature's initial ``State.ext`` entry.
    """

    ghost_fill: GhostFill | None = None
    forces: tuple[ForceOperator, ...] = ()
    post_step: PostStepHook | None = None
    init_state: InitStateHook | None = None


def compose_ghost_fill(features: tuple[tuple[str, Contribution], ...]) -> GhostFill | None:
    """Compose the features' ghost fills in their (name) order; ``None`` when none fills.

    Args:
        features: ``(name, contribution)`` pairs, as stored on ``SimulationSetup.features``.

    Returns:
        A single ``(padded, ext) -> padded``, or ``None`` so the differential
        operators are built without a fill.
    """
    fills = tuple(c.ghost_fill for _, c in features if c.ghost_fill is not None)
    if not fills:
        return None
    if len(fills) == 1:
        return fills[0]

    def composed(padded: jnp.ndarray, ext: Mapping[str, Any]) -> jnp.ndarray:
        for fill in fills:
            padded = fill(padded, ext)
        return padded

    return composed
