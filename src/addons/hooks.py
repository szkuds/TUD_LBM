"""What an addon contributes to a run: three fixed, typed slots.

An *addon* is behaviour that touches several parts of the simulation at once
(the wetting wall, the electric potential). It cannot be a plain operator,
because no single slot of the step holds it. Instead it returns
:class:`Hooks` filling some of three fixed slots, and
:func:`~src.pipeline.setup.build_setup` merges every active addon's hooks
into them:

==============  =======================================  ==========================
Slot            Shape                                    Merge rule
==============  =======================================  ==========================
``ghost_fill``  ``(padded, substate) -> padded``         composed, name order
``post_step``   ``(setup, prev, new) -> new``            run in name order
``init_state``  ``(setup) -> substate``                  ``ext[name] = substate``
==============  =======================================  ==========================

Addons must be independent: an addon reads and writes only its own
``State.ext[name]``. The ghost fill enforces it: it is handed its own substate,
never ``ext``. An addon that needs another's output is a real coupling and
needs a new slot, which is the one edit to ``build_setup`` this design allows.

A substate is a :class:`typing.NamedTuple`, so a snapshot can flatten it into
top-level keys (:func:`~src.simulation_io.callbacks._state_to_numpy`).
"""

from __future__ import annotations
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    import jax.numpy as jnp
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State

#: An addon's substate, ``State.ext[name]``.
S = TypeVar("S")

if TYPE_CHECKING:
    #: Rewrites the ghost layer of a padded scalar field from the addon's own substate.
    GhostFill = Callable[[jnp.ndarray, S], jnp.ndarray]
    #: The addons' fills composed: reads each substate off ``ext`` by name.
    ExtGhostFill = Callable[[jnp.ndarray, Mapping[str, Any]], jnp.ndarray]
    #: Runs after the kernel: ``(setup, state before the step, state after it) -> state``.
    PostStepHook = Callable[[SimulationSetup, State, State], State]
    #: Builds the addon's initial substate, ``State.ext[name]``.
    InitStateHook = Callable[[SimulationSetup], S]


@dataclass(frozen=True, eq=False)
class Hooks(Generic[S]):
    """The slots one addon fills; any slot may be left empty.

    Attributes:
        ghost_fill: Applied to the density gradient and Laplacian after padding;
            never to the chemical-potential gradient. Receives the addon's substate.
        post_step: Hook run after the kernel, receiving the state before and
            after the step.
        init_state: Builds the addon's initial ``State.ext`` entry.
    """

    ghost_fill: GhostFill[S] | None = None
    post_step: PostStepHook | None = None
    init_state: InitStateHook[S] | None = None


def _substate_fill(name: str, fill: GhostFill[Any], padded: jnp.ndarray, ext: Mapping[str, Any]) -> jnp.ndarray:
    """*fill* applied with the substate ``ext[name]``."""
    return fill(padded, ext[name])


def _composed_fill(fills: tuple[ExtGhostFill, ...], padded: jnp.ndarray, ext: Mapping[str, Any]) -> jnp.ndarray:
    """*fills* applied in order."""
    for fill in fills:
        padded = fill(padded, ext)
    return padded


def compose_ghost_fill(addons: tuple[tuple[str, Hooks], ...]) -> ExtGhostFill | None:
    """Compose the addons' ghost fills in their (name) order; ``None`` when none fills.

    Args:
        addons: ``(name, hooks)`` pairs, as stored on ``SimulationSetup.addons``.

    Returns:
        A single ``(padded, ext) -> padded`` handing each fill its own
        ``ext[name]``, or ``None`` so the differential operators are built
        without a fill.
    """
    fills = tuple(
        functools.partial(_substate_fill, name, hooks.ghost_fill)
        for name, hooks in addons
        if hooks.ghost_fill is not None
    )
    if not fills:
        return None
    if len(fills) == 1:
        return fills[0]
    return functools.partial(_composed_fill, fills)
