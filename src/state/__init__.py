"""Dynamic simulation state for TUD-LBM.

Provides :class:`State` and :class:`WettingState` — the pytree-compatible
carry objects used inside ``jax.lax.scan``.

Also provides state builder functions :func:`build_optional_fields` and
:func:`build_extra_state` that orchestrate state composition.

Public API::

    from state import State, WettingState, build_optional_fields, build_extra_state
"""

from state.state import State
from state.state import WettingState
from state._optional_fields import _build_optional_fields
from state._extra_state import _build_extra_state

__all__ = [
    "State",
    "WettingState",
    "build_optional_fields",
    "build_extra_state",
]


def build_optional_fields(setup, nx: int, ny: int, d: int):
    """Build the optional ``force`` and ``force_ext`` fields.

    Returns zero-filled arrays for fields that will be written by the
    step function, and ``None`` for fields that remain unused.
    JAX's ``lax.scan`` requires the pytree structure to be constant
    across iterations, so any field that transitions from ``None`` →
    array must start as zeros.

    Args:
        setup: :class:`~setup.simulation_setup.SimulationSetup`.
        nx: Grid size in x.
        ny: Grid size in y.
        d: Lattice dimension (e.g. 2 for D2Q9).

    Returns:
        A tuple ``(force, force_ext)``:

        * ``force``: Zeros ``(nx, ny, 1, d)`` for multiphase runs,
          ``None`` otherwise.
        * ``force_ext``: Zeros ``(nx, ny, 1, d)`` when forces are active,
          ``None`` otherwise.

    See Also:
        :func:`state._optional_fields._build_optional_fields`
    """
    return _build_optional_fields(setup, nx, ny, d)


def build_extra_state(setup):
    """Collect extra State fields initialised by registered force specs.

    Some force implementations define additional fields that must be
    stored in the State pytree (e.g. electric potential ``h`` for
    electrokinetic flows). This helper iterates over all active force
    specs and collects their initialised field contributions.

    Returns an empty dict when no forces are registered, keeping the
    call site unconditional and simplifying the orchestrator logic.

    Args:
        setup: :class:`~setup.simulation_setup.SimulationSetup`.

    Returns:
        A dictionary mapping field names to initialised arrays.
        Empty when no forces are active.

    See Also:
        :func:`state._extra_state._build_extra_state`
    """
    return _build_extra_state(setup)

