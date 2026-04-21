"""Dynamic simulation state for TUD-LBM.

Provides :class:`State` and :class:`WettingState` — the pytree-compatible
carry objects used inside ``jax.lax.scan``.

Also provides state builder functions :func:`build_optional_fields` and
:func:`build_extra_state` that orchestrate state composition.

Public API::

    from state import State, WettingState, build_optional_fields, build_extra_state
"""

from state._extra_state import _build_extra_state
from state._extra_state import _update_extra_state
from state._optional_fields import _build_optional_fields
from state.state import State
from state.state import WettingState

__all__ = [
    "State",
    "WettingState",
    "build_extra_state",
    "build_optional_fields",
    "update_extra_state",
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
    """Collect extra State fields initialised by active extra-state plugins.

    Args:
        setup: :class:`~setup.simulation_setup.SimulationSetup`.

    Returns:
        A dictionary mapping field names to initialised arrays.
        Empty when no plugins are active.

    See Also:
        :func:`state._extra_state._build_extra_state`
    """
    return _build_extra_state(setup)


def update_extra_state(setup, prev_state, new_state, **context):
    """Apply plugin-driven extra-state updates after one step."""
    return _update_extra_state(setup, prev_state, new_state, **context)
