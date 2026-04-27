"""Private helpers for plugin-driven extra-state init and update."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from setup.simulation_setup import SimulationSetup


_WARNED_MESSAGES: set[str] = set()


def _warn_once(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _build_extra_state(setup: SimulationSetup) -> dict[str, Any]:
    """Collect extra State fields initialised by active extra-state plugins.

    Args:
        setup: :class:`~setup.simulation_setup.SimulationSetup`.

    Returns:
        A dictionary mapping field names to initialised values.
    """
    extra: dict[str, Any] = {}

    for plugin in setup.extra_state_plugins:
        plugin_extra = plugin.init_state(setup)
        for key, value in plugin_extra.items():
            if key in extra:
                _warn_once(
                    f"Extra-state key '{key}' was produced by multiple plugins; keeping the first value."
                )
                continue
            extra[key] = value

    return extra


def _update_extra_state(
    setup: SimulationSetup,
    prev_state: Any,
    new_state: Any,
    **context: Any,
) -> Any:
    """Apply active extra-state plugins after a step and return updated state."""
    updated = new_state
    for plugin in setup.extra_state_plugins:
        next_state = plugin.update_state(setup, prev_state, updated, **context)
        if next_state is None:
            _warn_once(
                f"Extra-state plugin '{plugin.name}' returned None; ignoring update."
            )
            continue
        updated = next_state
    return updated
