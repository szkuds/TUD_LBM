"""Force operators — implementations of the ForceOperator protocol.

Public API: build_forces()

Implementation modules (_electric.py, _gravity.py, _gravity_masked.py,
_extra_state.py, _force_aggregator.py) are internal; use the factory to access.
What each force needs from its config section is declared at its registration
and checked by ``SimulationConfig``, never here.

Example:
    from operators.force import build_forces

    forces = build_forces(config, config.grid_shape, lattice)
    force = forces[0].compute(state)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.registry import get_operators

if TYPE_CHECKING:
    from src.config import SimulationConfig
    from src.lattice.lattice import Lattice
    from src.operators.protocols import ForceOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.operators.force")


def build_forces(
    config: SimulationConfig,
    grid_shape: tuple[int, ...],
    lattice: Lattice,
) -> tuple[ForceOperator, ...]:
    """Build every configured force, each bound to its payload.

    *config* is a validated :class:`~src.config.simulation_config.SimulationConfig`,
    whose ``active_forces`` holds only the configured ``*_force`` sections,
    each checked and defaulted against the schema its module registered — so
    this factory only looks up and binds.

    Args:
        config: A validated simulation configuration.
        grid_shape: Spatial dimensions, e.g. ``(64, 64, 1)`` or ``(64, 64, 32)`` for (nx, ny, nz).
        lattice: The simulation lattice.

    Returns:
        One bound :class:`~src.operators.protocols.ForceOperator` per configured
        force; empty when the run has none.
    """
    return tuple(
        get_operators("force")[name].target.build(params, grid_shape, config=config, lattice=lattice)
        for name, params in config.active_forces.items()
    )


__all__ = [
    "build_forces",
]
