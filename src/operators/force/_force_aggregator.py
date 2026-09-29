"""Step-time sum of the external force contributions.

Registers nothing; it is the running counterpart of
:func:`~src.operators.force.build_forces`, whose bound
:class:`~src.operators.protocols.ForceOperator` instances this sums.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import jax.numpy as jnp
    from src.operators.protocols import ForceOperator
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state import State


def compute_total_force_ext(
    setup: SimulationSetup,
    state: State,
    forces: tuple[ForceOperator, ...],
) -> tuple[jnp.ndarray | None, Any]:
    """Compute the summed external force contribution.

    Args:
        setup: The :class:`~src.pipeline.setup.SimulationSetup` (supplies the
            differential operators a force may need).
        state: Current :class:`~src.pipeline.state.State`.
        forces: The run's bound forces, ``setup.forces``.

    Returns:
        Tuple of ``(total_force, updated_state)`` where:
        - *total_force* is the summed force array, or ``state.force_ext`` when
          the run has no forces.
        - *updated_state* is the unchanged state (extra-state plugins handle updates).
    """
    if not forces:
        return state.force_ext, state

    # Recompute per-step external force from active contributions only.
    # Do not seed from state.force_ext, otherwise values accumulate over time.
    total_force: jnp.ndarray | None = None
    for force in forces:
        contribution = force.compute(
            state,
            gradient_standard=setup.gradient_standard,
            gradient_density=setup.gradient_density,
            laplacian_density=setup.laplacian_density,
        )
        total_force = contribution if total_force is None else total_force + contribution

    return total_force, state
