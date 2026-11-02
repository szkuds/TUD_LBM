"""Step-time sum of the external force hooks.

Registers nothing; it is the running counterpart of
:func:`~src.operators.force.build_forces`, whose bound
:class:`~src.protocols.ForceOperator` instances this sums.
"""

from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax.numpy as jnp
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state import State


def compute_total_force_ext(setup: SimulationSetup, state: State) -> jnp.ndarray | None:
    """Sum the run's forces (``setup.forces``) on *state*.

    The differential operators are handed to each force with the state's
    addon substates (``state.ext``) bound, so a force sees the same wall
    ghost cells as the macroscopic operator.

    Args:
        setup: The :class:`~src.pipeline.setup.SimulationSetup`.
        state: Current :class:`~src.pipeline.state.State`.

    Returns:
        The summed force array, or ``None`` when the run has no forces.
    """
    if not setup.forces:
        return None

    gradient_standard = partial(setup.gradient_standard, ext=state.ext)
    gradient_density = partial(setup.gradient_density, ext=state.ext)
    laplacian_density = partial(setup.laplacian_density, ext=state.ext)

    total_force: jnp.ndarray | None = None
    for force in setup.forces:
        hooks = force.compute(
            state,
            gradient_standard=gradient_standard,
            gradient_density=gradient_density,
            laplacian_density=laplacian_density,
        )
        total_force = hooks if total_force is None else total_force + hooks
    return total_force
