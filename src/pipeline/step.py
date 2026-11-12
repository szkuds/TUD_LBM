"""The one time step: :func:`kernel` and :func:`step`, and :func:`observe` for the derived fields.

``kernel`` is the physics pass, a pure function of the populations, the
external force and the addon substates: macroscopic fields, equilibrium,
collision (+ source when a force acts), streaming, obstacle and boundary
conditions. It passes ``ext`` to the differential operators and knows no
addon by name.

``step`` is the skeleton around it: the forces, the kernel, the new
:class:`~src.pipeline.state.state.State`, then each addon's ``post_step`` hook
in addon-name order (:mod:`src.addons.hooks`). The hysteresis
optimiser re-enters ``kernel`` from its hook for its trial steps.

The carry is only ``(f, t, ext)``: the physics reads nothing else. Every other
field — density, velocity, forces, pressure — is a function of that state, and
:func:`observe` is its one definition, evaluated where a field is wanted
(snapshots, the stability diagnostics), never stored between steps.
"""

from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
import jax.numpy as jnp
from src.operators.force._force_aggregator import compute_total_force_ext

if TYPE_CHECKING:
    from collections.abc import Mapping
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State


class Observables(NamedTuple):
    """The fields derived from a state, all at its own time level.

    Attributes:
        rho: Density, ``(nx, ny, nz, 1, 1)``.
        u: Velocity, force-corrected when a force acts, ``(nx, ny, nz, 1, d)``;
            zero inside an obstacle (for plotting).
        force: Total force (interaction + external), ``(nx, ny, nz, 1, d)``, or
            ``None`` when none acts.
        force_ext: External force, ``(nx, ny, nz, 1, d)``, or ``None`` without forces.
        pressure: Bulk pressure, ``(nx, ny, nz, 1, 1)``.
    """

    rho: jnp.ndarray
    u: jnp.ndarray
    force: jnp.ndarray | None
    force_ext: jnp.ndarray | None
    pressure: jnp.ndarray


def _moments(
    setup: SimulationSetup,
    f: jnp.ndarray,
    force_ext: jnp.ndarray | None,
    ext: Mapping[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray, jnp.ndarray | None]:
    """``(rho, u, force, pressure, lattice_pressure)`` of *f* under *force_ext* and the addon substates *ext*."""
    return setup.macroscopic_fn(
        f,
        setup.lattice,
        force_ext,
        gradient_standard=partial(setup.gradient_standard, ext=ext),
        laplacian_density=partial(setup.laplacian_density, ext=ext),
    )


def kernel(
    setup: SimulationSetup,
    f: jnp.ndarray,
    force_ext: jnp.ndarray | None,
    ext: Mapping[str, Any],
) -> jnp.ndarray:
    """Advance the populations *f* by one physics pass.

    Args:
        setup: Closed-over :class:`~src.pipeline.setup.SimulationSetup`.
        f: Populations, shape ``(nx, ny, nz, q, 1)``.
        force_ext: External force, shape ``(nx, ny, nz, 1, d)``, or ``None``.
        ext: Addon substates (``State.ext``, or a candidate variant of it),
            bound into the differential operators.

    Returns:
        The populations after collision, streaming, obstacle and boundary conditions.
    """
    lattice = setup.lattice
    rho, u, force_tot, _pressure, lattice_pressure = _moments(setup, f, force_ext, ext)

    feq = setup.equilibrium_fn(rho, u, lattice, lattice_pressure)
    if force_tot is None:
        f_col = setup.collision_fn(f, feq, setup.tau)
    else:
        source = setup.source_fn(rho, u, force_tot, lattice, gradient=partial(setup.gradient_density, ext=ext))
        f_col = setup.collision_fn(f, feq, setup.tau, source)

    f_stream = setup.streaming_fn(f_col, lattice)
    return setup.bc_fn(f_stream, f_col)


def observe(setup: SimulationSetup, state: State) -> Observables:
    """The fields derived from *state*: the moments, forces and pressure the kernel would use on it.

    Args:
        setup: Closed-over :class:`~src.pipeline.setup.SimulationSetup`.
        state: Any :class:`~src.pipeline.state.state.State`.

    Returns:
        :class:`Observables`, all at ``state.t``.
    """
    force_ext = compute_total_force_ext(setup, state)
    rho, u, force, pressure, _lattice_pressure = _moments(setup, state.f, force_ext, state.ext)
    if setup.obstacle_mask is not None:
        u = jnp.where(setup.obstacle_mask, 0.0, u)
    return Observables(rho=rho, u=u, force=force, force_ext=force_ext, pressure=pressure)


def step(setup: SimulationSetup, state: State) -> State:
    """One LBM time step: forces, kernel, then the addons' ``post_step`` hooks.

    Args:
        setup: Closed-over :class:`~src.pipeline.setup.SimulationSetup`.
        state: Current :class:`~src.pipeline.state.state.State`.

    Returns:
        The state after one time step.
    """
    f_next = kernel(setup, state.f, compute_total_force_ext(setup, state), state.ext)
    new_state = state._replace(f=f_next, t=state.t + 1)
    for _name, hooks in setup.addons:
        if hooks.post_step is not None:
            new_state = hooks.post_step(setup, state, new_state)
    return new_state
