"""Force operators — composite builder and ForceParams.

Public API: build_forces(), ForceParams, ForceSetup

Implementation modules are internal; use the factory to access.

Example:
    from operators.force import build_forces

    force_setup = build_forces(config, (64, 64))
    specs = force_setup.specs
    source_fn = force_setup.source_term
"""

from __future__ import annotations
import dataclasses
from collections.abc import Callable
from typing import Any
from typing import NamedTuple
from typing import cast
import jax.numpy as jnp
from operators._loader import auto_load_operators
from operators.factory import build_operator
from operators.protocols import ForceOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("operators.force")


class ForceParams(NamedTuple):
    """One pre-built force contribution.

    Attributes:
        name: Registry key, e.g. "gravity_force".
        compute_fn: Pure function (state, precomputed, lattice) → jnp.ndarray of shape (nx, ny, 1, d).
                    Returns the force contribution for this physics.
        init_fn: (setup) → dict of extra State fields needed at t=0.
                 Stateful forces may return values such as ``{"h": ...}``;
                 stateless forces use a no-op default that returns ``{}``.
        precomputed: Optional pre-computed data (e.g. gravity template array).
    """

    name: str
    compute_fn: Any
    init_fn: Any
    update_state_fn: Any
    precomputed: Any | None = None


class ForceSetup(NamedTuple):
    """Container for force definitions and the resolved source-term callable.

    Bundles force specifications with the source-term function to provide
    a unified interface for the rest of the codebase.

    Attributes:
        specs: Tuple of :class:`ForceParams` for each active force.
        source_term: Callable that computes the well-balanced forcing source term,
                    signature ``(rho, u, force, lattice, *, gradient) → jnp.ndarray``.
    """

    specs: tuple[ForceParams, ...]
    source_term: Callable[[Any, Any, Any, Any], Any]


def _state_force_init(
    grid_shape: tuple[int, ...],
    lattice: Any,
    precomputed: Any,
) -> dict[str, jnp.ndarray]:
    """Default initialiser for stateless forces."""
    return {}


def _state_force_update(
    state: Any,
    precomputed: Any,
    lattice: Any,
    stream_fn: Any,
) -> Any:
    """Default state update for stateless forces."""
    return state


def _build_force_fn(scheme: str) -> Callable[..., object] | type:
    """Return a registered force module (internal helper).

    Args:
        scheme: Force model name ("gravity_force", "electric_force", etc).

    Returns:
        A registry-backed force module exposing ``build`` and ``compute``.

    Raises:
        ValueError: If scheme is not registered.
    """
    # Lazy imports trigger module registration via decorators.
    from operators.force import _electric as _elec_impl  # noqa: F401
    from operators.force import _gravity as _grav_impl  # noqa: F401

    return build_operator("force", scheme)


def build_forces(
    config: Any,
    grid_shape: tuple[int, ...],
    lattice: Any,
) -> ForceSetup:
    """Discover *_force fields on config, build ForceSetup with specs and source term.

    Each force operator in the registry must expose:
      - build(params, grid_shape, config, lattice) → precomputed data (or None)
      - compute(state, precomputed, **kwargs) → force array
      - init_state(grid_shape, lattice, precomputed) → dict of extra State fields
      - update_state(state, precomputed, lattice, stream_fn) → state

    Stateless force modules may omit the state hooks; they are replaced
    here with no-op defaults.

    Args:
        config: A validated configuration object with ``*_force`` fields.
        grid_shape: Spatial dimensions, e.g. ``(64, 64)``.
        lattice: The simulation lattice.

    Returns:
        A :class:`ForceSetup` containing force specs and the source-term callable.
    """
    # Lazy import to avoid circular imports and ensure registry is populated
    from operators.force._source_term import source as compute_source

    specs: list[ForceParams] = []
    seen: set[str] = set()
    for f in dataclasses.fields(config):
        if not f.name.endswith("_force"):
            continue
        params = getattr(config, f.name)
        if params is None:
            continue
        seen.add(f.name)

        op = cast("ForceOperator", cast("object", _build_force_fn(f.name)))
        build_fn = op.build
        compute_fn = op.compute
        init_fn = getattr(op, "init_state", _state_force_init)
        update_state_fn = getattr(op, "update_state", _state_force_update)

        precomputed = build_fn(params, grid_shape, config=config, lattice=lattice)

        specs.append(
            ForceParams(
                name=f.name,
                compute_fn=compute_fn,
                init_fn=init_fn,
                precomputed=precomputed,
                update_state_fn=update_state_fn,
            )
        )

    return ForceSetup(specs=tuple(specs), source_term=compute_source)


def compute_total_force_ext(
    setup: Any,
    state: Any,
    force_setup: ForceSetup | None,
    streaming_fn: Any,
) -> tuple[jnp.ndarray | None, Any]:
    """Compute the summed external force contribution and update stateful hooks.

    Iterates over all force specs in the setup, calls each force's `compute_fn`,
    accumulates contributions, and applies state updates when present.

    Args:
        setup: The :class:`~setup.simulation_setup.SimulationSetup`.
        state: Current :class:`~state.state.State`.
        force_setup: The :class:`ForceSetup` containing force specs, or None if no forces.
        streaming_fn: The streaming operator (passed to state update hooks).

    Returns:
        Tuple of ``(total_force, updated_state)`` where:
        - *total_force* is the summed force array, or None if no forces are active.
        - *updated_state* has stateful force fields updated (e.g., wetting height).
    """
    total_force = state.force_ext

    if force_setup is None or not force_setup.specs:
        return total_force, state

    for spec in force_setup.specs:
        contribution = spec.compute_fn(
            state,
            spec.precomputed,
            gradient_standard=setup.gradient_standard,
            gradient_density=setup.gradient_density,
            laplacian_density=setup.laplacian_density,
        )
        total_force = contribution if total_force is None else total_force + contribution
        state = spec.update_state_fn(state, spec.precomputed, setup.lattice, streaming_fn)

    return total_force, state


__all__ = ["ForceParams", "ForceSetup", "build_forces", "compute_total_force_ext"]
