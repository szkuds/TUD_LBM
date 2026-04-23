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
from tud_lbm.operators._loader import auto_load_operators
from tud_lbm.operators.factory import build_operator
from tud_lbm.operators.protocols import ForceOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("tud_lbm.operators.force")


class ForceParams(NamedTuple):
    """One pre-built force contribution.

    Attributes:
        name: Registry key, e.g. ``"gravity_force"``.
        compute_fn: Pure function ``(state, precomputed, lattice) → jnp.ndarray``
            of shape ``(nx, ny, 1, d)``.
            Returns the force contribution for this physics.
        precomputed: Optional pre-computed data (e.g. gravity template array).
    """

    name: str
    compute_fn: Any
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
    """Discover ``*_force`` fields on config, build ForceSetup with specs and source term.

    Each force operator in the registry must expose:

    - ``build(params, grid_shape, config, lattice)`` → precomputed data (or None)
    - ``compute(state, precomputed, **kwargs)`` → force array
    Args:
        config: A validated configuration object with ``*_force`` fields.
        grid_shape: Spatial dimensions, e.g. ``(64, 64)``.
        lattice: The simulation lattice.

    Returns:
        A :class:`ForceSetup` containing force specs and the source-term callable.
    """
    # Lazy import to avoid circular imports and ensure registry is populated
    from tud_lbm.operators.force._source_term import source as compute_source

    specs: list[ForceParams] = []
    for f in dataclasses.fields(config):
        if not f.name.endswith("_force"):
            continue
        params = getattr(config, f.name)
        if params is None:
            continue
        op = cast("ForceOperator", cast("object", _build_force_fn(f.name)))
        build_fn = op.build
        compute_fn = op.compute

        precomputed = build_fn(params, grid_shape, config=config, lattice=lattice)

        specs.append(
            ForceParams(
                name=f.name,
                compute_fn=compute_fn,
                precomputed=precomputed,
            )
        )

    return ForceSetup(specs=tuple(specs), source_term=compute_source)


def compute_total_force_ext(
    setup: Any,
    state: Any,
    force_setup: ForceSetup | None,
) -> tuple[jnp.ndarray | None, Any]:
    """Compute the summed external force contribution.

    Iterates over all force specs in the setup, calls each force's `compute_fn`,
    and accumulates contributions.

    Args:
        setup: The :class:`~setup.simulation_setup.SimulationSetup`.
        state: Current :class:`~state.state.State`.
        force_setup: The :class:`ForceSetup` containing force specs, or None if no forces.

    Returns:
        Tuple of ``(total_force, updated_state)`` where:
        - *total_force* is the summed force array, or None if no forces are active.
        - *updated_state* is the unchanged state (extra-state plugins handle updates).
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

    return total_force, state


__all__ = ["ForceParams", "ForceSetup", "build_forces", "compute_total_force_ext"]
