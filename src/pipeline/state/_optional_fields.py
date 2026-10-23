"""Private helper: optional force fields for the simulation State."""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp

if TYPE_CHECKING:
    from src.pipeline.setup import SimulationSetup


def _build_optional_fields(
    setup: SimulationSetup,
    nx: int,
    ny: int,
    nz: int,
    d: int,
) -> tuple[jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray]:
    """Return ``(force, force_ext, pressure)`` pre-populated with zeros or ``None``.

    For multiphase simulations, the ``force`` field is initialised to zeros.
    For runs with active forces, the ``force_ext`` field is initialised to zeros.
    ``pressure`` is always initialised to zeros: every macroscopic operator
    returns one and every step writes it. In all cases, ``lax.scan`` requires the carry pytree structure to be
    constant across iterations; fields that will later be written must start
    as zeros (not ``None``).

    Args:
        setup: :class:`~setup.simulation_setup.SimulationSetup`.
        nx: Grid size in x.
        ny: Grid size in y.
        nz: Grid size in z.
        d: Lattice dimension (e.g. 2 for D2Q9).

    Returns:
        A tuple ``(force, force_ext, pressure)``: the forces are each either a
        zero-filled array of shape ``(nx, ny, nz, 1, d)`` or ``None``; the
        pressure is zeros of shape ``(nx, ny, nz, 1, 1)``.
    """
    force = jnp.zeros((nx, ny, nz, 1, d)) if setup.multiphase_params is not None or setup.config.force_enabled else None
    force_ext = jnp.zeros((nx, ny, nz, 1, d)) if setup.config.force_enabled else None
    pressure = jnp.zeros((nx, ny, nz, 1, 1))
    return force, force_ext, pressure
