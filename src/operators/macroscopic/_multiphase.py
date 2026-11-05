"""Multiphase macroscopic field computation - pure function.

Computes density, force-corrected velocity, the interparticle
(chemical-potential) force and the bulk pressure for diffuse-interface
multiphase models.

Bulk EOS handling is delegated to the macroscopic EOS subpackage,
selected by ``mp.eos``.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.macroscopic.eos import build_eos_fn
from src.operators.macroscopic.eos import build_pressure_fn
from src.registry import macroscopic_operator

if TYPE_CHECKING:
    from src.config.multiphase import MultiphaseParams
    from src.lattice.lattice import Lattice
    from src.protocols import DifferentialOperator


@macroscopic_operator(name="multiphase")
def compute_macroscopic_multiphase(
    f: jnp.ndarray,
    lattice: Lattice,
    force: jnp.ndarray | None = None,
    *,
    gradient_standard: DifferentialOperator,
    laplacian_density: DifferentialOperator,
    mp: MultiphaseParams,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute ``(rho, u, force, pressure)`` using the EOS selected by ``mp.eos``.

    *mp* is not part of :class:`~src.protocols.MacroscopicOperator`;
    :func:`~src.operators.macroscopic.build_multiphase_macroscopic_fn` binds it.
    The chemical potential and the bulk pressure come from the same EOS, so the
    force and the reported pressure share one free energy.

    Args:
        f: Populations, shape ``(nx, ny, nz, q, 1)``.
        lattice: :class:`~src.lattice.lattice.Lattice`.
        force: External force, shape ``(nx, ny, nz, 1, d)``, or ``None``.
        gradient_standard: Gradient for the chemical potential.
        laplacian_density: Laplacian of the density.
        mp: The run's multiphase parameters.
    """
    eos_fn = build_eos_fn(mp.eos, mp)
    pressure_fn = build_pressure_fn(mp)

    # Density - zeroth moment
    rho = jnp.sum(f, axis=-2, keepdims=True)

    # Momentum - first moment
    u = jnp.sum(f * lattice.c, axis=-2, keepdims=True) / rho

    # Total chemical potential: bulk EOS with the Laplacian correction
    mu = eos_fn(rho) - mp.kappa * laplacian_density(rho)

    # Interparticle force from chemical potential gradient
    force_int = -rho * gradient_standard(mu)

    # Total force
    force_total = force_int if force is None else force_int + force

    # Force-corrected velocity for equilibrium
    u_eq = u + force_total / (2.0 * rho)

    # Bulk pressure from the same EOS; the interfacial kappa terms are not included.
    pressure = pressure_fn(rho)

    return rho, u_eq, force_total, pressure
