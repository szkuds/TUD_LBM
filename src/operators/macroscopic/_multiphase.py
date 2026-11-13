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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Compute ``(rho, u, force, pressure, lattice_pressure)`` using the EOS selected by ``mp.eos``.

    *mp* is not part of :class:`~src.protocols.MacroscopicOperator`;
    :func:`~src.operators.macroscopic.build_multiphase_macroscopic_fn` binds it.
    The chemical potential and the bulk pressure come from the same EOS, so the
    force and the reported pressure share one free energy.

    With ``mp.equilibrium_reference_density`` (``rho_0``, set for an equilibrium
    that carries a pressure) the pressure force is split, after Zhang et al.,
    Int. J. Multiphase Flow 198, 105668 (2026): the equilibrium takes
    ``lattice_pressure = rho_0 * (mu - mu_b)`` and the interparticle force becomes
    ``-(rho - rho_0) * grad(mu)``. The momentum equation still sees
    ``-rho * grad(mu)`` in total. ``mu_b`` is the chemical potential at
    coexistence (``mp.equilibrium_reference_potential``, from
    :func:`~src.config.multiphase.multiphase_params`), so the lattice pressure
    vanishes at a flat equilibrium interface; any constant there is dynamically
    neutral. Without ``rho_0`` nothing is supplied and the
    force is the whole ``-rho * grad(mu)``.

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
    rho_0 = mp.equilibrium_reference_density
    if rho_0 is None:
        lattice_pressure = None
        force_int = -rho * gradient_standard(mu)
    else:
        mu_b = mp.equilibrium_reference_potential
        assert mu_b is not None  # noqa: S101 - multiphase_params sets it with rho_0
        lattice_pressure = rho_0 * (mu - mu_b)
        force_int = -(rho - rho_0) * gradient_standard(mu)

    # Total force
    force_total = force_int if force is None else force_int + force

    # Force-corrected velocity for equilibrium
    u_eq = u + force_total / (2.0 * rho)

    # Bulk pressure from the same EOS; the interfacial kappa terms are not included.
    pressure = pressure_fn(rho)

    return rho, u_eq, force_total, pressure, lattice_pressure
