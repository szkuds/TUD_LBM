"""Multiphase macroscopic field computation - pure function.

Computes density, force-corrected velocity, and the interparticle
(chemical-potential) force for diffuse-interface multiphase models.

Bulk EOS handling is delegated to the macroscopic EOS subpackage,
selected by ``mp.eos``.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.macroscopic.eos import build_eos_fn
from src.registry import macroscopic_operator

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice
    from src.operators.macroscopic import MultiphaseParams
    from src.operators.protocols import DifferentialOperator
    from src.operators.protocols import EOSFunction


def _compute_macroscopic_multiphase_impl(
    f: jnp.ndarray,
    lattice: Lattice,
    mp: MultiphaseParams,
    eos_fn: EOSFunction,
    force_ext: jnp.ndarray | None = None,
    *,
    gradient_standard: DifferentialOperator,
    laplacian_density: DifferentialOperator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generic multiphase macroscopic computation shared between EOS models."""
    # Density - zeroth moment
    rho = jnp.sum(f, axis=-2, keepdims=True)

    # Momentum - first moment
    u = jnp.sum(f * lattice.c, axis=-2, keepdims=True) / rho

    # Bulk chemical potential from EOS
    mu_0 = eos_fn(rho)

    # Total chemical potential with Laplacian correction
    mu = mu_0 - mp.kappa * laplacian_density(rho)

    # Interparticle force from chemical potential gradient
    grad_mu = gradient_standard(mu)
    force_int = -rho * grad_mu

    # Total force
    force_total = force_int if force_ext is None else force_int + force_ext

    # Force-corrected velocity for equilibrium
    u_eq = u + force_total / (2.0 * rho)

    return rho, u_eq, force_total


@macroscopic_operator(name="multiphase")
def compute_macroscopic_multiphase(
    f: jnp.ndarray,
    lattice: Lattice,
    mp: MultiphaseParams,
    force_ext: jnp.ndarray | None = None,
    *,
    gradient_standard: DifferentialOperator,
    laplacian_density: DifferentialOperator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute multiphase macroscopic fields using EOS selected from ``mp.eos``."""
    eos_fn = build_eos_fn(mp.eos, mp)
    return _compute_macroscopic_multiphase_impl(
        f,
        lattice,
        mp,
        eos_fn,
        force_ext,
        gradient_standard=gradient_standard,
        laplacian_density=laplacian_density,
    )
