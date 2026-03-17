"""Multiphase (double-well) macroscopic field computation — pure function.

Extracted from
:class:`simulation_operators.macroscopic.MacroscopicMultiphaseDW`.

Computes density, force-corrected velocity, and the interparticle
(chemical-potential) force for the diffuse-interface model with a
double-well bulk free energy.

Uses pre-built :class:`~operators.differential.operators.DifferentialOperators`
for the LBM-stencil gradient and Laplacian operators (with correct per-edge
padding and optional wetting ghost-cell correction).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import jax.numpy as jnp

from setup.lattice import Lattice
from setup.simulation_setup import MultiphaseParams
from registry import macroscopic_operator

if TYPE_CHECKING:
    from operators.differential.operators import DifferentialOperators


# ── EOS and chemical potential ───────────────────────────────────────


def _eos_double_well(
    rho_2d: jnp.ndarray,
    beta: float,
    rho_l: float,
    rho_v: float,
) -> jnp.ndarray:
    """Double-well equation-of-state derivative (chemical potential bulk part).

    Args:
        rho_2d: Density field, shape ``(nx, ny)``.
        beta: ``8 κ / (W² (ρ_l − ρ_v)²)``.
        rho_l: Liquid density.
        rho_v: Vapour density.

    Returns:
        ``μ_0(ρ)``, shape ``(nx, ny)``.
    """
    return (
        2.0
        * beta
        * (rho_2d - rho_l)
        * (rho_2d - rho_v)
        * (2.0 * rho_2d - rho_l - rho_v)
    )


# ── Public API ───────────────────────────────────────────────────────


@macroscopic_operator(name="double-well")
def compute_macroscopic_multiphase(
    f: jnp.ndarray,
    lattice: Lattice,
    mp: MultiphaseParams,
    force_ext: jnp.ndarray | None = None,
    *,
    diff_ops: "DifferentialOperators",
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute density, equilibrium velocity, and total force for multiphase.

    Args:
        f: Populations, shape ``(nx, ny, q, 1)``.
        lattice: :class:`~setup.lattice.Lattice`.
        mp: :class:`~setup.simulation_setup.MultiphaseParams`.
        force_ext: Optional external force, shape ``(nx, ny, 1, 2)``.
        diff_ops: Pre-built
            :class:`~operators.differential.operators.DifferentialOperators`.
            Provides the LBM-stencil gradient / Laplacian with correct
            per-edge pad modes and optional wetting correction.

    Returns:
        ``(rho, u_eq, force_total)``

        * ``rho``: shape ``(nx, ny, 1, 1)``
        * ``u_eq``: force-corrected velocity, shape ``(nx, ny, 1, 2)``
        * ``force_total``: total force, shape ``(nx, ny, 1, 2)``
    """
    cx = lattice.c[0]  # (q,)
    cy = lattice.c[1]  # (q,)
    q = lattice.q

    # 1. Density
    rho = jnp.sum(f, axis=2, keepdims=True)  # (nx, ny, 1, 1)

    # 2. Uncorrected velocity
    cx4 = cx.reshape((1, 1, q, 1))
    cy4 = cy.reshape((1, 1, q, 1))
    ux = jnp.sum(f * cx4, axis=2, keepdims=True)
    uy = jnp.sum(f * cy4, axis=2, keepdims=True)
    u = jnp.concatenate([ux, uy], axis=-1) / rho  # (nx, ny, 1, 2)

    # 3. Interparticle force from chemical potential
    beta = (
        8.0 * mp.kappa / (float(mp.interface_width) ** 2 * (mp.rho_l - mp.rho_v) ** 2)
    )

    # Laplacian and grad_standard are always pad-modes-only.
    mu_0 = _eos_double_well(rho[:, :, 0, 0], beta, mp.rho_l, mp.rho_v)
    lap_rho = diff_ops.laplacian(rho)                    # (nx, ny, 1, 1)
    mu = mu_0[..., None, None] - mp.kappa * lap_rho      # (nx, ny, 1, 1)

    # Chemical-potential gradient — always the standard (non-wetting) gradient
    grad_mu = diff_ops.grad_standard(mu)                 # (nx, ny, 1, 2)

    # F_int = −ρ ∇μ
    force_int = -rho * grad_mu                           # (nx, ny, 1, 2)

    # 4. Total force
    force_total = force_int
    if force_ext is not None:
        force_total = force_total + force_ext

    # 5. Force-corrected velocity for equilibrium
    u_eq = u + force_total / (2.0 * rho)

    return rho, u_eq, force_total
