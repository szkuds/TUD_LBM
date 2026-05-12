"""Multiphase (double-well) macroscopic field computation — pure function.

Extracted from
:class:`simulation_operators.macroscopic.MacroscopicMultiphaseDW`.

Computes density, force-corrected velocity, and the interparticle
(chemical-potential) force for the diffuse-interface model with a
double-well bulk free energy.

Uses LBM-stencil gradient and Laplacian operators (with correct per-edge
padding and optional wetting ghost-cell correction).
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from tud_lbm.registry import macroscopic_operator

if TYPE_CHECKING:
    from tud_lbm.lattice.lattice import Lattice
    from tud_lbm.operators.macroscopic import MultiphaseParams
    from tud_lbm.operators.protocols import DifferentialOperator

# ── EOS and chemical potential ───────────────────────────────────────


def _eos_double_well(
    rho: jnp.ndarray,
    beta: float,
    rho_l: float,
    rho_v: float,
) -> jnp.ndarray:
    """Double-well equation-of-state derivative (chemical potential bulk part).

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        beta: ``8 κ / (W² (ρ_l − ρ_v)²)``.
        rho_l: Liquid density.
        rho_v: Vapour density.

    Returns:
        ``μ_0(ρ)``, shape ``(nx, ny, nz, 1, 1)``.
    """
    return 2.0 * beta * (rho - rho_l) * (rho - rho_v) * (2.0 * rho - rho_l - rho_v)


# ── Public API ───────────────────────────────────────────────────────


@macroscopic_operator(name="double-well")
def compute_macroscopic_multiphase(
    f: jnp.ndarray,
    lattice: Lattice,
    mp: MultiphaseParams,
    force_ext: jnp.ndarray | None = None,
    *,
    gradient_standard: DifferentialOperator,
    laplacian_density: DifferentialOperator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute density, equilibrium velocity, and total force for multiphase.

    Args:
        f: Populations, shape ``(nx, ny, nz, q, 1)``.
        lattice: :class:`~setup.lattice.Lattice`.
        mp: :class:`~setup.simulation_setup.MultiphaseParams`.
        force_ext: Optional external force, shape ``(nx, ny, nz, 1, 2)``.
        gradient_standard: Standard LBM-stencil gradient for chemical potential ``∇μ``.
            Signature ``(grid) → gradient``. Must be a **single-argument** grid-only
            closure. **Never wetting-corrected** — used only for ``∇μ``.
        laplacian_density: LBM-stencil Laplacian for density ``∇²ρ``.
            Signature ``(grid) → laplacian``. Must be a **single-argument** grid-only
            closure. For wetting simulations, this is wetting-corrected via
            :func:`step_multiphase` shim injection.

    Returns:
        ``(rho, u_eq, force_total)``

        * ``rho``: shape ``(nx, ny, nz, 1, 1)``
        * ``u_eq``: force-corrected velocity, shape ``(nx, ny, nz, 1, 2)``
        * ``force_total``: total force, shape ``(nx, ny, nz, 1, 2)``
    """
    # Density — zeroth moment. Sum over q
    rho = jnp.sum(f, axis=-2, keepdims=True)  # (nx, ny, nz, 1, 1)

    # Momentum — first moment
    u = jnp.sum(f * lattice.c, axis=-2, keepdims=True) / rho  # (nx, ny, nz, 1, D)

    # 3. Interparticle force from chemical potential
    beta = 8.0 * mp.kappa / (float(mp.interface_width) ** 2 * (mp.rho_l - mp.rho_v) ** 2)

    # Laplacian and gradient are always pad-modes-only.
    mu_0 = _eos_double_well(rho, beta, mp.rho_l, mp.rho_v)  # (nx, ny, nz, 1, 1)
    lap_rho = laplacian_density(rho)  # (nx, ny, nz, 1, 1)
    mu = mu_0 - mp.kappa * lap_rho  # (nx, ny, nz, 1, 1)

    # Chemical-potential gradient - always the standard (non-wetting) gradient
    grad_mu = gradient_standard(mu)  # (nx, ny, nz, 1, 2)

    # F_int = -ρ ∇μ
    force_int = -rho * grad_mu  # (nx, ny, nz, 1, 2)

    # 4. Total force
    force_total = force_int
    if force_ext is not None:
        force_total = force_total + force_ext

    # 5. Force-corrected velocity for equilibrium
    u_eq = u + force_total / (2.0 * rho)

    return rho, u_eq, force_total
