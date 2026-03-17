"""Multiphase (double-well) macroscopic field computation — pure function.

Extracted from
:class:`simulation_operators.macroscopic.MacroscopicMultiphaseDW`.

Computes density, force-corrected velocity, and the interparticle
(chemical-potential) force for the diffuse-interface model with a
double-well bulk free energy.

When a pre-built :class:`~operators.differential.operators.DifferentialOperators`
tuple is supplied via *diff_ops*, the proper LBM-stencil gradient and
Laplacian operators (with correct per-edge padding and optional wetting
ghost-cell correction) are used.  Without *diff_ops* the function falls
back to simple periodic central-difference / 5-point stencil helpers so
that existing callers and tests continue to work unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import jax.numpy as jnp

from setup.lattice import Lattice
from setup.simulation_setup import MultiphaseParams
from registry import macroscopic_operator

if TYPE_CHECKING:
    from operators.differential.operators import DifferentialOperators

# ── Internal fallback helpers (pure, jittable, periodic-only) ────────


def _gradient_2d(field_2d: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Central-difference gradient on a periodic 2D field.

    Args:
        field_2d: shape ``(nx, ny)``.

    Returns:
        ``(df_dx, df_dy)`` each shape ``(nx, ny)``.
    """
    df_dx = (jnp.roll(field_2d, -1, axis=0) - jnp.roll(field_2d, 1, axis=0)) / 2.0
    df_dy = (jnp.roll(field_2d, -1, axis=1) - jnp.roll(field_2d, 1, axis=1)) / 2.0
    return df_dx, df_dy


def _laplacian_2d(field_2d: jnp.ndarray) -> jnp.ndarray:
    """5-point Laplacian on a periodic 2D field.

    Args:
        field_2d: shape ``(nx, ny)``.

    Returns:
        Laplacian, shape ``(nx, ny)``.
    """
    return (
        jnp.roll(field_2d, 1, axis=0)
        + jnp.roll(field_2d, -1, axis=0)
        + jnp.roll(field_2d, 1, axis=1)
        + jnp.roll(field_2d, -1, axis=1)
        - 4.0 * field_2d
    )


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


def _chemical_potential(
    rho_2d: jnp.ndarray,
    kappa: float,
    beta: float,
    rho_l: float,
    rho_v: float,
) -> jnp.ndarray:
    """Full chemical potential: μ = μ_0(ρ) − κ ∇²ρ.

    Fallback path using the simple periodic ``_laplacian_2d`` helper.
    """
    mu_0 = _eos_double_well(rho_2d, beta, rho_l, rho_v)
    lap_rho = _laplacian_2d(rho_2d)
    return mu_0 - kappa * lap_rho


# ── Public API ───────────────────────────────────────────────────────


@macroscopic_operator(name="double-well")
def compute_macroscopic_multiphase(
    f: jnp.ndarray,
    lattice: Lattice,
    mp: MultiphaseParams,
    force_ext: jnp.ndarray | None = None,
    diff_ops: "DifferentialOperators | None" = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute density, equilibrium velocity, and total force for multiphase.

    Args:
        f: Populations, shape ``(nx, ny, q, 1)``.
        lattice: :class:`~setup.lattice.Lattice`.
        mp: :class:`~setup.simulation_setup.MultiphaseParams`.
        force_ext: Optional external force, shape ``(nx, ny, 1, 2)``.
        diff_ops: Optional pre-built
            :class:`~operators.differential.operators.DifferentialOperators`.
            When provided, the proper LBM-stencil gradient / Laplacian
            (with correct per-edge pad modes and optional wetting
            correction) are used.  When ``None``, the function falls
            back to the simple periodic helpers ``_gradient_2d`` /
            ``_laplacian_2d``.

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

    if diff_ops is not None:
        # ── LBM-stencil path (correct pad modes / wetting) ──────────
        # Laplacian and grad_standard are always pad-modes-only.
        mu_0 = _eos_double_well(rho[:, :, 0, 0], beta, mp.rho_l, mp.rho_v)
        lap_rho = diff_ops.laplacian(rho)                    # (nx, ny, 1, 1)
        mu = mu_0[..., None, None] - mp.kappa * lap_rho      # (nx, ny, 1, 1)

        # Chemical-potential gradient — always the standard (non-wetting) gradient
        grad_mu = diff_ops.grad_standard(mu)                 # (nx, ny, 1, 2)

        # F_int = −ρ ∇μ
        force_int = -rho * grad_mu                           # (nx, ny, 1, 2)
    else:
        # ── Fallback: simple periodic central-difference helpers ─────
        rho_2d = rho[:, :, 0, 0]
        mu = _chemical_potential(rho_2d, mp.kappa, beta, mp.rho_l, mp.rho_v)
        grad_mu_x, grad_mu_y = _gradient_2d(mu)

        # F_int = −ρ ∇μ
        force_int_x = -rho_2d * grad_mu_x
        force_int_y = -rho_2d * grad_mu_y

        # Pack to 4D
        force_int = jnp.stack(
            [force_int_x[:, :, None, None], force_int_y[:, :, None, None]],
            axis=-1,
        )[
            :, :, :, 0, :
        ]  # (nx, ny, 1, 2)  – squeeze the extra dim

    # 4. Total force
    force_total = force_int
    if force_ext is not None:
        force_total = force_total + force_ext

    # 5. Force-corrected velocity for equilibrium
    u_eq = u + force_total / (2.0 * rho)

    return rho, u_eq, force_total
