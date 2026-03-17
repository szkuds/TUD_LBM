"""One-time factory that builds all differential operators.

Mirrors :func:`~operators.boundary.composite.build_composite_bc` exactly:
resolve configuration in pure Python, build jitted closures, return the
:class:`DifferentialOperators` NamedTuple.

This is the **only** place in the codebase that knows about the
wetting / no-wetting split for differential operators.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from operators.differential.config import DifferentialConfig
from operators.differential.operators import DifferentialOperators
from operators.differential.gradient import compute_gradient, compute_wetting_gradient
from operators.differential.laplacian import compute_laplacian


# ── Private closure builders ─────────────────────────────────────────


def _make_standard_gradient(cfg: DifferentialConfig):
    """Jitted closure over (w, c, pad_modes) — no wetting, ever."""
    w, c, pad_modes = cfg.w, cfg.c, cfg.pad_modes

    @jax.jit
    def grad(grid: jnp.ndarray) -> jnp.ndarray:
        return compute_gradient(grid, w, c, tuple(pad_modes))

    return grad


def _make_laplacian(cfg: DifferentialConfig):
    """Jitted closure over (w, pad_modes)."""
    w, pad_modes = cfg.w, cfg.pad_modes

    @jax.jit
    def lap(grid: jnp.ndarray) -> jnp.ndarray:
        return compute_laplacian(grid, w, tuple(pad_modes))

    return lap


# ── Public factory ───────────────────────────────────────────────────


def build_differential_operators(cfg: DifferentialConfig) -> DifferentialOperators:
    """Build all pre-compiled differential operators from *cfg*.

    Called **once** at setup time; the returned
    :class:`DifferentialOperators` is stored on
    :class:`~setup.simulation_setup.SimulationSetup` and used throughout
    the simulation loop.

    ``grad_standard`` is **always** the pad-modes-only standard gradient,
    regardless of wetting configuration.

    Args:
        cfg: A :class:`DifferentialConfig` with all required inputs.

    Returns:
        A :class:`DifferentialOperators` NamedTuple of jitted callables.
    """
    # grad_standard: pad_modes only — NEVER wetting-dependent
    grad_standard = _make_standard_gradient(cfg)

    # grad_field: wetting-corrected when wetting is on, standard otherwise
    if cfg.wetting_enabled:
        grad_field = compute_wetting_gradient(
            cfg.w,
            cfg.c,
            cfg.pad_modes,
            cfg.wetting_params,
            cfg.chemical_step,
        )
    else:
        grad_field = grad_standard  # same object, zero overhead

    # laplacian: uniform for now; add a wetting branch here if needed
    laplacian = _make_laplacian(cfg)

    return DifferentialOperators(
        grad_standard=grad_standard,
        grad_field=grad_field,
        laplacian=laplacian,
    )

