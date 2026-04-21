"""Differential operators — composite builder and primitives.

Public API: build_diff_ops()

Implementation modules are internal; use the factory to access.

Example:
    from operators.differential import build_diff_ops

    gradient_standard, gradient, laplacian = build_diff_ops(config, mp_params, lattice)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from operators._loader import auto_load_operators
from operators.factory import build_operator

if TYPE_CHECKING:
    from collections.abc import Callable
    import jax.numpy as jnp
    from config.simulation_config import SimulationConfig
    from operators.macroscopic import MultiphaseParams
    from operators.protocols import DifferentialOperator
    from setup.lattice import Lattice

# Auto-discover and import private operator modules for registry registration.
auto_load_operators("operators.differential")


def build_differential_fn(scheme: str) -> DifferentialOperator:
    """Return a differential operator satisfying DifferentialOperator protocol.

    Args:
        scheme: Differential operator name.

    Returns:
        A callable satisfying the DifferentialOperator protocol.

    Raises:
        ValueError: If scheme is not registered.
    """
    return build_operator("differential", scheme)


def build_diff_ops(
    config: SimulationConfig,
    mp_params: MultiphaseParams | None,
    lattice: Lattice,
) -> tuple[Callable, Callable, Callable]:
    """Build gradient/laplacian closures, wetting-aware if applicable.

    Ensures boundary-condition modules are imported so that their
    ``@boundary_condition`` decorators have fired before we query
    pad-edge-mode metadata.

    The returned callables are **closures** that depend on how wetting is configured:

    * **Non-wetting**: ``(gradient_standard, gradient_density, laplacian_density)``
      each accept only ``(grid)`` and close over lattice/boundary data.
    * **Wetting**: ``(gradient_standard, gradient_density, laplacian_density)`` are
      all ``(grid) → result`` closures. For density operators, static parameters
      (rho_l, rho_v, width) are baked in at build time, so the returned closures
      accept only ``(grid, phi_l, phi_r, d_rho_l, d_rho_r)`` at runtime.
      Live wetting parameters are injected at step time by :func:`step_multiphase`.

    Args:
        config: Validated simulation configuration.
        mp_params: Multiphase parameters (``None`` for single-phase).
        lattice: The simulation :class:`Lattice` (weights, velocities).

    Returns:
        ``(gradient_standard, gradient_density, laplacian_density)`` — three
        callable differential-operator closures.
    """
    from operators.boundary import _bounce_back as _bb  # noqa: F401
    from operators.boundary import _periodic as _per  # noqa: F401
    from operators.boundary import _symmetry as _sym  # noqa: F401
    from operators.differential._pad_utils import determine_pad_modes

    # Standard gradient closure: (grid) → (nx, ny, 1, 2)
    _gradient_raw = build_differential_fn("gradient")

    def gradient_standard(grid: jnp.ndarray) -> jnp.ndarray:
        return _gradient_raw(grid, lattice.w, lattice.c, tuple(determine_pad_modes(config.bc_config)))

    wetting_config = config.wetting_config
    if wetting_config is not None and mp_params is not None:
        # Wetting: build factories with rho_l, rho_v, width baked in.
        # The returned closures have signature (grid, phi_l, phi_r, d_rho_l, d_rho_r) → result.
        _gradient_wetting_factory = build_differential_fn("gradient_wetting")
        _laplacian_wetting_factory = build_differential_fn("laplacian_wetting")
        gradient_density = _gradient_wetting_factory(
            lattice.w,
            lattice.c,
            tuple(determine_pad_modes(config.bc_config)),
            config.bc_config,
            rho_l=mp_params.rho_l,
            rho_v=mp_params.rho_v,
            width=config.interface_width,
        )
        laplacian_density = _laplacian_wetting_factory(
            lattice.w,
            tuple(determine_pad_modes(config.bc_config)),
            config.bc_config,
            rho_l=mp_params.rho_l,
            rho_v=mp_params.rho_v,
            width=config.interface_width,
        )
    else:
        # Non-wetting: wrap the raw functions into single-argument grid→result closures
        _laplacian_raw = build_differential_fn("laplacian")

        def gradient_density(grid: jnp.ndarray) -> jnp.ndarray:
            return _gradient_raw(grid, lattice.w, lattice.c, tuple(determine_pad_modes(config.bc_config)))

        def laplacian_density(grid: jnp.ndarray) -> jnp.ndarray:
            return _laplacian_raw(grid, lattice.w, tuple(determine_pad_modes(config.bc_config)))

    return gradient_standard, gradient_density, laplacian_density


__all__ = [
    "build_diff_ops",
    "build_differential_fn",
]
