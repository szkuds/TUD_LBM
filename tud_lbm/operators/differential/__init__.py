"""Differential operators — composite builder and primitives.

Public API: build_diff_ops()

Implementation modules are internal; use the factory to access.

Example:
    from operators.differential import build_diff_ops

    gradient_standard, gradient, laplacian = build_diff_ops(config, mp_params, lattice)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from tud_lbm.operators._loader import auto_load_operators
from tud_lbm.operators.factory import build_operator

if TYPE_CHECKING:
    from collections.abc import Callable
    import jax.numpy as jnp
    from tud_lbm.config.simulation_config import SimulationConfig
    from tud_lbm.lattice.lattice import Lattice
    from tud_lbm.operators.macroscopic import MultiphaseParams
    from tud_lbm.operators.protocols import DifferentialOperator

# Auto-discover and import private operator modules for registry registration.
auto_load_operators("tud_lbm.operators.differential")


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
) -> tuple[Callable, Callable, Callable, Callable | None, Callable | None]:
    """Build gradient/laplacian closures, wetting-aware if applicable.

    Ensures boundary-condition modules are imported so that their
    ``@boundary_condition`` decorators have fired before we query
    pad-edge-mode metadata.

    Returns five items:

    * **gradient_standard**: ``(grid) → result``. Standard gradient ∇μ.
    * **gradient_density**: Density gradient ∇ρ, used in source terms.
    * **laplacian_density**: Laplacian ∇²ρ, used in chemical potential.
    * **gradient_density_wetting**: Parametric density gradient ``(grid, phi_l, phi_r, d_rho_l, d_rho_r) → result``.
      ``None`` unless wetting+hysteresis are configured.
    * **laplacian_density_wetting**: Parametric Laplacian ``(grid, phi_l, phi_r, d_rho_l, d_rho_r) → result``.
      ``None`` unless wetting+hysteresis are configured.

    Design:

    * **Non-wetting**: All returned closures accept only ``(grid)``.
      ``gradient_density_wetting`` and ``laplacian_density_wetting`` are ``None``.
    * **Fixed wetting** (wetting config but no hysteresis): Same behavior as non-wetting.
      ``gradient_density`` and ``laplacian_density`` close over static wetting parameters.
      ``gradient_density_wetting`` and ``laplacian_density_wetting`` are ``None``.
    * **Hysteresis** (wetting + hysteresis configs): ``gradient_density`` and ``laplacian_density``
      are initial closures (gradient_standard variant with seed parameters from config).
      ``gradient_density_wetting`` and ``laplacian_density_wetting`` are the parametric factories
      used by the hysteresis optimizer to build trial-step operators.

    Args:
        config: Validated simulation configuration.
        mp_params: Multiphase parameters (``None`` for single-phase).
        lattice: The simulation :class:`Lattice` (weights, velocities).

    Returns:
        ``(gradient_standard,
        gradient_density, laplacian_density,
        gradient_density_wetting, laplacian_density_wetting)``
    """
    import jax.numpy as jnp
    from tud_lbm.operators.boundary import _bounce_back as _bb  # noqa: F401
    from tud_lbm.operators.boundary import _periodic as _per  # noqa: F401
    from tud_lbm.operators.boundary import _symmetry as _sym  # noqa: F401
    from tud_lbm.operators.differential._pad_utils import determine_pad_modes

    # Standard gradient closure: (grid) → (nx, ny, 1, 2)
    _gradient_raw = build_differential_fn("gradient")

    def gradient_standard(grid: jnp.ndarray) -> jnp.ndarray:
        return _gradient_raw(grid, lattice.w, lattice.c, tuple(determine_pad_modes(config.bc_config)))

    wetting_config = config.wetting_config
    hysteresis_config = config.hysteresis_config

    if wetting_config is not None and mp_params is not None:
        # Wetting: build parametric closures with rho_l, rho_v, width baked in.
        # Signature: (grid, phi_l, phi_r, d_rho_l, d_rho_r) → result.
        _gradient_wetting_factory = build_differential_fn("gradient_wetting")
        _laplacian_wetting_factory = build_differential_fn("laplacian_wetting")

        _grad_wetting = _gradient_wetting_factory(
            lattice.w,
            lattice.c,
            tuple(determine_pad_modes(config.bc_config)),
            config.bc_config,
            rho_l=mp_params.rho_l,
            rho_v=mp_params.rho_v,
            width=config.interface_width,
        )
        _lap_wetting = _laplacian_wetting_factory(
            lattice.w,
            tuple(determine_pad_modes(config.bc_config)),
            config.bc_config,
            rho_l=mp_params.rho_l,
            rho_v=mp_params.rho_v,
            width=config.interface_width,
        )

        # Extract wetting params once, used in both branches below.
        _phi_l = jnp.array(float(wetting_config.get("phi_left", wetting_config.get("phi_l", 1.0))))
        _phi_r = jnp.array(float(wetting_config.get("phi_right", wetting_config.get("phi_r", 1.0))))
        _d_rho_l = jnp.array(float(wetting_config.get("d_rho_left", wetting_config.get("d_rho_l", 0.0))))
        _d_rho_r = jnp.array(float(wetting_config.get("d_rho_right", wetting_config.get("d_rho_r", 0.0))))

        def gradient_density(grid: jnp.ndarray) -> jnp.ndarray:
            return _grad_wetting(grid, _phi_l, _phi_r, _d_rho_l, _d_rho_r)

        def laplacian_density(grid: jnp.ndarray) -> jnp.ndarray:
            return _lap_wetting(grid, _phi_l, _phi_r, _d_rho_l, _d_rho_r)

        if hysteresis_config is not None:
            # Hysteresis: expose parametric slots for the optimizer.
            # gradient_density / laplacian_density serve as seed-param closures for t=0.
            gradient_density_wetting = _grad_wetting
            laplacian_density_wetting = _lap_wetting
        else:
            # Fixed wetting: parameters never change; no parametric slots needed.
            gradient_density_wetting = None
            laplacian_density_wetting = None
    else:
        # Non-wetting: plain single-argument closures.
        _laplacian_raw = build_differential_fn("laplacian")

        def gradient_density(grid: jnp.ndarray) -> jnp.ndarray:
            return _gradient_raw(grid, lattice.w, lattice.c, tuple(determine_pad_modes(config.bc_config)))

        def laplacian_density(grid: jnp.ndarray) -> jnp.ndarray:
            return _laplacian_raw(grid, lattice.w, tuple(determine_pad_modes(config.bc_config)))

        gradient_density_wetting = None
        laplacian_density_wetting = None

    return gradient_standard, gradient_density, laplacian_density, gradient_density_wetting, laplacian_density_wetting


__all__ = [
    "build_diff_ops",
    "build_differential_fn",
]
