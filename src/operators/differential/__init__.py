"""Differential operators — composite builder and primitives.

Public API: build_diff_ops(), build_differential_fn(),
build_wetting_gradient_fn(), build_wetting_laplacian_fn()

Implementation modules are internal; use the factories to access.

The ``differential`` registry kind holds two structurally different target
shapes. ``gradient`` / ``laplacian`` register the operator itself and are
resolved by name with :func:`build_differential_fn`. ``gradient_wetting`` /
``laplacian_wetting`` register a *builder* whose return value is the operator,
and each gets its own accessor — :func:`build_wetting_gradient_fn` and
:func:`build_wetting_laplacian_fn` — because their signatures differ: the
gradient builder takes the lattice velocities ``c`` and the Laplacian builder
does not.

Example:
    from operators.differential import build_diff_ops

    (
        gradient_standard,
        gradient_density,
        laplacian_density,
        gradient_density_wetting,
        laplacian_density_wetting,
    ) = build_diff_ops(config, mp_params, lattice)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import cast
from src.operators._loader import auto_load_operators
from src.operators.factory import build_operator

if TYPE_CHECKING:
    from typing import Any
    import jax.numpy as jnp
    from src.config.simulation_config import SimulationConfig
    from src.lattice.lattice import Lattice
    from src.operators.macroscopic import MultiphaseParams
    from src.operators.protocols import BoundDifferentialOperator
    from src.operators.protocols import DifferentialOperator
    from src.operators.protocols import WettingDifferentialOperator
    from src.operators.protocols import WettingGradientBuilder
    from src.operators.protocols import WettingLaplacianBuilder

# Auto-discover and import private operator modules for registry registration.
auto_load_operators("src.operators.differential")


def build_differential_fn(scheme: str) -> DifferentialOperator:
    """Return a differential operator satisfying DifferentialOperator protocol.

    For the plain operators only (``"gradient"``, ``"laplacian"``). The wetting
    schemes register a builder rather than an operator — resolve those with
    :func:`build_wetting_gradient_fn` / :func:`build_wetting_laplacian_fn`.

    Args:
        scheme: Differential operator name.

    Returns:
        A callable satisfying the DifferentialOperator protocol.

    Raises:
        ValueError: If scheme is not registered.
    """
    return cast("DifferentialOperator", build_operator("differential", scheme))


def build_wetting_gradient_fn() -> WettingGradientBuilder:
    """Return the builder for the parametric wetting gradient.

    Unlike :func:`build_differential_fn`, the registry target here is a factory:
    calling it with the static configuration returns the operator itself. The
    registry name is this function's identity, so there is no *scheme* argument.

    The cast sits at the registry boundary, which is where the target type is
    genuinely erased — see :func:`~src.operators.factory.build_operator`.

    Returns:
        The ``gradient_wetting`` builder, satisfying
        :class:`WettingGradientBuilder`.

    Raises:
        ValueError: If ``gradient_wetting`` is not registered.
    """
    return cast("WettingGradientBuilder", build_operator("differential", "gradient_wetting"))


def build_wetting_laplacian_fn() -> WettingLaplacianBuilder:
    """Return the builder for the parametric wetting Laplacian.

    The counterpart of :func:`build_wetting_gradient_fn`. Kept separate because
    the two builders take different arguments — the Laplacian stencil needs no
    lattice velocities — so no single type describes both.

    Returns:
        The ``laplacian_wetting`` builder, satisfying
        :class:`WettingLaplacianBuilder`.

    Raises:
        ValueError: If ``laplacian_wetting`` is not registered.
    """
    return cast("WettingLaplacianBuilder", build_operator("differential", "laplacian_wetting"))


def _wetting_scalar(
    wetting: dict[str, Any],
    name: str,
    legacy_name: str,
    *,
    default: float,
) -> jnp.ndarray:
    """Read one wetting scalar as a 0-d array, accepting the legacy short key.

    ``wetting_config`` is a free-form ``dict[str, Any]`` straight off the TOML,
    so a lookup is ``Any`` and a missing key is ``None`` — neither of which
    ``float()`` accepts. Resolving the two spellings here keeps that narrowing
    in one place instead of a nested ``.get`` chain per parameter.

    Args:
        wetting: The effective wetting configuration.
        name: Canonical key, e.g. ``"phi_left"``.
        legacy_name: Older short spelling, e.g. ``"phi_l"``.
        default: Value used when neither key carries a number.

    Returns:
        The value as a 0-d :mod:`jax.numpy` array.
    """
    import jax.numpy as jnp

    for key in (name, legacy_name):
        value = wetting.get(key)
        if value is not None:
            return jnp.array(float(value))
    return jnp.array(default)


def build_diff_ops(
    config: SimulationConfig,
    mp_params: MultiphaseParams | None,
    lattice: Lattice,
) -> tuple[
    BoundDifferentialOperator,
    BoundDifferentialOperator,
    BoundDifferentialOperator,
    WettingDifferentialOperator | None,
    WettingDifferentialOperator | None,
]:
    """Build gradient/laplacian closures, wetting-aware if applicable.

    Ensures boundary-condition modules are imported so that their
    ``@boundary_condition`` decorators have fired before we query
    pad-edge-mode metadata.

    Returns five items:

    * **gradient_standard**: ``(grid) → result``. Standard gradient ∇μ.
    * **gradient_density**: Density gradient ∇ρ, used in source terms.
    * **laplacian_density**: Laplacian ∇²ρ, used in chemical potential.
    * **gradient_density_wetting**: Parametric density gradient ``(grid, phi_l, phi_r, d_rho_l, d_rho_r) → result``.
      ``None`` unless hysteresis is configured.
    * **laplacian_density_wetting**: Parametric Laplacian ``(grid, phi_l, phi_r, d_rho_l, d_rho_r) → result``.
      ``None`` unless hysteresis is configured.

    Design:

    * **Non-wetting**: All returned closures accept only ``(grid)``.
      ``gradient_density_wetting`` and ``laplacian_density_wetting`` are ``None``.
    * **Fixed wetting** (wetting config but no hysteresis): Same behavior as non-wetting.
      ``gradient_density`` and ``laplacian_density`` close over static wetting parameters.
      ``gradient_density_wetting`` and ``laplacian_density_wetting`` are ``None``.
    * **Hysteresis**: ``gradient_density`` and ``laplacian_density`` are initial closures
      seeded with wetting parameters (explicit or neutral defaults).
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
    from src.operators.boundary import _bounce_back as _bb  # noqa: F401
    from src.operators.boundary import _periodic as _per  # noqa: F401
    from src.operators.boundary import _symmetry as _sym  # noqa: F401
    from src.operators.differential._pad_utils import determine_pad_modes

    # Standard gradient closure: (grid) → (nx, ny, 1, 2)
    _gradient_raw = build_differential_fn("gradient")

    def gradient_standard(grid: jnp.ndarray) -> jnp.ndarray:
        return _gradient_raw(grid, lattice.w, lattice.c, tuple(determine_pad_modes(config.bc_config)))

    wetting_config = config.wetting_config
    hysteresis_config = config.hysteresis_config
    effective_wetting = wetting_config
    if hysteresis_config is not None and effective_wetting is None:
        effective_wetting = {
            "phi_left": 1.0,
            "phi_right": 1.0,
            "d_rho_left": 0.0,
            "d_rho_right": 0.0,
        }

    if effective_wetting is not None and mp_params is not None:
        # Wetting: build parametric closures with rho_l, rho_v baked in.
        # Signature: (grid, phi_l, phi_r, d_rho_l, d_rho_r) → result.
        _gradient_wetting_factory = build_wetting_gradient_fn()
        _laplacian_wetting_factory = build_wetting_laplacian_fn()

        _grad_wetting = _gradient_wetting_factory(
            lattice.w,
            lattice.c,
            tuple(determine_pad_modes(config.bc_config)),
            config.bc_config,
            rho_l=mp_params.rho_l,
            rho_v=mp_params.rho_v,
        )
        _lap_wetting = _laplacian_wetting_factory(
            lattice.w,
            tuple(determine_pad_modes(config.bc_config)),
            config.bc_config,
            rho_l=mp_params.rho_l,
            rho_v=mp_params.rho_v,
        )

        # Extract wetting params once, used in both branches below.
        _phi_l = _wetting_scalar(effective_wetting, "phi_left", "phi_l", default=1.0)
        _phi_r = _wetting_scalar(effective_wetting, "phi_right", "phi_r", default=1.0)
        _d_rho_l = _wetting_scalar(effective_wetting, "d_rho_left", "d_rho_l", default=0.0)
        _d_rho_r = _wetting_scalar(effective_wetting, "d_rho_right", "d_rho_r", default=0.0)

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
    "build_wetting_gradient_fn",
    "build_wetting_laplacian_fn",
]
