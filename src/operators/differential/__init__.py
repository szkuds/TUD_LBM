"""Differential operators — composite builder and typed accessors.

Public API: build_diff_ops(), build_gradient_fn(), build_laplacian_fn(),
build_wetting_gradient_fn(), build_wetting_laplacian_fn()

Every accessor returns a :class:`~src.operators.protocols.DifferentialOperator`
``op(grid, wetting=None)`` with the lattice, pad modes and any static wetting
configuration already bound. The closures live in the implementation modules
(_gradient.py, _laplacian.py, _gradient_wetting.py, _laplacian_wetting.py);
there is no ``differential`` registry kind, because nothing is selected between
by configuration. The pad modes and wetting defaults come from the
configuration (``SimulationConfig.pad_modes`` / ``.wetting_defaults``).

Example:
    from src.operators.differential import build_diff_ops

    gradient_standard, gradient_density, laplacian_density = build_diff_ops(config, mp_params, lattice)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators.differential._gradient import build_gradient
from src.operators.differential._gradient_wetting import build_wetting_gradient
from src.operators.differential._laplacian import build_laplacian
from src.operators.differential._laplacian_wetting import build_wetting_laplacian
from src.operators.wetting._params import WettingParams

if TYPE_CHECKING:
    from typing import Any
    from src.config.chemical_step import ChemicalStepWall
    from src.config.multiphase_params import MultiphaseParams
    from src.config.simulation_config import SimulationConfig
    from src.lattice.lattice import Lattice
    from src.operators.protocols import DifferentialOperator


def build_gradient_fn(lattice: Lattice, pad_modes: tuple[str, ...]) -> DifferentialOperator:
    """Return the plain LBM-stencil gradient, padded with *pad_modes*.

    Args:
        lattice: The simulation :class:`Lattice` (weights, velocities).
        pad_modes: ``SimulationConfig.pad_modes``, ``(top, bottom, right, left)``.

    Returns:
        ``grad(grid) → (nx, ny, nz, 1, 2)``. Passing *wetting* raises :class:`TypeError`.
    """
    return build_gradient(lattice.w, lattice.c, pad_modes)


def build_laplacian_fn(lattice: Lattice, pad_modes: tuple[str, ...]) -> DifferentialOperator:
    """Return the plain LBM-stencil Laplacian, padded with *pad_modes*.

    Args:
        lattice: The simulation :class:`Lattice` (weights).
        pad_modes: ``SimulationConfig.pad_modes``, ``(top, bottom, right, left)``.

    Returns:
        ``lap(grid) → (nx, ny, nz, 1, 1)``. Passing *wetting* raises :class:`TypeError`.
    """
    return build_laplacian(lattice.w, pad_modes)


def build_wetting_gradient_fn(
    lattice: Lattice,
    pad_modes: tuple[str, ...],
    bc_config: dict[str, Any],
    *,
    rho_l: float,
    rho_v: float,
    default: WettingParams,
    step: ChemicalStepWall | None = None,
) -> DifferentialOperator:
    """Return the wetting-corrected gradient, padded with *pad_modes*.

    Args:
        lattice: The simulation :class:`Lattice` (weights, velocities).
        pad_modes: ``SimulationConfig.pad_modes``, ``(top, bottom, right, left)``.
        bc_config: Boundary-condition edge map naming the wetting wall.
        rho_l: Liquid density, baked into the closure.
        rho_v: Vapour density, baked into the closure.
        default: Wetting parameters used when the operator is called without *wetting*.
        step: ``SimulationConfig.chemical_step_wall``, or ``None``.

    Returns:
        ``grad(grid, wetting=None) → (nx, ny, nz, 1, 2)``.
    """
    return build_wetting_gradient(
        lattice.w, lattice.c, pad_modes, bc_config, rho_l=rho_l, rho_v=rho_v, default=default, step=step
    )


def build_wetting_laplacian_fn(
    lattice: Lattice,
    pad_modes: tuple[str, ...],
    bc_config: dict[str, Any],
    *,
    rho_l: float,
    rho_v: float,
    default: WettingParams,
    step: ChemicalStepWall | None = None,
) -> DifferentialOperator:
    """Return the wetting-corrected Laplacian, padded with *pad_modes*.

    Args:
        lattice: The simulation :class:`Lattice` (weights).
        pad_modes: ``SimulationConfig.pad_modes``, ``(top, bottom, right, left)``.
        bc_config: Boundary-condition edge map naming the wetting wall.
        rho_l: Liquid density, baked into the closure.
        rho_v: Vapour density, baked into the closure.
        default: Wetting parameters used when the operator is called without *wetting*.
        step: ``SimulationConfig.chemical_step_wall``, or ``None``.

    Returns:
        ``lap(grid, wetting=None) → (nx, ny, nz, 1, 1)``.
    """
    return build_wetting_laplacian(
        lattice.w, pad_modes, bc_config, rho_l=rho_l, rho_v=rho_v, default=default, step=step
    )


def build_diff_ops(
    config: SimulationConfig,
    mp_params: MultiphaseParams | None,
    lattice: Lattice,
) -> tuple[DifferentialOperator, DifferentialOperator, DifferentialOperator]:
    """Build the gradient/Laplacian operators, wetting-aware if applicable.

    Returns three :class:`~src.operators.protocols.DifferentialOperator` instances:

    * **gradient_standard**: Standard gradient ∇μ. Never wetting-aware.
    * **gradient_density**: Density gradient ∇ρ, used in source terms.
    * **laplacian_density**: Laplacian ∇²ρ, used in chemical potential.

    Without a wetting configuration (or single-phase) the density operators are
    the plain stencils. With one — every hysteresis run has at least a neutral
    one — they are wetting-aware, seeded with ``config.wetting_defaults``; the
    hysteresis optimiser passes live parameters as their *wetting* argument.

    Args:
        config: Validated simulation configuration.
        mp_params: Multiphase parameters (``None`` for single-phase).
        lattice: The simulation :class:`Lattice` (weights, velocities).

    Returns:
        ``(gradient_standard, gradient_density, laplacian_density)``
    """
    pad_modes = config.pad_modes
    gradient_standard = build_gradient_fn(lattice, pad_modes)

    wetting_defaults = config.wetting_defaults
    if wetting_defaults is None or mp_params is None:
        return gradient_standard, gradient_standard, build_laplacian_fn(lattice, pad_modes)

    assert config.bc_config is not None  # noqa: S101 - guaranteed by SimulationConfig._apply_defaults
    default = WettingParams.from_defaults(wetting_defaults)
    rho_l, rho_v = mp_params.rho_l, mp_params.rho_v
    step = config.chemical_step_wall
    gradient_density = build_wetting_gradient_fn(
        lattice, pad_modes, config.bc_config, rho_l=rho_l, rho_v=rho_v, default=default, step=step
    )
    laplacian_density = build_wetting_laplacian_fn(
        lattice, pad_modes, config.bc_config, rho_l=rho_l, rho_v=rho_v, default=default, step=step
    )
    return gradient_standard, gradient_density, laplacian_density


__all__ = [
    "build_diff_ops",
    "build_gradient_fn",
    "build_laplacian_fn",
    "build_wetting_gradient_fn",
    "build_wetting_laplacian_fn",
]
