"""Equation-of-state operators for multiphase macroscopic computation.

Public API: build_eos_fn(), build_pressure_fn(), analytical_surface_tension(),
has_analytical_surface_tension(), coexistence_densities()

Implementation modules (_double_well.py, _carnahan_starling.py) are internal;
use the factories to access them. Each module registers its EOS under the
``"eos"`` kind and, if it has them, its bulk pressure under ``"pressure"``,
its closed-form surface tension under ``"surface_tension"`` and its Maxwell
coexistence solver under ``"coexistence"``, using the same name in each. To ask
whether an EOS has one of those, query the registry —
``eos in get_operator_names("pressure")``,
:func:`has_analytical_surface_tension`, :func:`coexistence_densities` returning
``None`` — rather than any list maintained here.

Example:
    from src.operators.macroscopic.eos import build_eos_fn, build_pressure_fn

    eos_fn = build_eos_fn("carnahan-starling", mp)
    mu_0 = eos_fn(rho)

    pressure_fn = build_pressure_fn(mp)
    p_0 = pressure_fn(rho)

    pair = coexistence_densities(section["eos"], section)  # (rho_v, rho_l), or None for double-well
"""

from __future__ import annotations
import functools
from typing import TYPE_CHECKING
from typing import Any
import jax.numpy as jnp
from src.config.multiphase import EOS_PARAMETERS
from src.config.multiphase import multiphase_params
from src.operators._loader import auto_load_operators
from src.registry import get_operator_names
from src.registry import get_operators

if TYPE_CHECKING:
    from collections.abc import Mapping
    from src.config.multiphase import MultiphaseParams
    from src.protocols import EosOperator

# Auto-discover EOS modules in this subpackage.
auto_load_operators("src.operators.macroscopic.eos")


def build_eos_fn(eos: str, mp: MultiphaseParams) -> EosOperator:
    """Build an EOS callable with bound parameters.

    Args:
        eos: EOS scheme name from config (e.g. ``"double-well"``).
        mp: :class:`~src.operators.macroscopic.MultiphaseParams` carrying
            all EOS-specific scalars.

    Returns:
        A bound :class:`~src.protocols.EosOperator`
        ``eos_fn(rho) -> mu_0``.
    """
    return get_operators("eos")[eos].target(mp)


def build_pressure_fn(mp: MultiphaseParams) -> EosOperator:
    """Build the bulk-pressure callable for the EOS bound in *mp*.

    This is the bulk thermodynamic pressure only: the interfacial ``-kappa``
    terms are not included, so ``p_0`` swings across a diffuse interface.
    Consumers that need the full normal pressure add
    ``-kappa * (rho * laplacian(rho) + |grad rho|^2 / 2)`` themselves.

    Args:
        mp: :class:`~src.operators.macroscopic.MultiphaseParams` carrying the
            EOS name and its scalars.

    Returns:
        A bound :class:`~src.protocols.EosOperator`
        ``pressure_fn(rho) -> p_0``. The multiphase macroscopic operator
        evaluates it every step and returns it as the pressure field.

    Raises:
        ValueError: If the EOS's own parameters are missing.
    """
    return get_operators("pressure")[mp.eos].target(mp)


def has_analytical_surface_tension(eos: str | None) -> bool:
    """Whether *eos* has a closed-form surface tension, so needs no calibration.

    The inverse is the question the analysis layer actually asks: an EOS that
    is absent from the ``"surface_tension"`` kind has no closed form, and its
    sigma must be measured by
    :mod:`src.simulation_io.analysis.surface_tension`. Membership of the kind
    is the whole check — do not reintroduce a list of EOS names.
    """
    return eos is not None and eos in get_operator_names("surface_tension")


def analytical_surface_tension(mp: MultiphaseParams) -> float | None:
    """Closed-form liquid-gas surface tension for the EOS bound in *mp*.

    Returns ``None`` when the EOS has no closed form (Young-Laplace
    calibration is the only way to a sigma for it) and when the EOS has one
    but its parameters cannot produce a value. Both cases mean the same thing
    to every caller — fall back to a measured value — so they are not
    distinguished.
    """
    if not has_analytical_surface_tension(mp.eos):
        return None
    sigma = get_operators("surface_tension")[mp.eos].target(mp)
    return None if sigma is None else float(sigma)


def coexistence_densities(eos: str, parameters: Mapping[str, Any]) -> tuple[float, float] | None:
    """Maxwell coexistence densities ``(rho_v, rho_l)`` of *eos*, or ``None``.

    ``None`` when *eos* registers no coexistence solver: it is then
    parameterised *by* its densities (the double-well's wells), so there is
    nothing to compute. *parameters* is any mapping holding the EOS's row of
    :data:`~src.config.multiphase.EOS_PARAMETERS` — the raw ``[multiphase]``
    section or ``mp._asdict()`` alike; other keys are ignored.

    Raises:
        ValueError: If the parameters admit no liquid-vapour coexistence
            (at or above the critical temperature).
    """
    if eos not in get_operator_names("coexistence"):
        return None
    solver = get_operators("coexistence")[eos].target
    return solver(**{name: parameters[name] for name in EOS_PARAMETERS[eos]})


@functools.lru_cache(maxsize=32)
def coexistence_potential(mp: MultiphaseParams) -> float:
    """The bulk chemical potential ``mu_0`` at coexistence, the same in both phases.

    Evaluated at the Maxwell vapour density when the EOS registers a coexistence
    solver, and at the configured ``rho_v`` otherwise (the double-well's wells
    *are* its coexistence densities). A configured Carnahan-Starling pair need not
    be the Maxwell one, which is why the solver's is used there.
    """
    pair = coexistence_densities(mp.eos, mp._asdict())
    rho_v = mp.rho_v if pair is None else pair[0]
    return float(build_eos_fn(mp.eos, mp)(jnp.asarray(rho_v)))


__all__ = [
    "analytical_surface_tension",
    "build_eos_fn",
    "build_pressure_fn",
    "coexistence_densities",
    "coexistence_potential",
    "has_analytical_surface_tension",
    "multiphase_params",
]
