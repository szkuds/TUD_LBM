"""Equation-of-state operators for multiphase macroscopic computation.

Public API: build_eos_fn(), build_pressure_fn()
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import cast
import numpy as np
from src.operators._loader import auto_load_operators
from src.operators.factory import build_operator
from src.operators.macroscopic.eos._carnahan_starling import carnahan_starling_pressure
from src.operators.macroscopic.eos._double_well import double_well_pressure

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.operators.macroscopic import MultiphaseParams
    from src.operators.protocols import EOSFunction

# Auto-discover EOS modules in this subpackage.
auto_load_operators("src.operators.macroscopic.eos")

#: EOS with a bulk-pressure implementation, i.e. the ones ``build_pressure_fn``
#: can bind. Also the set supported by the Young-Laplace surface-tension
#: calibration and the pressure plot operators.
PRESSURE_EOS = frozenset({"double-well", "carnahan-starling"})


def build_eos_fn(eos: str, mp: MultiphaseParams) -> EOSFunction:
    """Build an EOS callable with bound parameters.

    Args:
        eos: EOS scheme name from config (e.g. ``"double-well"``).
        mp: :class:`~src.operators.macroscopic.MultiphaseParams` carrying
            all EOS-specific scalars.

    Returns:
        A bound :class:`~src.operators.protocols.EOSFunction`
        ``eos_fn(rho) -> mu_0``.

    Raises:
        ValueError: If *eos* is not registered in the EOS registry.
    """
    eos_builder = build_operator("eos", eos)
    return cast("EOSFunction", eos_builder(mp))


def build_pressure_fn(mp: MultiphaseParams) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``pressure(rho) -> p_0``, the bulk pressure for the EOS bound in *mp*.

    This is the bulk thermodynamic pressure only: the interfacial
    ``-kappa`` terms are not included, so ``p_0`` swings across a diffuse
    interface. Consumers that need the full normal pressure add
    ``-kappa * (rho * laplacian(rho) + |grad rho|^2 / 2)`` themselves.

    Args:
        mp: :class:`~src.operators.macroscopic.MultiphaseParams` carrying the
            EOS name and its scalars.

    Returns:
        A callable mapping a density array to a bulk-pressure array of the
        same shape. Accepts NumPy or JAX input; returns NumPy.

    Raises:
        ValueError: If ``mp.eos`` has no pressure implementation, or if the
            Carnahan-Starling parameters are missing.
    """
    if mp.eos == "carnahan-starling":
        if mp.a_eos is None or mp.b_eos is None or mp.r_eos is None or mp.t_eos is None:
            msg = "a_eos, b_eos, r_eos, t_eos are required for Carnahan-Starling pressure"
            raise ValueError(msg)
        a_eos, b_eos, r_eos, t_eos = mp.a_eos, mp.b_eos, mp.r_eos, mp.t_eos
        return lambda rho: np.asarray(carnahan_starling_pressure(rho, a_eos, b_eos, r_eos, t_eos))

    if mp.eos == "double-well":
        beta = 8.0 * mp.kappa / (float(mp.interface_width) ** 2 * (mp.rho_l - mp.rho_v) ** 2)
        return lambda rho: np.asarray(double_well_pressure(rho, beta, mp.rho_l, mp.rho_v))

    supported = ", ".join(sorted(PRESSURE_EOS))
    msg = f"bulk pressure is implemented for EOS {supported}; got '{mp.eos}'"
    raise ValueError(msg)


__all__ = [
    "PRESSURE_EOS",
    "build_eos_fn",
    "build_pressure_fn",
    "carnahan_starling_pressure",
    "double_well_pressure",
]
