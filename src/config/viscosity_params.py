"""Viscous-stress parameters — the JIT-static view of a decoupled-viscosity configuration.

Built only by :attr:`SimulationConfig.viscosity_params
<src.config.simulation_config.SimulationConfig.viscosity_params>`, after
``_validate_viscosity`` has checked the stability bound, so no consumer
re-checks the configuration.

Zhang, Guo & Wang (Phys. Fluids 34, 012110, 2022), Eqs. 7 and 29: the shear
moments relax at ``1/lambda_v`` and the equilibrium carries ``cs2*A*S``, giving
the kinematic viscosity ``nu = cs2*(lambda_v - 1/2 - A)``. ``nu`` is interpolated
linearly in ``rho`` between the phases, so ``A(rho)`` is linear too.
"""

from __future__ import annotations
from typing import NamedTuple

_CS2 = 1.0 / 3.0


class ViscosityParams(NamedTuple):
    """Kinematic viscosities of the two phases and the shear relaxation time.

    All fields are Python scalars (compile-time constants).
    """

    lambda_v: float
    nu_l: float
    nu_v: float
    rho_l: float
    rho_v: float


def build_viscosity_params(
    *,
    relaxation_time: float,
    tau_liquid: float,
    tau_gas: float,
    rho_l: float,
    rho_v: float,
) -> ViscosityParams:
    """Convert the configured relaxation times into phase viscosities ``cs2*(tau - 1/2)``."""
    return ViscosityParams(
        lambda_v=relaxation_time,
        nu_l=_CS2 * (tau_liquid - 0.5),
        nu_v=_CS2 * (tau_gas - 0.5),
        rho_l=rho_l,
        rho_v=rho_v,
    )
