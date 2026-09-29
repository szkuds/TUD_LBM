"""Multiphase parameters — the JIT-static view of a multiphase configuration.

Built only by :attr:`SimulationConfig.multiphase_params
<src.config.simulation_config.SimulationConfig.multiphase_params>`, after
``_validate_multiphase`` has guaranteed every required field, so no consumer
re-checks the configuration.
"""

from __future__ import annotations
from typing import NamedTuple


class MultiphaseParams(NamedTuple):
    """Equation-of-state and surface-tension parameters.

    All fields are Python scalars (compile-time constants).
    """

    eos: str
    kappa: float
    rho_l: float
    rho_v: float
    interface_width: int
    g: float | None = None
    a_eos: float | None = None
    b_eos: float | None = None
    r_eos: float | None = None
    t_eos: float | None = None
