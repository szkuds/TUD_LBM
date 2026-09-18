"""The resolved physical inputs every dimensionless number is built from.

Deliberately free of project imports: the number modules under ``numbers/``
import this, and :mod:`physical_parameters` imports both, so keeping this leaf
free of dependencies is what makes the package acyclic.

Resolution itself lives in
:func:`src.simulation_io.analysis.physical_parameters.physical_parameters.resolve_dimensionless_inputs`,
next to the resolvers it composes.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import NamedTuple

if TYPE_CHECKING:
    import numpy as np


class DimensionlessInputs(NamedTuple):
    """Everything the registered dimensionless numbers share, resolved once.

    ``gamma`` and ``drho`` carry their provenance because the two are resolved
    by deliberately different rules: ``gamma`` prefers a measured Young-Laplace
    value and falls back to the closed form, while ``drho`` is measured off the
    run's own init field wherever one exists. The ``drho`` that feeds the closed
    form is *not* this one -- see ``_resolve_buoyancy_delta_rho``.

    ``g`` and ``angle_deg`` are optional: a run may configure no gravity at all,
    which leaves the buoyancy-driven numbers unresolvable while ``Oh`` and
    ``La`` remain perfectly well defined. Each operator decides for itself.

    ``domain`` and ``pressure`` serve numbers that weigh a run against the
    *domain* rather than against the inclusion: ``domain`` is the grid's
    ``(nx, ny)``, and ``pressure`` the EOS bulk pressure ``p_0(rho)`` (NumPy in,
    NumPy out), which is ``None`` for an EOS with no operator registered under
    the ``pressure`` kind. Both default to ``None`` — "not available", like
    ``g`` and ``angle_deg`` — rather than to a neutral-looking value: ``Hs`` is
    a threshold diagnostic where ``>= 1`` means the run dies, so a caller that
    omits ``domain`` must get no answer, not the most reassuring one.
    """

    gamma: float
    gamma_source: str
    drho: float
    drho_source: str
    length: float
    length_label: str
    nu: float
    rho_l: float
    g: float | None
    angle_deg: float | None
    domain: tuple[float, float] | None = None
    pressure: Callable[[np.ndarray], np.ndarray] | None = None


#: A registered ``dimensionless`` operator: pure, total, and ``None`` rather
#: than raising when this run cannot supply the number.
DimensionlessNumberOperator = Callable[[DimensionlessInputs], float | None]
