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
from typing import NamedTuple


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


#: A registered ``dimensionless`` operator: pure, total, and ``None`` rather
#: than raising when this run cannot supply the number.
DimensionlessNumberOperator = Callable[[DimensionlessInputs], float | None]
