"""Config-derived scalar quantities used to normalise droplet metrics.

This module owns the single closed-form surface-tension formula. Policy about
*which* surface tension to prefer (measured vs analytical) lives with the
consumer; see
:func:`src.simulation_io.analysis.physical_parameters.physical_parameters._resolve_surface_tension`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import NamedTuple

if TYPE_CHECKING:
    from src.config import SimulationConfig

_WIDTH_EPS = 1e-15

#: Radius used when neither a setup contact line nor configured radii resolve.
_DEFAULT_R_ZERO = 27.0


class RZero(NamedTuple):
    """Resolved R₀ plus whether the nominal-radius fallback was used."""

    value: float
    used_fallback: bool


def analytical_sigma_lg(config: SimulationConfig) -> float | None:
    """Closed-form liquid-gas surface tension ``gamma = (2/3)(kappa/W)*drho**2``.

    Returns ``None`` when any required parameter is missing or the interface
    width is effectively zero. This is the *formula* only — it makes no
    judgement about whether a measured value should be preferred.
    """
    kappa = config.kappa
    width = config.interface_width
    rho_l = config.rho_l
    rho_v = config.rho_v
    if kappa is None or width is None or rho_l is None or rho_v is None:
        return None
    if abs(float(width)) < _WIDTH_EPS:
        return None
    drho = float(rho_l) - float(rho_v)
    return (2.0 / 3.0) * (float(kappa) / float(width)) * drho**2


def measured_sigma_lg(config: SimulationConfig) -> float | None:
    """Calibrated surface tension measured at run time, when one is stored.

    Written into ``config.extra["surface_tension"]`` by
    :mod:`src.simulation_io.analysis.surface_tension`. Absent for runs that were never
    calibrated.
    """
    measured = config.extra.get("surface_tension")
    return None if measured is None else float(measured)


def resolve_r_zero(config: SimulationConfig) -> RZero:
    """Initial droplet radius in lattice units.

    Derived from half of the setup contact-line length when the init file is
    readable. Otherwise falls back to ``initialisation.radii[0] * min(nx, ny)``
    (or 27.0 when no radii are given) and flags ``used_fallback=True``.
    """
    from src.simulation_io.analysis.physical_parameters.physical_parameters import _get_setup_contact_line_length

    length = _get_setup_contact_line_length(config)
    if length is not None:
        return RZero(length / 2.0, used_fallback=False)

    init = config.initialisation
    radii = init.get("radii", []) if isinstance(init, dict) else []
    if radii:
        nominal = float(radii[0]) * float(min(config.grid_shape[0], config.grid_shape[1]))
        return RZero(nominal, used_fallback=True)
    return RZero(_DEFAULT_R_ZERO, used_fallback=True)


def resolve_step_x(config: SimulationConfig) -> float | None:
    """Chemical-step x-position in lattice units, or ``None`` when unset."""
    chem = config.chemical_step_config
    if not chem or not isinstance(chem, dict):
        return None
    loc = chem.get("chemical_step_location")
    if loc is None:
        return None
    return float(loc) * float(config.grid_shape[0])


def inclination_angle_deg(config: SimulationConfig) -> float:
    """Gravity inclination angle in degrees, or ``0.0`` when not configured."""
    gravity = config.gravity_force
    if gravity and isinstance(gravity, dict):
        return float(gravity.get("inclination_angle_deg", 0.0))
    return 0.0


@dataclass(frozen=True)
class MetricScales:
    """Scalar quantities that normalise a run's droplet metrics."""

    rho_mean: float
    sigma_measured: float | None
    sigma_analytical: float | None
    nu: float
    r_zero: float
    r_zero_is_fallback: bool
    offset_x: float
    incl_deg: float
    save_interval: int

    @property
    def sigma_primary(self) -> float | None:
        """The surface tension to normalise with: measured when calibrated.

        Note that ``sigma_analytical`` is the closed-form double-well value. It
        is not valid for ``carnahan-starling``, where only a measured value is
        physically meaningful — hence the preference order.
        """
        return self.sigma_measured if self.sigma_measured is not None else self.sigma_analytical

    @property
    def sigma_source(self) -> str:
        """``"measured"`` or ``"analytical"`` — which value ``sigma_primary`` returned."""
        return "measured" if self.sigma_measured is not None else "analytical"


def resolve_scales(config: SimulationConfig) -> MetricScales | None:
    """Resolve every scaling quantity for *config*.

    Returns ``None`` when the config lacks the multiphase parameters required
    to define a liquid-gas interface — neither a measured nor a closed-form
    surface tension is obtainable. That is a capability failure rather than an
    error: callers skip the run.
    """
    if config.rho_l is None or config.rho_v is None:
        return None
    sigma_measured = measured_sigma_lg(config)
    sigma_analytical = analytical_sigma_lg(config)
    if sigma_measured is None and sigma_analytical is None:
        return None

    r_zero = resolve_r_zero(config)
    step_x = resolve_step_x(config)
    return MetricScales(
        rho_mean=(float(config.rho_l) + float(config.rho_v)) / 2.0,
        sigma_measured=sigma_measured,
        sigma_analytical=sigma_analytical,
        nu=(float(config.tau) - 0.5) / 3.0,
        r_zero=r_zero.value,
        r_zero_is_fallback=r_zero.used_fallback,
        offset_x=step_x if step_x is not None else float(config.grid_shape[0] // 2),
        incl_deg=inclination_angle_deg(config),
        save_interval=max(config.save_interval, 1),
    )
