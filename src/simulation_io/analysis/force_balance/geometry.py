"""Circular-cap geometry of a 2-D droplet, and the geometric closure model M6.

A 2-D droplet of area ``Omega`` that is a circular cap of contact angle
``theta`` has base width ``A`` with::

    A**2 / Omega = 4 sin(theta)**2 / (theta - sin(theta) cos(theta))

which falls monotonically from infinity at ``theta -> 0`` to zero at ``pi``.
With the mean of the two contact angles in place of ``theta``, the measured
droplets follow it to a few degrees (checked in the report), so the base
width is set by the angles.

The closure (model M6) needs nothing from the run but its windows, ``Omega``
and ``X``: each line sits at its window bound, the base width follows from the
mean angle, and while the trailing line is Gibbs-held at the step (it may not
recede onto the hydrophilic side before its angle reaches ``theta_R^phil``)
the base grows with the leading line, ``A = 2 (X - X_step)``, and the trailing
angle is whatever the cap of that base requires.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
from scipy.optimize import brentq

if TYPE_CHECKING:
    from src.simulation_io.analysis.force_balance.runs import Windows

_THETA_MIN = 1e-4
_THETA_MAX = math.pi - 1e-9

#: Stages of :class:`StepClosure`: before, straddling, after, trailing line held.
_BEFORE, _STRADDLING, _AFTER, _HELD = 1, 2, 3, 4


def base_squared_over_area(theta: float) -> float:
    """``A**2/Omega`` of a circular cap of contact angle *theta* (radians)."""
    return 4.0 * math.sin(theta) ** 2 / (theta - math.sin(theta) * math.cos(theta))


def cap_angle(a2_over_omega: float) -> float:
    """Inverse of :func:`base_squared_over_area`: the cap angle (radians)."""
    lo_val = base_squared_over_area(_THETA_MIN)
    if a2_over_omega >= lo_val:
        return _THETA_MIN
    if a2_over_omega <= 0.0:
        return math.pi
    return brentq(lambda t: base_squared_over_area(t) - a2_over_omega, _THETA_MIN, _THETA_MAX, xtol=1e-12)


def cap_base(theta: float) -> float:
    """Base width in units of ``R0 = sqrt(Omega)`` of a cap of angle *theta* (radians)."""
    return math.sqrt(base_squared_over_area(theta))


class StepClosure:
    """Base width and hysteresis of model M6 as functions of the centre position ``X``.

    ``X`` is measured from the step in units of R0, and the centre of mass is
    taken at the middle of the base.
    """

    def __init__(self, windows: Windows) -> None:
        """Cap base widths and stage boundaries for *windows*."""
        rad = math.radians
        self.w = windows
        self.a1 = cap_base(0.5 * (rad(windows.receding_phob) + rad(windows.advancing_phob)))
        self.a2 = cap_base(0.5 * (rad(windows.receding_phob) + rad(windows.advancing_phil)))
        self.a3 = cap_base(0.5 * (rad(windows.receding_phil) + rad(windows.advancing_phil)))
        #: Centre positions where the lead reaches the step, where the trailing
        #: line reaches it (hold begins) and where the hold ends.
        self.x_lead_step = -0.5 * self.a1
        self.x_hold = 0.5 * self.a2
        self.x_release = 0.5 * max(self.a3, self.a2)

    @property
    def breakpoints(self) -> tuple[float, float, float]:
        """Positions where the coefficients change form."""
        return self.x_lead_step, self.x_hold, self.x_release

    def stage(self, x: float) -> int:
        """1 before the step, 2 straddling, 4 trailing line held, 3 after."""
        if x < self.x_lead_step:
            return _BEFORE
        if x < self.x_hold:
            return _STRADDLING
        if x < self.x_release:
            return _HELD
        return _AFTER

    def base(self, x: float) -> float:
        """Base width ``A/R0``."""
        stage = self.stage(x)
        if stage == _BEFORE:
            return self.a1
        if stage == _STRADDLING:
            return self.a2
        if stage == _HELD:
            return 2.0 * x
        return self.a3

    def trailing_angle(self, x: float) -> float:
        """Trailing contact angle in degrees."""
        stage = self.stage(x)
        if stage in (_BEFORE, _STRADDLING):
            return self.w.receding_phob
        if stage == _AFTER:
            return self.w.receding_phil
        mean = math.degrees(cap_angle(self.base(x) ** 2))
        return min(self.w.receding_phob, max(self.w.receding_phil, 2.0 * mean - self.w.advancing_phil))

    def leading_angle(self, x: float) -> float:
        """Leading contact angle in degrees."""
        return self.w.advancing_phob if self.stage(x) == _BEFORE else self.w.advancing_phil

    def hysteresis(self, x: float) -> float:
        """``H = cos(theta_trail) - cos(theta_lead)``."""
        return math.cos(math.radians(self.trailing_angle(x))) - math.cos(math.radians(self.leading_angle(x)))

    def vectorised(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(A/R0, H)`` at every element of *x*."""
        return np.array([self.base(v) for v in x]), np.array([self.hysteresis(v) for v in x])
