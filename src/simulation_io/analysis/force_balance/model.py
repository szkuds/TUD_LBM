"""The reduced droplet model of the SI, in its dimensionless form.

Per unit depth, for a 2-D droplet of area ``Omega_D = R0**2``::

    (1/Oh**2) dCa/dT = Bo - H(X) - beta(X) * Ca,      dX/dT = Ca

with ``X = x/R0``, ``Ca = eta*v/gamma``, ``T = t*gamma/(eta*R0)``. ``H`` is the
hysteresis force ``cos(theta_R) - cos(theta_A)`` and ``beta`` the friction
coefficient. Hysteresis is a static (Coulomb) resistance: a droplet at rest
stays at rest while ``Bo <= H``, and the model never runs backwards, since
every run here moves down-slope only.

Two solvers, tested against each other:

* :func:`piecewise_trajectory` — the closed form for piecewise-constant
  ``H`` and ``beta`` (one exponential relaxation per regime, the regimes
  joined at the boundary crossing times).
* :func:`integrate_trajectory` — fixed-step RK4 for arbitrary ``H(X)`` and
  ``beta(X)``. Coefficients are frozen to the segment the step started in
  and each step that would cross a breakpoint is shortened to land on it,
  so a discontinuous ``H`` costs no accuracy.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

#: Bisection depth for landing a step on a breakpoint or on ``Ca = 0``.
_BISECT_ITERATIONS = 60

#: A friction coefficient below this is treated as frictionless.
_BETA_EPS = 1e-12


def steady_capillary(bo: float, h: float, beta: float) -> float:
    """Terminal ``Ca_inf = (Bo - H)/beta`` of a constant-coefficient regime."""
    return (bo - h) / beta


def relaxation_time(oh: float, beta: float) -> float:
    """Viscous relaxation time ``T_v = 1/(Oh**2 * beta)``."""
    return 1.0 / (oh**2 * beta)


def coasting_length(bo: float, h: float, beta: float, oh: float) -> float:
    """Coasting length ``Lambda = Ca_inf * T_v`` in units of R0."""
    return steady_capillary(bo, h, beta) * relaxation_time(oh, beta)


@dataclass(frozen=True)
class Regime:
    """Constant coefficients of one regime of the step model."""

    h: float
    beta: float


def regime_state(tau: np.ndarray, ca0: float, oh: float, bo: float, regime: Regime) -> tuple[np.ndarray, np.ndarray]:
    """``(Ca, X - X0)`` a time *tau* after entering *regime* with ``Ca = ca0``.

    The exponential relaxation ``Ca = Ca_inf + (Ca0 - Ca_inf) exp(-tau/T_v)``,
    or constant acceleration ``Oh**2 (Bo - H)`` when frictionless.
    """
    tau = np.asarray(tau, dtype=float)
    if regime.beta <= _BETA_EPS:
        acc = oh**2 * (bo - regime.h)
        return ca0 + acc * tau, ca0 * tau + 0.5 * acc * tau**2
    ca_inf = steady_capillary(bo, regime.h, regime.beta)
    t_v = relaxation_time(oh, regime.beta)
    decay = np.exp(-tau / t_v)
    return ca_inf + (ca0 - ca_inf) * decay, ca_inf * tau + (ca0 - ca_inf) * t_v * (1.0 - decay)


def stop_time(ca0: float, oh: float, bo: float, regime: Regime) -> float:
    """Time for ``Ca`` to fall to zero in *regime*; ``inf`` when it never does."""
    if ca0 <= 0.0:
        return 0.0 if bo <= regime.h else math.inf
    if regime.beta <= _BETA_EPS:
        acc = oh**2 * (bo - regime.h)
        return -ca0 / acc if acc < 0.0 else math.inf
    ca_inf = steady_capillary(bo, regime.h, regime.beta)
    if ca_inf >= 0.0:
        return math.inf
    return relaxation_time(oh, regime.beta) * math.log((ca0 - ca_inf) / (-ca_inf))


def crossing_time(distance: float, ca0: float, oh: float, bo: float, regime: Regime) -> float:
    """The single root of ``X(T) - X0 = distance`` in *regime*; ``inf`` if never reached.

    ``X`` is monotone while ``Ca > 0``, so the root is bracketed by the stop
    time (or grown geometrically when the droplet never stops) and bisected.
    """
    if distance <= 0.0:
        return 0.0
    t_stop = stop_time(ca0, oh, bo, regime)
    if math.isfinite(t_stop):
        reach = float(regime_state(np.array(t_stop), ca0, oh, bo, regime)[1])
        if reach < distance:
            return math.inf
        hi = t_stop
    else:
        hi = 1.0
        while float(regime_state(np.array(hi), ca0, oh, bo, regime)[1]) < distance:
            hi *= 2.0
    lo = 0.0
    for _ in range(_BISECT_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if float(regime_state(np.array(mid), ca0, oh, bo, regime)[1]) < distance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _regime_end(
    x_start: float, ca_start: float, oh: float, bo: float, boundary: float, regime: Regime
) -> tuple[float, bool]:
    """``(time spent in the regime, whether it ends by stopping)`` from ``(x_start, ca_start)``."""
    if ca_start <= 0.0 and bo <= regime.h:
        return math.inf, True
    dist = boundary - x_start
    t_cross = crossing_time(dist, ca_start, oh, bo, regime) if math.isfinite(dist) else math.inf
    t_stop = stop_time(ca_start, oh, bo, regime)
    return min(t_cross, t_stop), t_stop < t_cross


def piecewise_trajectory(
    t_grid: np.ndarray,
    x0: float,
    ca0: float,
    oh: float,
    bo: float,
    boundaries: Sequence[float],
    regimes: Sequence[Regime],
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ``(X, Ca)`` on *t_grid* for piecewise-constant coefficients.

    ``regimes[i]`` applies for ``boundaries[i-1] <= X < boundaries[i]``, so
    there is one more regime than boundary. Each regime relaxes
    exponentially; the time spent in it is the root of ``X(T) = boundary``.
    A droplet that stops with ``Bo <= H`` stays where it stopped.
    """
    if len(regimes) != len(boundaries) + 1:
        msg = "need exactly one more regime than boundaries"
        raise ValueError(msg)
    t_grid = np.asarray(t_grid, dtype=float)
    x_out = np.full_like(t_grid, np.nan)
    ca_out = np.zeros_like(t_grid)
    bounds = [*boundaries, math.inf]

    idx = int(np.searchsorted(np.asarray(boundaries), x0, side="right"))
    t_start, x_start, ca_start = float(t_grid[0]), x0, ca0
    stopped = False
    while not stopped:
        regime = regimes[idx]
        t_end, stopped = _regime_end(x_start, ca_start, oh, bo, bounds[idx], regime)
        if stopped and not math.isfinite(t_end):
            break
        mask = (t_grid >= t_start) & (t_grid <= t_start + t_end)
        ca_seg, dx_seg = regime_state(t_grid[mask] - t_start, ca_start, oh, bo, regime)
        x_out[mask], ca_out[mask] = x_start + dx_seg, ca_seg
        if not math.isfinite(t_end) or t_start + t_end >= t_grid[-1]:
            break
        ca_end, dx_end = regime_state(np.array(t_end), ca_start, oh, bo, regime)
        t_start, x_start, ca_start = t_start + t_end, x_start + float(dx_end), float(ca_end)
        idx += 0 if stopped else 1
    # After a stop the droplet stays where it stopped.
    rest = np.isnan(x_out)
    x_out[rest], ca_out[rest] = x_start, 0.0
    return x_out, ca_out


def _rk4_step(
    x: float,
    ca: float,
    dt: float,
    oh: float,
    bo: float,
    h_of_x: Callable[[float], float],
    beta_of_x: Callable[[float], float],
) -> tuple[float, float]:
    """One RK4 step of ``(dX/dT, dCa/dT) = (Ca, Oh**2 (Bo - H - beta Ca))``."""

    def rhs(xx: float, cc: float) -> tuple[float, float]:
        return cc, oh**2 * (bo - h_of_x(xx) - beta_of_x(xx) * cc)

    k1 = rhs(x, ca)
    k2 = rhs(x + 0.5 * dt * k1[0], ca + 0.5 * dt * k1[1])
    k3 = rhs(x + 0.5 * dt * k2[0], ca + 0.5 * dt * k2[1])
    k4 = rhs(x + dt * k3[0], ca + dt * k3[1])
    return (
        x + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        ca + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
    )


def _segment_bounds(x: float, breakpoints: np.ndarray) -> tuple[float, float]:
    """The breakpoint interval ``[lo, hi)`` containing *x*."""
    i = int(np.searchsorted(breakpoints, x, side="right"))
    lo = float(breakpoints[i - 1]) if i > 0 else -math.inf
    hi = float(breakpoints[i]) if i < breakpoints.size else math.inf
    return lo, hi


def _frozen(fn: Callable[[float], float], lo: float, hi: float) -> Callable[[float], float]:
    """*fn* evaluated with ``X`` clamped strictly inside ``[lo, hi)``."""
    eps = 1e-12
    upper = hi - eps * max(1.0, abs(hi)) if math.isfinite(hi) else math.inf

    def clamped(x: float) -> float:
        return fn(min(max(x, lo), upper))

    return clamped


def _land(
    x: float,
    ca: float,
    dt: float,
    hi: float,
    oh: float,
    bo: float,
    coeffs: tuple[Callable[[float], float], Callable[[float], float]],
) -> tuple[float, float, float]:
    """Shortest step within *dt* that reaches ``X = hi`` or ``Ca = 0``: ``(x, ca, step)``."""
    a, b = 0.0, dt
    for _ in range(_BISECT_ITERATIONS):
        mid = 0.5 * (a + b)
        xm, cm = _rk4_step(x, ca, mid, oh, bo, *coeffs)
        if xm >= hi or cm < 0.0:
            b = mid
        else:
            a = mid
    x_new, ca_new = _rk4_step(x, ca, b, oh, bo, *coeffs)
    return x_new, ca_new, b


def _advance(
    state: tuple[float, float, float],
    t_target: float,
    oh: float,
    bo: float,
    funcs: tuple[Callable[[float], float], Callable[[float], float]],
    bps: np.ndarray,
    dt_max: float,
) -> tuple[float, float, float, bool]:
    """Integrate ``(x, ca, t)`` to *t_target*; returns the new state and whether it got stuck."""
    h_of_x, beta_of_x = funcs
    x, ca, t = state
    while t < t_target - 1e-15:
        lo, hi = _segment_bounds(x, bps)
        coeffs = (_frozen(h_of_x, lo, hi), _frozen(beta_of_x, lo, hi))
        dt = min(dt_max, t_target - t)
        x_new, ca_new = _rk4_step(x, ca, dt, oh, bo, *coeffs)
        if x_new < hi and ca_new >= 0.0:
            x, ca, t = x_new, ca_new, t + dt
            continue
        x, ca, step = _land(x, ca, dt, hi, oh, bo, coeffs)
        t += step
        if ca < 0.0 and bo <= h_of_x(x):
            return x, 0.0, t_target, True
        ca = max(ca, 0.0)
    return x, ca, t, False


def integrate_trajectory(
    t_grid: np.ndarray,
    x0: float,
    ca0: float,
    oh: float,
    bo: float,
    h_of_x: Callable[[float], float],
    beta_of_x: Callable[[float], float],
    *,
    breakpoints: Sequence[float] = (),
    dt_max: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical ``(X, Ca)`` on *t_grid* for arbitrary ``H(X)`` and ``beta(X)``.

    Pass the positions where ``H`` or ``beta`` jump as *breakpoints*: within a
    step the coefficients are those of the segment the step started in, and
    a step that would leave the segment is shortened to end on its boundary.
    ``Ca`` reaching zero is located the same way; the droplet then stays put
    while ``Bo <= H`` there.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    bps = np.sort(np.asarray(breakpoints, dtype=float))
    x_out = np.empty_like(t_grid)
    ca_out = np.empty_like(t_grid)
    x, ca, t = float(x0), float(ca0), float(t_grid[0])
    stuck = ca <= 0.0 and bo <= h_of_x(x)
    x_out[0], ca_out[0] = x, max(ca, 0.0)
    for k in range(1, t_grid.size):
        if not stuck:
            x, ca, t, stuck = _advance((x, ca, t), float(t_grid[k]), oh, bo, (h_of_x, beta_of_x), bps, dt_max)
        x_out[k], ca_out[k] = x, ca
    return x_out, ca_out


def frictionless_stop_distance(h2: float, h3: float, width: float) -> float:
    """``X_f - X0 = (w/2)(1 - H2/H3)`` of a droplet released centred on the step.

    ``Bo = 0``, ``beta = 0``: it gains ``-H2 * w/2`` of kinetic energy until its
    trailing line reaches the step and spends it against ``H3``.
    """
    return 0.5 * width * (1.0 - h2 / h3)


def frictionless_crossing_capillary(oh: float, h2: float, width: float) -> float:
    """``Ca_s = Oh * sqrt(-H2 * w)`` when the trailing line reaches the step."""
    return oh * math.sqrt(-h2 * width)
