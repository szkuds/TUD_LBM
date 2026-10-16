"""Forward integration of a fitted model for one run, and the trajectory metrics.

The ODE needs ``H`` and ``beta`` as functions of the centre-of-mass position
``X``. Geometry is not part of the reduced model, so it is taken from the run
itself: the contact-line offsets ``X_lead - X_cm`` and ``X_trail - X_cm``, the
measured angles and ``H_meas``, and the line-to-centre speed ratios are
tabulated against the run's own (smoothed, monotone) ``X_cm`` and read back at
the model's ``X``. A model is therefore judged on its dynamics, with the
droplet's shape given.
"""

from __future__ import annotations
import bisect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from scipy.signal import savgol_filter
from src.simulation_io.analysis.force_balance.geometry import StepClosure
from src.simulation_io.analysis.force_balance.model import integrate_trajectory
from src.simulation_io.analysis.force_balance.points import AFTER
from src.simulation_io.analysis.force_balance.points import STRADDLING
from src.simulation_io.analysis.force_balance.points import TRAIL_CROSS_MARGIN_LU
from src.simulation_io.analysis.force_balance.points import regime_of
from src.simulation_io.analysis.force_balance.points import split_contact_length

if TYPE_CHECKING:
    from collections.abc import Callable
    import pandas as pd
    from src.simulation_io.analysis.force_balance.config import AnalysisSettings
    from src.simulation_io.analysis.force_balance.fits import FrictionModel
    from src.simulation_io.analysis.force_balance.runs import RunInfo

#: Resolution of the tabulated coefficient functions, in samples over the run.
_TABLE_SIZE = 4000

#: The model is integrated this many times past the data's duration when it is
#: still short of the data's furthest position, so ``Ca(X)`` can be compared.
_EXTENSION_FACTOR = 3.0

#: Minimum points of a regime for its plateau to be defined.
_MIN_PLATEAU_POINTS = 4

#: Fraction of the run whose median ``Ca`` is the final value.
_FINAL_FRACTION = 0.1

_E_FOLD = math.exp(-1.0)

#: Samples of the extension past the data's duration (the first is the data's last time).
_EXTENSION_POINTS = 200

#: Smallest step of the running-maximum X kept in the geometry tables.
_MONOTONE_EPS = 1e-9


class _Table:
    """Piecewise-linear lookup on python lists: a scalar ``np.interp`` without the overhead."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]

    def __call__(self, x: float) -> float:
        xs, ys = self.x, self.y
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect.bisect_right(xs, x)
        w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
        return ys[i - 1] + w * (ys[i] - ys[i - 1])


@dataclass(frozen=True)
class RunGeometry:
    """A run's shape and measured angles as functions of ``X_cm``."""

    x: np.ndarray
    d_lead: np.ndarray
    d_trail: np.ndarray
    h_meas: np.ndarray
    theta_lead: np.ndarray
    theta_trail: np.ndarray
    r_lead: np.ndarray
    r_trail: np.ndarray
    x0: float
    info: RunInfo

    @classmethod
    def from_points(cls, df: pd.DataFrame, info: RunInfo, settings: AnalysisSettings) -> RunGeometry:
        """Tabulate *df* against its smoothed, strictly increasing ``X_cm``."""
        x_s = savgol_filter(
            df["X_cm"].to_numpy(), settings.smoothing_window, settings.smoothing_polyorder, mode="interp"
        )
        running = np.maximum.accumulate(x_s)
        keep = np.concatenate([[True], np.diff(running) > _MONOTONE_EPS])
        idx = np.nonzero(keep)[0]
        ca = df["Ca_cm"].to_numpy()
        moving = df["moving"].to_numpy()
        safe = np.where(moving, ca, 1.0)
        r_lead = np.where(moving, df["Ca_lead"].to_numpy() / safe, 1.0)
        r_trail = np.where(moving, df["Ca_trail"].to_numpy() / safe, 1.0)
        x_cm = df["X_cm"].to_numpy()
        return cls(
            x=running[idx],
            d_lead=(df["X_lead"].to_numpy() - x_cm)[idx],
            d_trail=(df["X_trail"].to_numpy() - x_cm)[idx],
            h_meas=df["H_meas"].to_numpy()[idx],
            theta_lead=np.radians(df["theta_lead"].to_numpy())[idx],
            theta_trail=np.radians(df["theta_trail"].to_numpy())[idx],
            r_lead=r_lead[idx],
            r_trail=r_trail[idx],
            x0=float(x_s[0]),
            info=info,
        )

    def lines_lu(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(lead, trail)`` contact-line positions in lattice units at centre positions *x*."""
        r0, xs = self.info.r0, self.info.step_x
        lead = xs + r0 * (x + np.interp(x, self.x, self.d_lead))
        trail = xs + r0 * (x + np.interp(x, self.x, self.d_trail))
        return lead, trail

    def regime_bounds(self) -> tuple[float, float]:
        """Centre positions where the leading, then the trailing, line reaches the step."""
        grid = np.linspace(self.x[0] - 1.0, self.x[-1] + 1.0, _TABLE_SIZE)
        lead, trail = self.lines_lu(grid)
        regime = regime_of(lead, trail, self.info.step_x)
        first2 = np.nonzero(regime >= STRADDLING)[0]
        first3 = np.nonzero(regime >= AFTER)[0]
        b2 = float(grid[first2[0]]) if first2.size else math.inf
        b3 = float(grid[first3[0]]) if first3.size else math.inf
        return b2, max(b2, b3)

    def table(self, values: np.ndarray) -> _Table:
        """*values* (on a fine grid over the run) as a function of ``X``."""
        return _Table(self.grid(), values)

    def grid(self) -> np.ndarray:
        """The fine ``X`` grid the coefficient tables are built on."""
        return np.linspace(self.x[0] - 1.0, self.x[-1] + 1.0, _TABLE_SIZE)


def _closure_coefficients(
    info: RunInfo, params: np.ndarray
) -> tuple[Callable[[float], float], Callable[[float], float], tuple[float, ...]]:
    """Model M6: everything from the windows and ``Omega``, nothing from the run."""
    closure = StepClosure(info.windows)
    c = float(np.nan_to_num(params)[0])

    def beta_of_x(x: float) -> float:
        return c * closure.base(x) ** 2

    return closure.hysteresis, beta_of_x, closure.breakpoints


def _stepwise(values: tuple[float, ...], bounds: tuple[float, float]) -> Callable[[float], float]:
    """``values[i]`` in regime ``i + 1``, the regimes split at *bounds*."""

    def of_x(x: float) -> float:
        return values[bisect.bisect_right(bounds, x)]

    return of_x


def _tabulated_friction(geom: RunGeometry, key: str, p: np.ndarray) -> np.ndarray:
    """``beta`` on :meth:`RunGeometry.grid` for the models whose friction varies with the shape."""
    info = geom.info
    grid = geom.grid()
    lead, trail = geom.lines_lu(grid)
    if key in ("M2", "M3"):
        a_phob, a_phil = split_contact_length(lead, trail, info.step_x)
        return p[0] * a_phob + p[1] * a_phil
    if key == "M7":
        return p[0] * (lead - trail)
    if key in ("M4", "M4s"):
        return p[0] * (lead - trail) ** 2 / info.omega
    if key == "M5":
        lead_term = np.interp(grid, geom.x, geom.r_lead) / np.interp(grid, geom.x, geom.theta_lead)
        trail_term = np.interp(grid, geom.x, geom.r_trail) / np.interp(grid, geom.x, geom.theta_trail)
        return p[0] * (lead_term + trail_term)
    msg = f"no forward form for model {key}"
    raise ValueError(msg)


def _coefficients(
    geom: RunGeometry, model: FrictionModel, params: np.ndarray
) -> tuple[Callable[[float], float], Callable[[float], float], tuple[float, ...]]:
    """``H(X)``, ``beta(X)`` and the breakpoints of *model* on this run."""
    info = geom.info
    if model.hysteresis == "H_geom":
        return _closure_coefficients(info, params)
    bounds = geom.regime_bounds()
    p = np.nan_to_num(params)
    if model.hysteresis == "H_step":
        h_of_x = _stepwise((info.windows.h1, info.windows.h2, info.windows.h3), bounds)
    else:
        h_of_x = geom.table(np.interp(geom.grid(), geom.x, geom.h_meas))
    if model.key in ("M0", "M0m"):
        beta_of_x = _stepwise((float(p[0]),) * 3, bounds)
    elif model.key == "M1":
        beta_of_x = _stepwise(tuple(float(v) for v in p), bounds)
    else:
        beta_of_x = geom.table(_tabulated_friction(geom, model.key, p))
    return h_of_x, beta_of_x, tuple(b for b in bounds if math.isfinite(b))


@dataclass(frozen=True)
class Trajectory:
    """A model trajectory on the data's time grid plus its extension."""

    t: np.ndarray
    x: np.ndarray
    ca: np.ndarray
    n_data: int


def integrate_with(
    geom: RunGeometry,
    h_of_x: Callable[[float], float],
    beta_of_x: Callable[[float], float],
    breakpoints: tuple[float, ...],
    t_data: np.ndarray,
    settings: AnalysisSettings,
) -> Trajectory:
    """Integrate ``H(X)``, ``beta(X)`` from rest at the run's first position, gravity on at ``T = 0``."""
    info = geom.info
    t = time_grid(t_data)
    x, ca = integrate_trajectory(
        t, geom.x0, 0.0, info.oh, info.bo, h_of_x, beta_of_x, breakpoints=breakpoints, dt_max=settings.integration_dt
    )
    return trajectory_from(t, x, ca)


def trajectory_from(t: np.ndarray, x: np.ndarray, ca: np.ndarray) -> Trajectory:
    """A trajectory on a :func:`time_grid`, marking where the data's duration ends."""
    return Trajectory(t=t, x=x, ca=ca, n_data=t.size - _EXTENSION_POINTS + 1)


def time_grid(t_data: np.ndarray) -> np.ndarray:
    """``T = 0``, the data's times, then an extension to ``_EXTENSION_FACTOR`` times their span."""
    t_end = float(t_data[-1])
    extension = np.linspace(t_end, _EXTENSION_FACTOR * t_end, _EXTENSION_POINTS)[1:]
    return np.concatenate([[0.0], t_data[t_data > 0.0], extension])


def simulate(
    geom: RunGeometry, model: FrictionModel, params: np.ndarray, t_data: np.ndarray, settings: AnalysisSettings
) -> Trajectory:
    """Integrate *model* from rest at the run's first position, gravity on at ``T = 0``."""
    h_of_x, beta_of_x, bps = _coefficients(geom, model, params)
    return integrate_with(geom, h_of_x, beta_of_x, bps, t_data, settings)


def area_friction(geom: RunGeometry, c: float) -> Callable[[float], float]:
    """``beta(X) = c A(X)**2/Omega`` with the run's measured base width."""
    lead, trail = geom.lines_lu(geom.grid())
    return geom.table(c * (lead - trail) ** 2 / geom.info.omega)


def measured_hysteresis(geom: RunGeometry) -> Callable[[float], float]:
    """The run's measured ``H(X)``."""
    return geom.table(np.interp(geom.grid(), geom.x, geom.h_meas))


def _plateaus(ca: np.ndarray, regime: np.ndarray, steady: np.ndarray, reached: int) -> dict[str, float]:
    """``plateau_i``: median ``Ca`` of the quasi-steady points of each regime.

    Straddling ends, for its plateau, at row *reached*, where the trailing line
    reaches the step: a droplet that then decelerates or stops held there is in
    regime 2 but not on its plateau.
    """
    before = np.arange(ca.size) < reached
    out: dict[str, float] = {}
    for i in (1, 2, 3):
        sel = (regime == i) & steady & (before if i == STRADDLING else np.ones_like(before))
        out[f"plateau_{i}"] = float(np.median(ca[sel])) if np.sum(sel) >= _MIN_PLATEAU_POINTS else math.nan
    return out


def trajectory_metrics(
    traj: dict[str, np.ndarray],
    step_x: float,
    edge_width: float,
    stop_ca: float,
    steady_tol: float,
) -> dict[str, float]:
    """Plateaus, peak, rise and decay lengths and stopping position of one trajectory.

    *traj* holds ``t``, ``x``, ``ca``, ``inertia`` (``(1/Oh**2) dCa/dT``) and the
    contact lines ``lead``/``trail`` in lattice units.

    * ``plateau_i``: median ``Ca`` over the quasi-steady points of regime ``i``,
      those with ``|inertia| < steady_tol``; for regime 2 only those before the
      trailing line reaches the step. A droplet decelerating or stopped with
      its trailing line held there is in regime 2 but not on its plateau.
    * ``peak``: maximum ``Ca``; ``x_peak`` where it occurs.
    * ``rise_1``: ``X`` travelled from the start until ``Ca`` first reaches
      ``(1 - 1/e)`` of ``plateau_1``.
    * ``decay``: ``X`` travelled after the trailing line *reaches* the step
      (within ``edge_width`` of it, where the solver starts holding it) until
      ``Ca - Ca_final`` has fallen to ``1/e`` of its value there.
    * ``x_stop``: final ``X`` when the final ``Ca`` is below *stop_ca*.
    """
    x, ca, inertia = traj["x"], traj["ca"], traj["inertia"]
    lead, trail = traj["lead"], traj["trail"]
    regime = regime_of(lead, trail, step_x)
    reached = np.nonzero(trail >= step_x - edge_width - TRAIL_CROSS_MARGIN_LU)[0]
    out = _plateaus(ca, regime, np.abs(inertia) < steady_tol, int(reached[0]) if reached.size else ca.size)
    k = int(np.argmax(ca))
    out["peak"], out["x_peak"] = float(ca[k]), float(x[k])
    n_final = max(1, int(_FINAL_FRACTION * ca.size))
    ca_final = float(np.median(ca[-n_final:]))
    out["ca_final"] = ca_final
    out["x_final"] = float(x[-1])
    out["x_stop"] = float(x[-1]) if ca_final < stop_ca else math.nan

    p1 = out["plateau_1"]
    out["rise_1"] = math.nan
    if math.isfinite(p1) and p1 > 0:
        hit = np.nonzero(ca >= (1.0 - _E_FOLD) * p1)[0]
        if hit.size:
            out["rise_1"] = float(x[hit[0]] - x[0])

    out["x_trail_at_step"] = math.nan
    out["decay"] = math.nan
    if reached.size:
        c = int(reached[0])
        out["x_trail_at_step"] = float(x[c])
        excess = ca[c:] - ca_final
        if excess[0] > 0:
            below = np.nonzero(excess <= _E_FOLD * excess[0])[0]
            if below.size:
                out["decay"] = float(x[c + below[0]] - x[c])
    return out


def steady_tolerance(df: pd.DataFrame) -> float:
    """Inertia below three times the run's median noise counts as quasi-steady."""
    return 3.0 * float(np.median(df["sigma_inertia_cm"]))


def data_metrics(df: pd.DataFrame, info: RunInfo, stop_ca: float) -> dict[str, float]:
    """:func:`trajectory_metrics` of the measured run."""
    traj = {
        "t": df["T"].to_numpy(),
        "x": df["X_cm"].to_numpy(),
        "ca": df["Ca_cm"].to_numpy(),
        "inertia": df["inertia_cm"].to_numpy(),
        "lead": df["cll_lead"].to_numpy(),
        "trail": df["cll_trail"].to_numpy(),
    }
    return trajectory_metrics(traj, info.step_x, info.edge_width, stop_ca, steady_tolerance(df))


def model_metrics(geom: RunGeometry, traj: Trajectory, stop_ca: float, steady_tol: float) -> dict[str, float]:
    """:func:`trajectory_metrics` of a model trajectory over the data's duration.

    The contact lines are the run's own, read at the model's ``X`` (the shape
    the model was integrated with). ``x_stop`` uses the extended trajectory,
    so a model that stops just after the data end still reports where.
    """
    n = traj.n_data
    t, x, ca = traj.t[:n], traj.x[:n], traj.ca[:n]
    lead, trail = geom.lines_lu(x)
    info = geom.info
    inertia = np.gradient(ca, t) / info.oh**2
    series = {"t": t, "x": x, "ca": ca, "inertia": inertia, "lead": lead, "trail": trail}
    out = trajectory_metrics(series, info.step_x, info.edge_width, stop_ca, steady_tol)
    out["x_stop"] = float(traj.x[-1]) if traj.ca[-1] < stop_ca else math.nan
    return out


def model_ca_at_data_x(df: pd.DataFrame, traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """``(model Ca at each measured X_cm, mask of the points it can be compared at)``.

    Beyond the model's furthest point its ``Ca`` is zero if it has stopped there;
    if it is still moving, those points are masked out.
    """
    order = np.argsort(traj.x, kind="stable")
    mx, mca = traj.x[order], traj.ca[order]
    xd = df["X_cm"].to_numpy()
    inside = xd <= mx[-1]
    pred = np.interp(xd, mx, mca)
    if traj.ca[-1] <= 0.0:
        return np.where(inside, pred, 0.0), np.ones_like(inside)
    return pred, inside


def ca_of_x_error(df: pd.DataFrame, traj: Trajectory) -> tuple[float, float, float]:
    """``(RMSE of Ca(X), RMSE of Ca(T), fraction of data X the model reaches)``."""
    pred, use = model_ca_at_data_x(df, traj)
    cd = df["Ca_cm"].to_numpy()
    rmse_x = float(np.sqrt(np.mean((pred[use] - cd[use]) ** 2))) if use.any() else math.nan
    n = traj.n_data
    pred_t = np.interp(df["T"].to_numpy(), traj.t[:n], traj.ca[:n])
    rmse_t = float(np.sqrt(np.mean((pred_t - cd) ** 2)))
    return rmse_x, rmse_t, float(np.mean(use))
