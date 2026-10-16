"""The per-data-point table: every saved snapshot as one row of the force balance.

Positions are measured from the step and scaled by ``R0 = sqrt(Omega_D)``.
Velocities and accelerations are Savitzky-Golay derivatives of the positions
(window and order from the settings), and the noise of each position series is
estimated from the median absolute deviation of its second differences and
propagated through the filter coefficients into a standard deviation for every
derivative.

The right contact line leads and the left trails: gravity drives every run
toward +x.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs
from scipy.signal import savgol_filter
from src.simulation_io.analysis.force_balance.geometry import StepClosure

if TYPE_CHECKING:
    from src.simulation_io.analysis.droplet_metrics import DropletSeries
    from src.simulation_io.analysis.force_balance.config import AnalysisSettings
    from src.simulation_io.analysis.force_balance.runs import AssumptionLog
    from src.simulation_io.analysis.force_balance.runs import RunInfo

#: Gaussian consistency factor of the median absolute deviation.
_MAD_TO_SIGMA = 1.4826

#: The second difference of white noise of variance s**2 has variance 6 s**2.
_SECOND_DIFF_VARIANCE = 6.0

#: Step-model regimes: both lines before the step, straddling it, both past it.
BEFORE, STRADDLING, AFTER = 1, 2, 3

#: A trailing line counts as past the step only this far beyond it (LU). A line
#: held at the step sits within a cell of it and would otherwise flip between
#: regimes 2 and 3 from one snapshot to the next.
TRAIL_CROSS_MARGIN_LU = 0.5


def noise_sigma(values: np.ndarray) -> float:
    """Noise standard deviation of a smooth series from its second differences."""
    d2 = np.diff(np.asarray(values, dtype=float), n=2)
    if d2.size == 0:
        return 0.0
    mad = float(np.median(np.abs(d2 - np.median(d2))))
    return _MAD_TO_SIGMA * mad / math.sqrt(_SECOND_DIFF_VARIANCE)


def _derivatives(
    values: np.ndarray, dt: float, settings: AnalysisSettings
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """``(d/dt, d2/dt2, sigma_d1, sigma_d2)`` of *values* sampled every *dt*."""
    w, p = settings.smoothing_window, settings.smoothing_polyorder
    d1 = savgol_filter(values, w, p, deriv=1, delta=dt, mode="interp")
    d2 = savgol_filter(values, w, p, deriv=2, delta=dt, mode="interp")
    sigma = noise_sigma(values)
    s1 = sigma * float(np.linalg.norm(savgol_coeffs(w, p, deriv=1, delta=dt)))
    s2 = sigma * float(np.linalg.norm(savgol_coeffs(w, p, deriv=2, delta=dt)))
    return d1, d2, s1, s2


def _surface_window(position: np.ndarray, info: RunInfo, *, receding: bool) -> np.ndarray:
    """Window bound the solver applies to a line at *position* (lattice units).

    The solver switches a line to the post-step window once it is within
    ``edge_width`` of the step, so that is the rule here too.
    """
    post = position >= info.step_x - info.edge_width
    w = info.windows
    if receding:
        return np.where(post, w.receding_phil, w.receding_phob)
    return np.where(post, w.advancing_phil, w.advancing_phob)


def _line_state(theta: np.ndarray, bound: np.ndarray, tol: float, *, receding: bool) -> np.ndarray:
    """``at_bound`` within *tol* of the bound, ``inside`` the window, else ``beyond``."""
    at = np.abs(theta - bound) <= tol
    inside = theta > bound + tol if receding else theta < bound - tol
    return np.where(at, "at_bound", np.where(inside, "inside", "beyond"))


def regime_of(lead_lu: np.ndarray, trail_lu: np.ndarray, step_x: float) -> np.ndarray:
    """Step-model regime from the contact-line positions: 1 before, 2 straddling, 3 after.

    Regime 3 starts once the trailing line is :data:`TRAIL_CROSS_MARGIN_LU` past the step.
    """
    return np.where(lead_lu < step_x, BEFORE, np.where(trail_lu < step_x + TRAIL_CROSS_MARGIN_LU, STRADDLING, AFTER))


def split_contact_length(lead_lu: np.ndarray, trail_lu: np.ndarray, step_x: float) -> tuple[np.ndarray, np.ndarray]:
    """``(A_phob, A_phil)`` in lattice units: the base clipped at the step."""
    phob = np.clip(np.minimum(lead_lu, step_x) - trail_lu, 0.0, None)
    phil = np.clip(lead_lu - np.maximum(trail_lu, step_x), 0.0, None)
    return phob, phil


def si_regime(x_cm: np.ndarray, width: float) -> np.ndarray:
    """The SI's regime of a centre at *x_cm* (from the step, units of R0) with constant base *width*.

    The lines sit at ``X -/+ w/2`` (SI Eq. Hstep), so the switches are at ``X = -w/2`` and ``+w/2``.
    """
    return np.where(x_cm < -0.5 * width, BEFORE, np.where(x_cm < 0.5 * width, STRADDLING, AFTER))


def step_hysteresis(regime: np.ndarray, info: RunInfo) -> np.ndarray:
    """``H1``, ``H2`` or ``H3`` by regime."""
    w = info.windows
    return np.select([regime == BEFORE, regime == STRADDLING], [w.h1, w.h2], default=w.h3)


def build_point_table(
    info: RunInfo, series: DropletSeries, settings: AnalysisSettings, log: AssumptionLog
) -> pd.DataFrame:
    """One row per saved snapshot of *series*, with every term of the force balance."""
    it = series.iteration.astype(float)
    keep = series.cll_right < info.nx - settings.seam_margin_lu
    if not np.all(keep):
        cut = int(np.argmin(keep))
        log.add(
            info.run,
            f"truncated at t={int(it[cut])}: leading line within {settings.seam_margin_lu} LU of the periodic seam",
        )
        keep[cut:] = False
    gaps = np.diff(it[keep])
    if gaps.size and not np.allclose(gaps, gaps[0]):
        msg = f"{info.run}: non-uniform snapshot spacing; Savitzky-Golay derivatives need a uniform grid"
        raise ValueError(msg)
    dt = float(gaps[0])

    t = it[keep]
    cm, lead, trail = series.cm_x[keep], series.cll_right[keep], series.cll_left[keep]
    theta_lead, theta_trail = series.theta_right[keep], series.theta_left[keep]
    xs, r0 = info.step_x, info.r0
    ca_per_v = info.eta / info.gamma
    dca_dt_per_a = ca_per_v * info.time_scale
    inv_oh2 = 1.0 / info.oh**2

    table: dict[str, np.ndarray | float | str] = {
        "run": info.run,
        "name": info.name,
        "t": t,
        "T": (t - info.t_onset) / info.time_scale,
        "cm_x": cm,
        "cll_lead": lead,
        "cll_trail": trail,
        "X_cm": (cm - xs) / r0,
        "X_lead": (lead - xs) / r0,
        "X_trail": (trail - xs) / r0,
        "groups": ";".join(info.groups),
        "Bo": info.bo,
        "Oh": info.oh,
        "R0": info.r0,
        "Omega": info.omega,
        "eta": info.eta,
        "step_x": xs,
        "edge_width": info.edge_width,
    }
    for label, pos in (("cm", cm), ("lead", lead), ("trail", trail)):
        v, a, sv, sa = _derivatives(pos, dt, settings)
        table[f"Ca_{label}"] = ca_per_v * v
        table[f"sigma_Ca_{label}"] = ca_per_v * sv
        table[f"inertia_{label}"] = inv_oh2 * dca_dt_per_a * a
        table[f"sigma_inertia_{label}"] = inv_oh2 * dca_dt_per_a * sa
    table["Ca_u"] = ca_per_v * series.avg_u_x[keep]

    a_lu = lead - trail
    a_phob, a_phil = split_contact_length(lead, trail, xs)
    regime = regime_of(lead, trail, xs)
    table.update(
        A=a_lu / r0,
        A_lu=a_lu,
        A_phob_lu=a_phob,
        A_phil_lu=a_phil,
        theta_lead=theta_lead,
        theta_trail=theta_trail,
        H_meas=np.cos(np.radians(theta_trail)) - np.cos(np.radians(theta_lead)),
        regime=regime,
        H_step=step_hysteresis(regime, info),
    )
    # The SI model: one base width, the measured initial one, for the whole run.
    width = float(a_lu[0] / r0)
    regime_si = si_regime((cm - xs) / r0, width)
    table.update(w_si=width, regime_si=regime_si, H_si=step_hysteresis(regime_si, info))
    closure = StepClosure(info.windows)
    a_geom, h_geom = closure.vectorised(table["X_cm"])
    table.update(A_geom=a_geom, H_geom=h_geom)
    rec = _surface_window(trail, info, receding=True)
    adv = _surface_window(lead, info, receding=False)
    tol = settings.angle_tolerance_deg
    table.update(
        trail_bound=rec,
        lead_bound=adv,
        H_window=np.cos(np.radians(rec)) - np.cos(np.radians(adv)),
        trail_state=_line_state(theta_trail, rec, tol, receding=True),
        lead_state=_line_state(theta_lead, adv, tol, receding=False),
    )
    df = pd.DataFrame(table)
    df["lead_pinned"] = np.abs(df["Ca_lead"]) < settings.line_pinned_sigma * df["sigma_Ca_lead"]
    df["trail_pinned"] = np.abs(df["Ca_trail"]) < settings.line_pinned_sigma * df["sigma_Ca_trail"]
    floor = np.maximum(settings.moving_sigma * df["sigma_Ca_cm"], settings.ca_min)
    df["moving"] = df["Ca_cm"] > floor
    near_step = np.abs(df["cll_trail"] - xs) <= info.edge_width + TRAIL_CROSS_MARGIN_LU
    df["trail_held"] = (df["regime"] == STRADDLING) & near_step & df["trail_pinned"]
    return df
