"""Sections D to G of the force-balance analysis, as tables.

D. Does the friction per unit contact length depend on anything else?
E. Measured against stepwise hysteresis: transition widths, the Gibbs hold of
   the trailing line at the step, depinning, and the onset of motion.
F. The Ca overshoot after the leading line reaches the step, as an inverse
   problem for the effective hysteresis.
G. Scaling collapse onto the regime plateaus and coasting lengths.
"""

from __future__ import annotations
import math
import warnings
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from src.simulation_io.analysis.force_balance.fits import MODEL_BY_KEY
from src.simulation_io.analysis.force_balance.fits import param_vector
from src.simulation_io.analysis.force_balance.geometry import StepClosure
from src.simulation_io.analysis.force_balance.model import integrate_trajectory
from src.simulation_io.analysis.force_balance.model import relaxation_time
from src.simulation_io.analysis.force_balance.model import steady_capillary
from src.simulation_io.analysis.force_balance.points import AFTER
from src.simulation_io.analysis.force_balance.points import STRADDLING
from src.simulation_io.analysis.force_balance.points import TRAIL_CROSS_MARGIN_LU
from src.simulation_io.analysis.force_balance.simulate import RunGeometry
from src.simulation_io.analysis.force_balance.simulate import steady_tolerance

if TYPE_CHECKING:
    from src.simulation_io.analysis.force_balance.config import AnalysisSettings
    from src.simulation_io.analysis.force_balance.fits import FitResult
    from src.simulation_io.analysis.force_balance.runs import RunInfo

#: Points with a smaller ``Ca`` are left out of the per-point friction ``beta_eff``.
_BETA_EFF_MIN_CA = 0.02

#: Minimum quasi-steady points of a regime for a per-run friction value.
_MIN_REGIME_POINTS = 4

#: Points either side of the transition used in the tanh fits.
_TANH_PAD = 0.4

_Z95 = 1.959963984540054

#: A dependence test needs this many distinct parameter values.
_MIN_DISTINCT_VALUES = 3

#: A tanh fit has four parameters; fewer points than this are not fitted.
_MIN_TANH_POINTS = 6

#: A plateau or coasting length below this cannot scale a collapse.
_NEGLIGIBLE = 1e-9

#: Run parameters tested for an influence on the friction per contact length.
DEPENDENCE_PARAMETERS = ("Bo", "Oh", "R0", "tau", "H1", "theta_A_phil", "theta_R_phil")


def add_point_friction(points: pd.DataFrame) -> pd.DataFrame:
    """The friction each point implies.

    ``beta_eff = (Bo - H_meas - inertia)/Ca``, and the SI's coefficient from
    ``beta_f = lambda A/eta``: ``lambda_eta_eff = beta_eff/A`` in 1/LU and
    ``lambda_eff = lambda_eta_eff * eta`` in lattice units.
    """
    out = points.copy()
    y = out["Bo"] - out["H_meas"] - out["inertia_cm"]
    ok = out["moving"] & (out["Ca_cm"] > _BETA_EFF_MIN_CA)
    out["beta_eff"] = np.where(ok, y / out["Ca_cm"].where(ok, 1.0), np.nan)
    out["lambda_eta_eff"] = out["beta_eff"] / out["A_lu"]
    out["lambda_eff"] = out["lambda_eta_eff"] * out["eta"]
    return out


def run_parameters(info: RunInfo) -> dict[str, float]:
    """The run-level parameters the dependence tests regress against."""
    return {
        "Bo": info.bo,
        "Oh": info.oh,
        "R0": info.r0,
        "tau": info.tau,
        "H1": info.windows.h1,
        "theta_A_phil": info.windows.advancing_phil,
        "theta_R_phil": info.windows.receding_phil,
        "W_over_R0": info.interface_width / info.r0,
    }


def friction_by_regime(points: pd.DataFrame, infos: dict[str, RunInfo]) -> pd.DataFrame:
    """Per run and regime: median ``beta_eff``, contact length, Ca and mean angle.

    Only quasi-steady moving points enter, and none with the trailing line held.
    """
    rows = []
    for run, df in points.groupby("run", sort=False):
        info = infos[str(run)]
        tol = steady_tolerance(df)
        for regime in (1, 2, 3):
            sel = df[
                (df["regime"] == regime) & df["beta_eff"].notna() & ~df["trail_held"] & (df["inertia_cm"].abs() < tol)
            ]
            if len(sel) < _MIN_REGIME_POINTS:
                continue
            beta = float(sel["beta_eff"].median())
            a_lu = float(sel["A_lu"].median())
            rows.append(
                {
                    "run": run,
                    "name": info.name,
                    "groups": ";".join(info.groups),
                    "regime": regime,
                    "n": len(sel),
                    "beta": beta,
                    "beta_iqr": float(sel["beta_eff"].quantile(0.75) - sel["beta_eff"].quantile(0.25)),
                    "A_over_R0": float(sel["A"].median()),
                    "A_lu": a_lu,
                    "A2_over_Omega": a_lu**2 / info.omega,
                    "Ca": float(sel["Ca_cm"].median()),
                    "theta_mean": float(((sel["theta_lead"] + sel["theta_trail"]) / 2).median()),
                    "lambda_per_lu": beta / a_lu,
                    "c_lub": beta / (a_lu**2 / info.omega),
                    **run_parameters(info),
                }
            )
    return pd.DataFrame(rows)


def _slope_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    """Least-squares slope of *y* on *x* with a bootstrap 95% interval over points."""
    slope = float(np.polyfit(x, y, 1)[0])
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, x.size, x.size)
        boot[b] = np.polyfit(x[pick], y[pick], 1)[0] if np.ptp(x[pick]) > 0 else np.nan
    return slope, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def dependence_tests(
    by_regime: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
    *,
    quantities: tuple[str, ...] = ("lambda_per_lu", "c_lub", "beta"),
    parameters: tuple[str, ...] = DEPENDENCE_PARAMETERS,
) -> pd.DataFrame:
    """Is ``lambda/eta = beta/A`` (or ``c = beta Omega/A**2``) independent of the run parameters?

    Per regime and parameter: Spearman's rank correlation over runs, and the
    slope of ``ln q`` on ``ln p`` (an exponent) with a bootstrap interval.
    """
    rows = []
    for quantity in quantities:
        for regime, sub in by_regime.groupby("regime"):
            for param in parameters:
                p = sub[param].to_numpy(dtype=float)
                q = sub[quantity].to_numpy(dtype=float)
                ok = np.isfinite(p) & np.isfinite(q) & (p > 0) & (q > 0)
                if np.unique(p[ok]).size < _MIN_DISTINCT_VALUES:
                    continue
                rho, pval = spearmanr(p[ok], q[ok])
                slope, lo, hi = _slope_ci(np.log(p[ok]), np.log(q[ok]), rng, n_boot)
                rows.append(
                    {
                        "quantity": quantity,
                        "regime": regime,
                        "parameter": param,
                        "n_runs": int(ok.sum()),
                        "spearman_rho": float(rho),
                        "p_value": float(pval),
                        "exponent": slope,
                        "exponent_ci_low": lo,
                        "exponent_ci_high": hi,
                        "cv": float(np.std(q[ok]) / np.mean(q[ok])),
                    }
                )
    return pd.DataFrame(rows)


def _tanh_step(x: np.ndarray, ha: float, hb: float, xc: float, ell: float) -> np.ndarray:
    return ha + 0.5 * (hb - ha) * (1.0 + np.tanh((x - xc) / ell))


def _fit_tanh(x: np.ndarray, h: np.ndarray, p0: tuple[float, float, float, float]) -> dict[str, float | bool]:
    """Fit :func:`_tanh_step`; width ``ell`` and centre with 95% intervals.

    A transition sharper than the sample spacing leaves ``ell`` undetermined
    (singular covariance). It is then reported as unresolved, with ``ell``
    set to the upper bound the data allow: the largest gap in ``X`` between
    the samples either side of the fitted centre.
    """
    nan: dict[str, float | bool] = {
        "ell": math.nan,
        "ell_ci": math.nan,
        "ell_resolved": False,
        "xc": math.nan,
        "xc_ci": math.nan,
        "ha": math.nan,
        "hb": math.nan,
    }
    if x.size < _MIN_TANH_POINTS:
        return nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt, pcov = curve_fit(_tanh_step, x, h, p0=p0, maxfev=20000)
        except (RuntimeError, ValueError):
            return nan
    err = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
    xc, ell = float(popt[2]), abs(float(popt[3]))
    order = np.sort(x)
    k = int(np.clip(np.searchsorted(order, xc), 1, order.size - 1))
    spacing = float(order[k] - order[k - 1])
    resolved = bool(np.isfinite(err[3]) and ell > 0.5 * spacing)
    return {
        "ell": ell if resolved else spacing,
        "ell_ci": _Z95 * float(err[3]) if resolved else math.nan,
        "ell_resolved": resolved,
        "xc": xc,
        "xc_ci": _Z95 * float(err[2]) if np.isfinite(err[2]) else math.nan,
        "ha": float(popt[0]),
        "hb": float(popt[1]),
    }


def _first(mask: np.ndarray) -> int | None:
    idx = np.nonzero(mask)[0]
    return int(idx[0]) if idx.size else None


def classify_motion(df: pd.DataFrame, info: RunInfo, settings: AnalysisSettings) -> str:
    """``moving`` (reaches the step or travels ``moving_dx``), ``creep``, or ``pinned``."""
    dx = float(df["X_cm"].iloc[-1] - df["X_cm"].iloc[0])
    reaches = bool((df["cll_lead"] >= info.step_x).any())
    if reaches or dx > settings.moving_dx:
        return "moving"
    if dx > settings.creep_dx:
        return "creep"
    return "pinned"


def _onset_fields(df: pd.DataFrame, info: RunInfo, settings: AnalysisSettings) -> dict[str, object]:
    """Observed against predicted motion, and when both lines first sit at their bounds.

    The threshold is the hysteresis of the regime the droplet starts in: H1
    normally, H2 for a droplet released straddling the step.
    """
    w = info.windows
    observed = classify_motion(df, info, settings)
    h_start = (w.h1, w.h2, w.h3)[int(df["regime"].iloc[0]) - 1]
    predicted = "moving" if info.bo > h_start else "pinned"
    both = _first((df["lead_state"] == "at_bound").to_numpy() & (df["trail_state"] == "at_bound").to_numpy())
    return {
        "H_start": h_start,
        "observed": observed,
        "predicted": predicted,
        "agrees": (observed == "moving") == (predicted == "moving"),
        "dX_total": float(df["X_cm"].iloc[-1] - df["X_cm"].iloc[0]),
        "t_bounds_reached": float(df["t"].iloc[both] - info.t_onset) if both is not None else math.nan,
        "T_bounds_reached": float(df["T"].iloc[both]) if both is not None else math.nan,
    }


def _hold_fields(df: pd.DataFrame, info: RunInfo, start: int) -> dict[str, object]:
    """The Gibbs hold of the trailing line from row *start*, and the tanh width of its H transition."""
    closure = StepClosure(info.windows)
    x, t, tt = df["X_cm"].to_numpy(), df["t"].to_numpy(), df["T"].to_numpy()
    after = np.nonzero(df["regime"].to_numpy()[start:] == AFTER)[0]
    end = start + int(after[0]) if after.size else x.size - 1
    sel = (x >= x[start] - _TANH_PAD) & (x <= x[end] + _TANH_PAD)
    p0 = (info.windows.h2, info.windows.h3, 0.5 * (x[start] + x[end]), 0.1)
    fit = _fit_tanh(x[sel], df["H_meas"].to_numpy()[sel], p0)
    return {
        "hold_start_X": float(x[start]),
        "hold_end_X": float(x[end]),
        "hold_dX": float(x[end] - x[start]),
        "hold_dT": float(tt[end] - tt[start]),
        "hold_dt_lattice": float(t[end] - t[start]),
        "hold_released": bool(after.size),
        "hold_theta_trail_start": float(df["theta_trail"].iloc[start]),
        "hold_theta_trail_end": float(df["theta_trail"].iloc[end]),
        "hold_A_start": float(df["A"].iloc[start]),
        "hold_A_end": float(df["A"].iloc[end]),
        "hold_dX_closure": closure.x_release - closure.x_hold,
        "A2_closure": closure.a2,
        "A3_closure": closure.a3,
        **{f"trail_{k}": v for k, v in fit.items()},
    }


def _lead_fields(df: pd.DataFrame, info: RunInfo, hold_start: int | None) -> dict[str, object]:
    """The tanh width of the H1 -> H2 transition as the leading line reaches the step."""
    lead_at = _first(df["regime"].to_numpy() >= STRADDLING)
    if lead_at is None:
        return {}
    x = df["X_cm"].to_numpy()
    hi = x[hold_start] if hold_start is not None else x[-1]
    sel = (x >= x[lead_at] - _TANH_PAD) & (x <= hi)
    fit = _fit_tanh(x[sel], df["H_meas"].to_numpy()[sel], (info.windows.h1, info.windows.h2, float(x[lead_at]), 0.05))
    return {f"lead_{k}": v for k, v in fit.items()}


def _coasting_fields(info: RunInfo, betas: tuple[float, float, float]) -> dict[str, float]:
    """Plateau ``Ca_inf_i`` and coasting length ``Lambda_i`` of each regime with the M1 friction."""
    w = info.windows
    out: dict[str, float] = {}
    for i, (h, beta) in enumerate(zip((w.h1, w.h2, w.h3), betas, strict=True), start=1):
        ca_inf = steady_capillary(info.bo, h, beta)
        out[f"Ca_inf_{i}_M1"] = ca_inf
        out[f"Lambda_{i}_M1"] = ca_inf * relaxation_time(info.oh, beta)
    return out


def hysteresis_table(
    points: pd.DataFrame, infos: dict[str, RunInfo], fits_all: dict[str, FitResult], settings: AnalysisSettings
) -> pd.DataFrame:
    """Per run: onset, depinning, Gibbs hold, tanh transition widths, coasting lengths."""
    m1 = fits_all["M1"]
    betas = (m1.params["beta_1"], m1.params["beta_2"], m1.params["beta_3"])
    rows = []
    for run, df in points.groupby("run", sort=False):
        info = infos[str(run)]
        w = info.windows
        row: dict[str, object] = {
            "run": run,
            "name": info.name,
            "groups": ";".join(info.groups),
            "Bo": info.bo,
            "Oh": info.oh,
            "R0": info.r0,
            "H1": w.h1,
            "H2": w.h2,
            "H3": w.h3,
            "learning_rate": info.learning_rate,
            **_onset_fields(df, info, settings),
        }
        start = _first(df["trail_held"].to_numpy())
        if start is not None:
            row.update(_hold_fields(df, info, start))
        row.update(_lead_fields(df, info, start))
        row.update(_coasting_fields(info, betas))
        rows.append(row)
    out = pd.DataFrame(rows)
    if "trail_ell" in out:
        out["trail_ell_over_Lambda3"] = out["trail_ell"] / out["Lambda_3_M1"].abs()
    return out


def smooth_limit_test(points: pd.DataFrame, fit_m4: FitResult, settings: AnalysisSettings) -> pd.DataFrame:
    """The SI's smooth-variation limit across the trailing-line transition.

    With ``beta(X)`` and ``H(X)`` varying on a length ``ell`` much longer than
    the coasting length ``Lambda``, ``Ca ~ Ca_inf(X) - Lambda dCa_inf/dX`` with
    ``Ca_inf = (Bo - H)/beta``. Evaluated with the measured ``H`` and the M4
    friction over the hold window, against the measured ``Ca``.
    """
    from scipy.signal import savgol_filter

    c = fit_m4.params["c_lub"]
    rows = []
    for run, df in points.groupby("run", sort=False):
        start = _first(df["trail_held"].to_numpy())
        if start is None:
            continue
        after = np.nonzero(df["regime"].to_numpy()[start:] == AFTER)[0]
        end = start + (int(after[0]) if after.size else len(df) - 1 - start)
        lo, hi = max(0, start - 5), min(len(df), end + 6)
        win = df.iloc[lo:hi]
        if len(win) < settings.smoothing_window:
            continue
        x = win["X_cm"].to_numpy()
        beta = c * win["A_lu"].to_numpy() ** 2 / win["Omega"].to_numpy()
        h = savgol_filter(win["H_meas"].to_numpy(), settings.smoothing_window, 1)
        oh = float(win["Oh"].iloc[0])
        ca_inf = (win["Bo"].to_numpy() - h) / beta
        lam = ca_inf / (oh**2 * beta)
        corrected = ca_inf - lam * np.gradient(ca_inf, x)
        data = win["Ca_cm"].to_numpy()
        rows.append(
            {
                "run": run,
                "rmse_quasi_steady": float(np.sqrt(np.mean((ca_inf - data) ** 2))),
                "rmse_smooth_limit": float(np.sqrt(np.mean((corrected - data) ** 2))),
                "Lambda_median": float(np.median(np.abs(lam))),
            }
        )
    return pd.DataFrame(rows)


#: Points at the end of the straddling phase whose median is the settled level.
_SETTLED_POINTS = 5


def _straddling_rows(df: pd.DataFrame) -> np.ndarray:
    """Positions of the regime-2 rows before the trailing line reaches the step.

    Reaching, not being held: a fast droplet's trailing line can arrive and be
    released between two snapshots, and the deceleration that follows must not
    count as the straddling level.
    """
    reach = float(df["step_x"].iloc[0]) - float(df["edge_width"].iloc[0]) - TRAIL_CROSS_MARGIN_LU
    arrived = df["cll_trail"].to_numpy() >= reach
    mask = df["regime"].to_numpy() == STRADDLING
    start = _first(arrived)
    if start is not None:
        mask[start:] = False
    return np.nonzero(mask)[0]


def _overshoot_of(df: pd.DataFrame, ca: np.ndarray) -> float:
    """Peak of *ca* while straddling minus its settled level there; NaN if too short."""
    rows = _straddling_rows(df)
    if rows.size <= _SETTLED_POINTS:
        return math.nan
    seg = ca[rows]
    return float(seg.max() - np.median(seg[-_SETTLED_POINTS:]))


def overshoot_table(points: pd.DataFrame) -> pd.DataFrame:
    """Per run: the Ca overshoot after the leading line reaches the step.

    ``overshoot = max Ca`` in regime 2 before the trailing line is held, minus
    the median of the last five such points (the level the droplet relaxes to
    while still straddling).
    """
    out = []
    for run, df in points.groupby("run", sort=False):
        rows = _straddling_rows(df)
        if rows.size <= _SETTLED_POINTS:
            continue
        ca = df["Ca_cm"].to_numpy()[rows]
        area = df["A"].to_numpy()[rows]
        k = int(np.argmax(ca))
        out.append(
            {
                "run": run,
                "peak": float(ca[k]),
                "x_peak": float(df["X_cm"].to_numpy()[rows][k]),
                "settled": float(np.median(ca[-_SETTLED_POINTS:])),
                "overshoot": _overshoot_of(df, df["Ca_cm"].to_numpy()),
                "A_at_peak": float(area[k]),
                "A_settled": float(np.median(area[-_SETTLED_POINTS:])),
            }
        )
    return pd.DataFrame(out)


def effective_hysteresis(points: pd.DataFrame, fit_m4: FitResult) -> pd.DataFrame:
    """``H_eff = Bo - (1/Oh**2) dCa/dT - beta_M4 Ca``: the hysteresis the data imply."""
    c = fit_m4.params["c_lub"]
    out = points[["run", "X_cm", "T", "Ca_cm", "H_meas", "H_step", "Oh", "Bo"]].copy()
    beta = c * points["A_lu"] ** 2 / points["Omega"]
    out["H_eff"] = points["Bo"] - points["inertia_cm"] - beta * points["Ca_cm"]
    out["sigma_H_eff"] = points["sigma_inertia_cm"]
    return out


def common_hysteresis_test(
    points: pd.DataFrame,
    infos: dict[str, RunInfo],
    runs: list[str],
    fit_m4: FitResult,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Forward-integrate each run with the group-mean ``H_eff(X)`` and its own Oh, Bo, A(X).

    If the overshoot is inertial, a common ``H_eff`` with run-specific
    ``Oh**2`` must reproduce each run's overshoot.
    """
    heff = effective_hysteresis(points[points["run"].isin(runs)], fit_m4)
    grid = np.linspace(heff["X_cm"].min(), heff["X_cm"].max(), 400)
    curves = []
    for run in runs:
        sub = heff[heff["run"] == run].sort_values("X_cm")
        curves.append(np.interp(grid, sub["X_cm"], sub["H_eff"], left=np.nan, right=np.nan))
    stack = np.vstack(curves)
    mean = np.nanmean(stack, axis=0)
    valid = np.isfinite(mean)
    grid, mean = grid[valid], mean[valid]
    spread = float(np.nanmean(np.nanstd(stack[:, valid], axis=0)))

    model = MODEL_BY_KEY["M4"]
    rows = []
    for run in runs:
        df = points[points["run"] == run]
        info = infos[run]
        geom = RunGeometry.from_points(df, info, settings)
        a_lu = np.abs(np.interp(grid, geom.x, geom.d_lead) - np.interp(grid, geom.x, geom.d_trail)) * info.r0
        beta = param_vector(fit_m4)[0] * a_lu**2 / info.omega
        hx, bx = grid.tolist(), beta.tolist()
        hy = mean.tolist()

        def h_of_x(v: float, hx: list[float] = hx, hy: list[float] = hy) -> float:
            return float(np.interp(v, hx, hy))

        def beta_of_x(v: float, hx: list[float] = hx, bx: list[float] = bx) -> float:
            return float(np.interp(v, hx, bx))

        t = np.concatenate([[0.0], df["T"].to_numpy()])
        _, ca_m = integrate_trajectory(t, geom.x0, 0.0, info.oh, info.bo, h_of_x, beta_of_x, dt_max=0.01)
        ca_m = ca_m[1:]
        data_os = _overshoot_of(df, df["Ca_cm"].to_numpy())
        model_os = _overshoot_of(df, ca_m)
        rows.append(
            {
                "run": run,
                "model": model.key + "+common_H_eff",
                "Oh": info.oh,
                "R0": info.r0,
                "rmse_T": float(np.sqrt(np.mean((ca_m - df["Ca_cm"].to_numpy()) ** 2))),
                "overshoot_data": data_os,
                "overshoot_model": model_os,
                "overshoot_fraction": model_os / data_os if data_os > 0 else math.nan,
                "H_eff_spread": spread,
            }
        )
    return pd.DataFrame(rows)


def _regime_rows(df: pd.DataFrame, regime: int) -> pd.DataFrame:
    """Rows of *regime*; for regime 2 only those before the trailing line reaches the step."""
    if regime == STRADDLING:
        return df.iloc[_straddling_rows(df)]
    return df[df["regime"] == regime]


def collapse_table(
    points: pd.DataFrame, infos: dict[str, RunInfo], fit_m1: FitResult, settings: AnalysisSettings
) -> pd.DataFrame:
    """Per point of a moving run: ``Ca/Ca_inf_i`` and ``(X - X_entry_i)/Lambda_i`` with the M1 coefficients."""
    betas = (fit_m1.params["beta_1"], fit_m1.params["beta_2"], fit_m1.params["beta_3"])
    parts = []
    for run, df in points.groupby("run", sort=False):
        info = infos[str(run)]
        if classify_motion(df, info, settings) != "moving":
            continue
        hs = (info.windows.h1, info.windows.h2, info.windows.h3)
        for i in (1, 2, 3):
            sel = _regime_rows(df, i)
            if sel.empty:
                continue
            ca_inf = steady_capillary(info.bo, hs[i - 1], betas[i - 1])
            lam = ca_inf * relaxation_time(info.oh, betas[i - 1])
            if abs(ca_inf) < _NEGLIGIBLE or abs(lam) < _NEGLIGIBLE:
                continue
            parts.append(
                pd.DataFrame(
                    {
                        "run": run,
                        "groups": ";".join(info.groups),
                        "regime": i,
                        "Ca_ratio": sel["Ca_cm"].to_numpy() / ca_inf,
                        "s": (sel["X_cm"].to_numpy() - float(sel["X_cm"].iloc[0])) / abs(lam),
                        "Ca_inf": ca_inf,
                        "Lambda": lam,
                    }
                )
            )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
