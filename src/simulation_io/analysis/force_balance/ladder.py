"""The model ladder: the SI model, then the same ODE with one SI assumption at a time replaced by data.

=============  ===========================  ==========================  =============================
rung           H(X)                         beta                        isolates
=============  ===========================  ==========================  =============================
SI             stepwise at X_s +/- w/2      beta_f, per run             the SI model, data-calibrated
SI+beta(A)     stepwise at X_s +/- w/2      c A(X)**2/Omega, measured   the contact-length friction
SI+H_meas      measured                     beta_f, per run             the hysteresis physics
M7             measured                     (lambda/eta) A(X), measured the SI friction law, filled in
M4             measured                     c A(X)**2/Omega, measured   both (the data-filled model)
M6             cap closure                  c A_cap(X)**2               the predictive closure
=============  ===========================  ==========================  =============================

The SI rung is the SI's own closed form (``piecewise_trajectory``), with its
constant base width ``w`` the run's measured initial one and its friction
``beta_f = (Bo - H1)/Ca_1`` from the run's own plateau before the step. A run
without that plateau (pinned, or released on the step) takes the value fitted to
all runs, and says so.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from src.simulation_io.analysis.force_balance.analyses import classify_motion
from src.simulation_io.analysis.force_balance.decomposition import PHASES
from src.simulation_io.analysis.force_balance.decomposition import phase_labels
from src.simulation_io.analysis.force_balance.fits import MODEL_BY_KEY
from src.simulation_io.analysis.force_balance.fits import FrictionModel
from src.simulation_io.analysis.force_balance.fits import fit_model
from src.simulation_io.analysis.force_balance.fits import param_vector
from src.simulation_io.analysis.force_balance.model import Regime
from src.simulation_io.analysis.force_balance.model import piecewise_trajectory
from src.simulation_io.analysis.force_balance.points import BEFORE
from src.simulation_io.analysis.force_balance.simulate import RunGeometry
from src.simulation_io.analysis.force_balance.simulate import Trajectory
from src.simulation_io.analysis.force_balance.simulate import area_friction
from src.simulation_io.analysis.force_balance.simulate import integrate_with
from src.simulation_io.analysis.force_balance.simulate import measured_hysteresis
from src.simulation_io.analysis.force_balance.simulate import model_ca_at_data_x
from src.simulation_io.analysis.force_balance.simulate import simulate
from src.simulation_io.analysis.force_balance.simulate import steady_tolerance
from src.simulation_io.analysis.force_balance.simulate import time_grid
from src.simulation_io.analysis.force_balance.simulate import trajectory_from

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.simulation_io.analysis.force_balance.config import AnalysisSettings
    from src.simulation_io.analysis.force_balance.fits import FitResult
    from src.simulation_io.analysis.force_balance.runs import RunInfo

RUNGS = ("SI", "SI+beta(A)", "SI+H_meas", "M7", "M4", "M6")

#: Quasi-steady points needed before the step for a plateau.
_MIN_PLATEAU_POINTS = 4

#: The SI model's one friction coefficient, fitted to every run with its own stepwise H.
SI_MODEL = FrictionModel(
    "SI", "SI: constant beta, H stepwise at X +/- w/2", "H_si", ("beta",), MODEL_BY_KEY["M0"].basis
)


@dataclass(frozen=True)
class Calibration:
    """The SI model's two free numbers for one run, and where ``beta_f`` came from."""

    beta_f: float
    beta_source: str
    width: float
    ca_plateau_1: float


def pre_step_plateau(df: pd.DataFrame) -> float:
    """Median ``Ca`` of the later half of the quasi-steady points before the step; NaN if fewer than four.

    The later half: the quasi-steady filter still admits a few points of the
    initial acceleration, and on a short pre-step stretch they bias the median low.
    """
    steady = (df["regime"].to_numpy() == BEFORE) & (np.abs(df["inertia_cm"].to_numpy()) < steady_tolerance(df))
    idx = np.nonzero(steady)[0]
    if idx.size < _MIN_PLATEAU_POINTS:
        return math.nan
    return float(np.median(df["Ca_cm"].to_numpy()[idx[idx.size // 2 :]]))


def calibrate(df: pd.DataFrame, info: RunInfo, global_beta: float, settings: AnalysisSettings) -> Calibration:
    """``beta_f`` from the run's pre-step plateau, else the global fit; ``w`` from its initial base.

    Only a run that moves has a plateau: a creeping or pinned run's tiny ``Ca``
    would make ``(Bo - H1)/Ca`` arbitrarily large.
    """
    plateau = pre_step_plateau(df)
    drive = info.bo - info.windows.h1
    width = float(df["w_si"].iloc[0])
    moving = classify_motion(df, info, settings) == "moving"
    if moving and math.isfinite(plateau) and plateau > settings.ca_min and drive > 0.0:
        return Calibration(drive / plateau, "pre-step plateau", width, plateau)
    return Calibration(global_beta, "global fit", width, plateau)


def _stepwise_si(info: RunInfo, width: float) -> tuple[Callable[[float], float], tuple[float, float]]:
    w = info.windows
    bounds = (-0.5 * width, 0.5 * width)
    hs = (w.h1, w.h2, w.h3)

    def h_of_x(x: float) -> float:
        return hs[int(np.searchsorted(bounds, x, side="right"))]

    return h_of_x, bounds


def _constant(value: float) -> Callable[[float], float]:
    def of_x(_x: float) -> float:
        return value

    return of_x


def run_ladder(
    geom: RunGeometry,
    cal: Calibration,
    fits_all: dict[str, FitResult],
    t_data: np.ndarray,
    settings: AnalysisSettings,
) -> dict[str, Trajectory]:
    """Every rung of the ladder for one run."""
    info = geom.info
    w = info.windows
    h_si, bounds = _stepwise_si(info, cal.width)
    t = time_grid(t_data)
    regimes = [Regime(w.h1, cal.beta_f), Regime(w.h2, cal.beta_f), Regime(w.h3, cal.beta_f)]
    x_si, ca_si = piecewise_trajectory(t, geom.x0, 0.0, info.oh, info.bo, bounds, regimes)
    c_step = fits_all["M4s"].params["c_lub"]
    return {
        "SI": trajectory_from(t, x_si, ca_si),
        "SI+beta(A)": integrate_with(geom, h_si, area_friction(geom, c_step), bounds, t_data, settings),
        "SI+H_meas": integrate_with(geom, measured_hysteresis(geom), _constant(cal.beta_f), (), t_data, settings),
        "M7": simulate(geom, MODEL_BY_KEY["M7"], param_vector(fits_all["M7"]), t_data, settings),
        "M4": simulate(geom, MODEL_BY_KEY["M4"], param_vector(fits_all["M4"]), t_data, settings),
        "M6": simulate(geom, MODEL_BY_KEY["M6"], param_vector(fits_all["M6"]), t_data, settings),
    }


def phase_errors(df: pd.DataFrame, run: str, trajectories: dict[str, Trajectory]) -> pd.DataFrame:
    """RMS error of ``Ca(X)`` per rung and phase, and over the whole run."""
    phases = phase_labels(df)
    data = df["Ca_cm"].to_numpy()
    rows = []
    for rung, traj in trajectories.items():
        pred, use = model_ca_at_data_x(df, traj)
        err = pred - data
        row: dict[str, object] = {"run": run, "rung": rung, "all": _rms(err[use])}
        for phase in PHASES:
            sel = use & (phases == phase)
            row[phase] = _rms(err[sel]) if sel.any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2))) if values.size else math.nan


def global_si_fit(points: pd.DataFrame, settings: AnalysisSettings) -> FitResult:
    """The SI's single ``beta_f`` fitted to every run, for the runs without a plateau."""
    return fit_model(
        points, SI_MODEL, "all", n_boot=settings.bootstrap_samples, rng=np.random.default_rng(settings.seed + 2)
    )
