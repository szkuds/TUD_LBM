"""What the friction coefficient beta depends on, and how it scales.

Input is the per-run, per-regime friction ``beta_eff`` of
:func:`~.analyses.friction_by_regime` (median over quasi-steady moving points).
Three questions:

1. **Geometry.** A free power law ``beta = k (A**2/Omega)**p`` over all runs and
   regimes. ``p = 1`` is the lubrication form ``beta = c A**2/Omega``; for a
   circular cap ``A**2/Omega`` is a function of the mean contact angle alone, so
   ``beta`` follows ``theta_mean``.
2. **What is left.** The dependence of ``c = beta Omega/A**2`` on Oh, R0, the
   interface width over the size ``W/R0``, Bo, Ca and the mean angle, as log-log
   exponents with bootstrap intervals over runs.
3. **Identifiability.** A parameter that is the same in every run cannot be
   separated from the others it is built with; such pairs are reported.

The SI writes the friction as ``F = lambda A v``, ``beta_f = lambda A/eta``
(model M7). :func:`lambda_by_phase` follows its coefficient along the step
traversal: if the SI form held, ``lambda/eta = beta_eff/A`` would be the same in
every phase.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from src.simulation_io.analysis.force_balance.analyses import dependence_tests
from src.simulation_io.analysis.force_balance.decomposition import PHASES
from src.simulation_io.analysis.force_balance.decomposition import phase_labels

if TYPE_CHECKING:
    from src.simulation_io.analysis.force_balance.runs import RunInfo

#: Parameters the friction constant ``c`` is tested against.
C_PARAMETERS = ("Oh", "R0", "W_over_R0", "Bo", "Ca", "theta_mean")

#: A power law needs this many distinct regressor values.
_MIN_DISTINCT_VALUES = 3


def power_law(
    x: np.ndarray, y: np.ndarray, runs: np.ndarray, rng: np.random.Generator, n_boot: int
) -> dict[str, float]:
    """``y = k x**p`` by least squares in log space; 95% intervals by resampling runs."""
    lx, ly = np.log(x), np.log(y)
    p, lk = np.polyfit(lx, ly, 1)
    ids = np.unique(runs)
    index = {r: np.nonzero(runs == r)[0] for r in ids}
    boot = np.full((n_boot, 2), np.nan)
    for b in range(n_boot):
        pick = np.concatenate([index[r] for r in rng.choice(ids, size=ids.size, replace=True)])
        if np.ptp(lx[pick]) > 0:
            boot[b] = np.polyfit(lx[pick], ly[pick], 1)
    res = ly - (p * lx + lk)
    return {
        "exponent": float(p),
        "exponent_ci_low": float(np.nanpercentile(boot[:, 0], 2.5)),
        "exponent_ci_high": float(np.nanpercentile(boot[:, 0], 97.5)),
        "prefactor": float(math.exp(lk)),
        "prefactor_ci_low": float(math.exp(np.nanpercentile(boot[:, 1], 2.5))),
        "prefactor_ci_high": float(math.exp(np.nanpercentile(boot[:, 1], 97.5))),
        "rms_log_residual": float(np.sqrt(np.mean(res**2))),
        "n_points": int(x.size),
        "n_runs": int(ids.size),
    }


def geometry_laws(friction: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    """``beta`` against ``A**2/Omega`` and against ``A/R0``, over all regimes and per regime."""
    rows = []
    for regime, sub in [("all", friction), *friction.groupby("regime")]:
        runs = sub["run"].to_numpy()
        for regressor in ("A2_over_Omega", "A_over_R0"):
            x = sub[regressor].to_numpy(dtype=float)
            if np.unique(x).size < _MIN_DISTINCT_VALUES:
                continue
            fit = power_law(x, sub["beta"].to_numpy(dtype=float), runs, rng, n_boot)
            rows.append({"law": f"beta ~ ({regressor})^p", "regime": regime, **fit})
    return pd.DataFrame(rows)


def c_dependence(friction: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    """Log-log exponents of ``c = beta Omega/A**2`` against each parameter, per regime."""
    return dependence_tests(friction, rng, n_boot, quantities=("c_lub",), parameters=C_PARAMETERS)


def identifiability(infos: dict[str, RunInfo]) -> list[str]:
    """Statements about parameters that the run set cannot separate."""
    notes = []
    widths = {info.interface_width for info in infos.values()}
    if len(widths) == 1:
        (w,) = widths
        notes.append(
            f"The interface width is W = {w:g} LU in every run, so W/R0 = {w:g}/R0: a dependence on the "
            "droplet size and on its numerical resolution cannot be told apart with these runs."
        )
    return notes


def lambda_by_phase(points: pd.DataFrame, infos: dict[str, RunInfo]) -> pd.DataFrame:
    """Per run and phase: the median SI friction coefficient ``lambda/eta = beta_eff/A``.

    Every moving point with a defined ``beta_eff`` enters, transients included,
    because the question is how the coefficient changes along the traversal.
    ``ratio_to_pre`` is the phase median over the run's own pre-step median.
    """
    rows = []
    for run, df in points.groupby("run", sort=False):
        info = infos[str(run)]
        phases = phase_labels(df.reset_index(drop=True))
        values = df["lambda_eta_eff"].to_numpy()
        pre = values[(phases == "pre") & np.isfinite(values)]
        reference = float(np.median(pre)) if pre.size else math.nan
        for phase in PHASES:
            sel = values[(phases == phase) & np.isfinite(values)]
            if sel.size == 0:
                continue
            median = float(np.median(sel))
            rows.append(
                {
                    "run": run,
                    "name": info.name,
                    "groups": ";".join(info.groups),
                    "phase": phase,
                    "n": int(sel.size),
                    "lambda_eta": median,
                    "lambda_eta_iqr": float(np.percentile(sel, 75) - np.percentile(sel, 25)),
                    "lambda": median * info.eta,
                    "lambda_eta_R0": median * info.r0,
                    "ratio_to_pre": median / reference if reference > 0.0 else math.nan,
                    "Oh": info.oh,
                    "Bo": info.bo,
                    "R0": info.r0,
                    "eta": info.eta,
                }
            )
    return pd.DataFrame(rows)
