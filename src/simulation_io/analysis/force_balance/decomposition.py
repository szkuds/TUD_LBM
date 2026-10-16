"""What the SI model misses, as forces, and in which phase of the step traversal.

The SI model's force balance (Sec. S4 of the SI) leaves a residual at every data
point. With the per-run friction ``beta_f`` of the SI calibration, the M4
friction ``beta_M4 = c A**2/Omega`` and the measured hysteresis ``H_meas``, the
residual splits exactly into four parts::

    F_res,SI = Oh^-2 dCa/dT - (Bo - H_SI - beta_f Ca)
             = dF_w + dF_H + dF_beta + R
    dF_w    = -(H_step - H_SI)               the constant base width: H switched at the wrong X
    dF_H    = -(H_meas - H_step)             the hysteresis dynamics (Gibbs hold, lead jump)
    dF_beta = -(beta_M4 - beta_f) * Ca        the contact-length friction it does not have
    R       = Oh^-2 dCa/dT - (Bo - H_meas - beta_M4 Ca)   what even M4 leaves

``H_step`` is the stepwise H at the *measured* contact-line positions, so ``dF_w``
is only the cost of the SI's constant ``w`` and ``dF_H`` only what the measured
angles add to a correctly timed step.

``R_M7`` is the same residual as ``R`` with the SI's own friction law,
``beta_M7 = (lambda/eta) A``, in place of M4's: what is left when the SI form is
given the measured A and H. It is reported beside the four terms, not in the sum.

Each term is a force per ``gamma`` (per unit depth): positive means the SI model
under-predicts the droplet's acceleration there.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from src.simulation_io.analysis.force_balance.points import AFTER
from src.simulation_io.analysis.force_balance.points import BEFORE
from src.simulation_io.analysis.force_balance.points import STRADDLING
from src.simulation_io.analysis.force_balance.points import TRAIL_CROSS_MARGIN_LU
from src.simulation_io.analysis.force_balance.simulate import steady_tolerance

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Phases of a step traversal, in the order the droplet meets them.
PHASES = ("pre", "lead jump", "straddle", "hold", "post")

#: The missing-force terms, in plotting order.
TERMS = ("dF_w", "dF_H", "dF_beta", "R")


def phase_labels(df: pd.DataFrame) -> np.ndarray:
    """Phase of every row of one run's point table.

    * ``pre``: both lines before the step (regime 1).
    * ``lead jump``: from the leading line reaching the step until the
      acceleration has died down to the run's noise level.
    * ``straddle``: the rest of regime 2 before the trailing line reaches the step.
    * ``hold``: from the trailing line reaching the step (within ``edge_width``)
      until it is past it; the Gibbs hold.
    * ``post``: both lines past the step (regime 3).
    """
    regime = df["regime"].to_numpy()
    trail = df["cll_trail"].to_numpy()
    reach = float(df["step_x"].iloc[0]) - float(df["edge_width"].iloc[0]) - TRAIL_CROSS_MARGIN_LU
    calm = np.abs(df["inertia_cm"].to_numpy()) < steady_tolerance(df)
    labels = np.full(regime.size, "post", dtype=object)
    labels[regime == BEFORE] = "pre"
    straddling = np.nonzero(regime == STRADDLING)[0]
    if straddling.size:
        arrived = straddling[trail[straddling] >= reach]
        first_hold = int(arrived[0]) if arrived.size else regime.size
        before_hold = straddling[straddling < first_hold]
        labels[straddling[straddling >= first_hold]] = "hold"
        labels[before_hold] = "straddle"
        settled = before_hold[calm[before_hold] & (before_hold > before_hold[0])] if before_hold.size else before_hold
        end_jump = int(settled[0]) if settled.size else first_hold
        labels[before_hold[before_hold < end_jump]] = "lead jump"
    labels[regime == AFTER] = "post"
    return labels


def missing_forces(points: pd.DataFrame, beta_f: Mapping[str, float], c_m4: float, lambda_m7: float) -> pd.DataFrame:
    """Per point: the SI residual and its four parts, the M7 residual, and the phase of the point."""
    beta_si = points["run"].map(beta_f).to_numpy(dtype=float)
    beta_m4 = c_m4 * points["A_lu"].to_numpy() ** 2 / points["Omega"].to_numpy()
    ca = points["Ca_cm"].to_numpy()
    inertia = points["inertia_cm"].to_numpy()
    bo = points["Bo"].to_numpy()
    h_si, h_meas = points["H_si"].to_numpy(), points["H_meas"].to_numpy()
    h_step = points["H_step"].to_numpy()
    out = points[["run", "name", "groups", "T", "X_cm", "Ca_cm", "Bo", "Oh", "R0"]].copy()
    phases = pd.Series(index=points.index, dtype=object)
    for _, df in points.groupby("run", sort=False):
        phases.loc[df.index] = phase_labels(df)
    out["phase"] = phases
    out["inertia"] = inertia
    out["gravity"] = bo
    out["hysteresis"] = -h_meas
    out["friction"] = -beta_m4 * ca
    out["F_res_SI"] = inertia - (bo - h_si - beta_si * ca)
    out["dF_w"] = -(h_step - h_si)
    out["dF_H"] = -(h_meas - h_step)
    out["dF_beta"] = -(beta_m4 - beta_si) * ca
    out["R"] = inertia - (bo - h_meas - beta_m4 * ca)
    out["R_M7"] = inertia - (bo - h_meas - lambda_m7 * points["A_lu"].to_numpy() * ca)
    return out


def by_phase(forces: pd.DataFrame) -> pd.DataFrame:
    """Per run and phase: number of points, RMS and mean of each missing-force term."""
    rows = []
    for run in forces["run"].unique():
        of_run = forces[forces["run"] == run]
        for phase in PHASES:
            sub = of_run[of_run["phase"] == phase]
            if not sub.empty:
                rows.append(_phase_row(sub, str(run), phase))
    return pd.DataFrame(rows)


def _phase_row(sub: pd.DataFrame, run: str, phase: str) -> dict[str, object]:
    row: dict[str, object] = {"run": run, "name": sub["name"].iloc[0], "phase": phase, "n": len(sub)}
    for term in (*TERMS, "R_M7", "F_res_SI"):
        values = sub[term].to_numpy()
        row[f"rms_{term}"] = float(np.sqrt(np.mean(values**2)))
        row[f"mean_{term}"] = float(np.mean(values))
    return row
