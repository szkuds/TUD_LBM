"""Friction models and their fit to the per-point force balance.

Every model is linear in its parameters. With ``y = Bo - H - (1/Oh**2) dCa/dT``
the balance reads ``y = beta(t) * Ca``, and each model writes ``beta * Ca`` as
a design matrix times its parameters, so all of them are fitted by the same
ordinary least squares over the same moving points. Confidence intervals come
from resampling *runs* with replacement (percentile bootstrap), so a group of
five runs gets an interval that reflects five runs, not seven hundred points.

=====  ==============================================  =================  ========
Model  friction ``beta * Ca``                          hysteresis         params
=====  ==============================================  =================  ========
M0     ``beta * Ca``                                   stepwise (window)  1
M1     ``beta_i * Ca`` in regime ``i``                 stepwise           3
M2     ``(l_phob A_phob + l_phil A_phil) * Ca``         stepwise           2
M3     as M2                                           measured           2
M4     ``c * (A**2 / Omega) * Ca`` (lubrication)        measured           1
M5     ``c * (Ca_lead/theta_lead + Ca_trail/theta_trail)``  measured      1
M6     ``c * A_geom**2 * Ca``, geometry from the windows  closure            1
M7     ``(lambda/eta) * A * Ca`` (the SI's ``beta_f``)   measured           1
M0m    as M0                                           measured           1
M4s    as M4                                           stepwise           1
=====  ==============================================  =================  ========

M0 to M3 are the models of the SI and the comparison asked for; M4 and M5 are
physical friction laws; M6 is the fully predictive closure of
:mod:`.geometry`; M0m and M4s complete the 2x2 of friction law against
hysteresis input, so the effect of each can be read off separately. M7 is the
friction of the SI as written, ``F = lambda A v`` or ``beta_f = lambda A/eta``,
with one ``lambda`` for the whole run set and the measured base width ``A``: M3
with ``lambda_phob = lambda_phil``.

``l = lambda/eta`` carries units of 1/LU (``A`` in lattice units), so
``1/l`` is a friction length. M4 is ``beta = c * A/h`` with the mean height
``h = Omega/A``; M5 is de Gennes' wedge dissipation per contact line,
``3 ln(L/l_micro) * eta * v / theta``, written with the line speeds.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    import pandas as pd


def _ca(df: pd.DataFrame, proxy: str) -> np.ndarray:
    return df[f"Ca_{proxy}"].to_numpy()


def _basis_m0(df: pd.DataFrame, proxy: str) -> np.ndarray:
    return _ca(df, proxy)[:, None]


def _basis_m1(df: pd.DataFrame, proxy: str) -> np.ndarray:
    regime = df["regime"].to_numpy()
    ca = _ca(df, proxy)
    return np.stack([ca * (regime == i) for i in (1, 2, 3)], axis=1)


def _basis_area(df: pd.DataFrame, proxy: str) -> np.ndarray:
    ca = _ca(df, proxy)
    return np.stack([df["A_phob_lu"].to_numpy() * ca, df["A_phil_lu"].to_numpy() * ca], axis=1)


def _basis_contact_length(df: pd.DataFrame, proxy: str) -> np.ndarray:
    return (df["A_lu"].to_numpy() * _ca(df, proxy))[:, None]


def _basis_lubrication(df: pd.DataFrame, proxy: str) -> np.ndarray:
    return (df["A_lu"].to_numpy() ** 2 / df["Omega"].to_numpy() * _ca(df, proxy))[:, None]


def _basis_geometric(df: pd.DataFrame, proxy: str) -> np.ndarray:
    return (df["A_geom"].to_numpy() ** 2 * _ca(df, proxy))[:, None]


def _basis_wedge(df: pd.DataFrame, _proxy: str) -> np.ndarray:
    lead = df["Ca_lead"].to_numpy() / np.radians(df["theta_lead"].to_numpy())
    trail = df["Ca_trail"].to_numpy() / np.radians(df["theta_trail"].to_numpy())
    return (lead + trail)[:, None]


@dataclass(frozen=True)
class FrictionModel:
    """One friction model: its design matrix, its hysteresis column, its parameters."""

    key: str
    title: str
    hysteresis: str
    params: tuple[str, ...]
    basis: Callable[[pd.DataFrame, str], np.ndarray]


MODELS: tuple[FrictionModel, ...] = (
    FrictionModel("M0", "constant beta", "H_step", ("beta",), _basis_m0),
    FrictionModel("M1", "beta per regime", "H_step", ("beta_1", "beta_2", "beta_3"), _basis_m1),
    FrictionModel("M2", "area-weighted", "H_step", ("lambda_phob", "lambda_phil"), _basis_area),
    FrictionModel("M3", "area-weighted, measured H", "H_meas", ("lambda_phob", "lambda_phil"), _basis_area),
    FrictionModel("M4", "lubrication A^2/Omega, measured H", "H_meas", ("c_lub",), _basis_lubrication),
    FrictionModel("M5", "contact-line wedge, measured H", "H_meas", ("c_wedge",), _basis_wedge),
    FrictionModel("M6", "geometric closure (predictive)", "H_geom", ("c_geom",), _basis_geometric),
    FrictionModel("M7", "SI friction lambda A/eta, measured H", "H_meas", ("lambda_eta",), _basis_contact_length),
    FrictionModel("M0m", "constant beta, measured H", "H_meas", ("beta",), _basis_m0),
    FrictionModel("M4s", "lubrication A^2/Omega, stepwise H", "H_step", ("c_lub",), _basis_lubrication),
)
MODEL_BY_KEY = {m.key: m for m in MODELS}

#: Parameter units, for the tables.
PARAM_UNITS = {
    "beta": "-",
    "beta_1": "-",
    "beta_2": "-",
    "beta_3": "-",
    "lambda_phob": "1/LU",
    "lambda_phil": "1/LU",
    "c_lub": "-",
    "c_wedge": "-",
    "c_geom": "-",
    "lambda_eta": "1/LU",
}


def target(df: pd.DataFrame, model: FrictionModel, proxy: str = "cm") -> np.ndarray:
    """``y = Bo - H - (1/Oh**2) dCa/dT``: the force friction has to account for."""
    return df["Bo"].to_numpy() - df[model.hysteresis].to_numpy() - df[f"inertia_{proxy}"].to_numpy()


def friction(df: pd.DataFrame, model: FrictionModel, params: np.ndarray, proxy: str = "cm") -> np.ndarray:
    """``beta * Ca`` of *model* at every row."""
    return model.basis(df, proxy) @ np.nan_to_num(params)


def _lstsq(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares; a column with no data gets ``NaN`` rather than zero."""
    active = np.any(np.abs(design) > 0.0, axis=0)
    out = np.full(design.shape[1], np.nan)
    if active.any() and y.size:
        out[active] = np.linalg.lstsq(design[:, active], y, rcond=None)[0]
    return out


@dataclass(frozen=True)
class FitResult:
    """A fitted model: point estimate, bootstrap interval, and goodness of fit."""

    model: str
    scope: str
    proxy: str
    params: dict[str, float]
    ci_low: dict[str, float]
    ci_high: dict[str, float]
    n_points: int
    n_runs: int
    rss: float
    rms: float
    aic: float

    def as_dict(self) -> dict[str, object]:
        """JSON-ready form."""
        return {
            "model": self.model,
            "scope": self.scope,
            "proxy": self.proxy,
            "params": self.params,
            "ci95_low": self.ci_low,
            "ci95_high": self.ci_high,
            "n_points": self.n_points,
            "n_runs": self.n_runs,
            "rss": self.rss,
            "rms_residual": self.rms,
            "aic": self.aic,
        }


def fit_model(
    points: pd.DataFrame,
    model: FrictionModel,
    scope: str,
    *,
    proxy: str = "cm",
    n_boot: int,
    rng: np.random.Generator,
) -> FitResult:
    """Fit *model* to the moving rows of *points*, with a run-level bootstrap."""
    rows = points[points["moving"]]
    design = model.basis(rows, proxy)
    y = target(rows, model, proxy)
    good = np.all(np.isfinite(design), axis=1) & np.isfinite(y)
    design, y = design[good], y[good]
    runs = rows["run"].to_numpy()[good]
    theta = _lstsq(design, y)
    res = design @ np.nan_to_num(theta) - y
    n, k = y.size, int(np.sum(np.isfinite(theta)))
    rss = float(np.sum(res**2))

    run_ids = np.unique(runs)
    index = {r: np.nonzero(runs == r)[0] for r in run_ids}
    boot = np.full((n_boot, theta.size), np.nan)
    if run_ids.size > 1:
        for b in range(n_boot):
            pick = np.concatenate([index[r] for r in rng.choice(run_ids, size=run_ids.size, replace=True)])
            boot[b] = _lstsq(design[pick], y[pick])
    with np.errstate(all="ignore"):
        low = np.nanpercentile(boot, 2.5, axis=0) if run_ids.size > 1 else np.full(theta.size, np.nan)
        high = np.nanpercentile(boot, 97.5, axis=0) if run_ids.size > 1 else np.full(theta.size, np.nan)
    names = model.params
    return FitResult(
        model=model.key,
        scope=scope,
        proxy=proxy,
        params=dict(zip(names, theta.tolist(), strict=True)),
        ci_low=dict(zip(names, low.tolist(), strict=True)),
        ci_high=dict(zip(names, high.tolist(), strict=True)),
        n_points=n,
        n_runs=int(run_ids.size),
        rss=rss,
        rms=float(np.sqrt(rss / n)) if n else float("nan"),
        aic=float(n * np.log(rss / n) + 2 * k) if n and rss > 0 else float("nan"),
    )


def param_vector(fit: FitResult) -> np.ndarray:
    """The fitted parameters in the model's order."""
    return np.array([fit.params[p] for p in MODEL_BY_KEY[fit.model].params])
