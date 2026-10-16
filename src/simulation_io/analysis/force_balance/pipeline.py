"""Run the whole force-balance analysis and write every table, figure and the report.

Output layout under the configured output directory::

    config.toml                  copy of the configuration used
    assumptions.log              every assumption and data problem, per run
    runs_inventory.csv           one row per analysed run
    points/<run>.csv             per-data-point table of one run
    points_all.csv               all of them
    fits.json                    every fit with its bootstrap interval
    model_comparison.csv         M0..M6 per group and overall
    model_run_metrics.csv        per run and model: RMSE, plateaus, peak, decay, stop
    friction_by_regime.csv       section D
    friction_dependence.csv      section D
    hysteresis.csv               section E
    smooth_limit.csv             section E
    overshoot.csv                section F
    common_heff.csv              section F
    collapse_points.csv          section G
    figures/*.pdf, figures/*.png
    report.md
"""

from __future__ import annotations
import json
import math
import re
import shutil
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from src.simulation_io.analysis.force_balance import analyses
from src.simulation_io.analysis.force_balance import decomposition
from src.simulation_io.analysis.force_balance import ladder
from src.simulation_io.analysis.force_balance import scaling
from src.simulation_io.analysis.force_balance.fits import MODELS
from src.simulation_io.analysis.force_balance.fits import FitResult
from src.simulation_io.analysis.force_balance.fits import fit_model
from src.simulation_io.analysis.force_balance.fits import param_vector
from src.simulation_io.analysis.force_balance.points import build_point_table
from src.simulation_io.analysis.force_balance.runs import AssumptionLog
from src.simulation_io.analysis.force_balance.runs import load_runs
from src.simulation_io.analysis.force_balance.simulate import RunGeometry
from src.simulation_io.analysis.force_balance.simulate import ca_of_x_error
from src.simulation_io.analysis.force_balance.simulate import data_metrics
from src.simulation_io.analysis.force_balance.simulate import model_metrics
from src.simulation_io.analysis.force_balance.simulate import simulate
from src.simulation_io.analysis.force_balance.simulate import steady_tolerance

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from src.simulation_io.analysis.droplet_metrics import DropletSeries
    from src.simulation_io.analysis.force_balance.config import ForceBalanceConfig
    from src.simulation_io.analysis.force_balance.fits import FrictionModel
    from src.simulation_io.analysis.force_balance.runs import RunInfo
    from src.simulation_io.analysis.force_balance.simulate import Trajectory

OVERALL = "all"

#: A group-mean effective hysteresis needs at least this many runs.
_MIN_RUNS_FOR_COMMON_H = 2

#: Velocity proxies compared in section B, for the models with measured H.
PROXY_MODELS = ("M3", "M4")

#: Metrics compared between model and data, per run.
COMPARED_METRICS = ("plateau_1", "plateau_2", "plateau_3", "peak", "decay", "x_stop")


def run_slug(run: str) -> str:
    """A file-name-safe form of a run path."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", run).strip("_")


@dataclass
class Results:
    """Everything the figures and the report are built from."""

    cfg: ForceBalanceConfig
    scopes: list[str]
    infos: dict[str, RunInfo]
    points: pd.DataFrame
    fits: dict[tuple[str, str, str], FitResult]
    run_metrics: pd.DataFrame
    comparison: pd.DataFrame
    trajectories: dict[tuple[str, str, str], Trajectory]
    friction: pd.DataFrame
    dependence: pd.DataFrame
    hysteresis: pd.DataFrame
    smooth_limit: pd.DataFrame
    overshoot: pd.DataFrame
    common_heff: pd.DataFrame
    collapse: pd.DataFrame
    log: AssumptionLog
    focus: Focus
    detailed: bool = False
    figures: list[str] = field(default_factory=list)

    def runs_in(self, scope: str) -> list[str]:
        """Runs of *scope*, in config order."""
        if scope == OVERALL:
            return list(self.infos)
        return [r for r, info in self.infos.items() if scope in info.groups]

    def fit(self, model: str, scope: str = OVERALL, proxy: str = "cm") -> FitResult:
        """The fit of *model* to *scope*."""
        return self.fits[(model, scope, proxy)]


@dataclass
class Focus:
    """The focused analysis: the model ladder, the missing forces and the beta scaling."""

    calibration: pd.DataFrame
    ladder: dict[tuple[str, str], Trajectory]
    ladder_errors: pd.DataFrame
    forces: pd.DataFrame
    forces_by_phase: pd.DataFrame
    geometry_laws: pd.DataFrame
    c_dependence: pd.DataFrame
    lambda_phases: pd.DataFrame
    notes: list[str]
    representatives: list[str]


def _representatives(cfg: ForceBalanceConfig, infos: dict[str, RunInfo], scopes: list[str]) -> list[str]:
    """The configured representative runs (a duplicate counts as its kept twin), else first and last per group."""
    alias = {dup: run for run, info in infos.items() for dup in info.duplicates}
    chosen = [alias.get(r, r) for r in cfg.representatives]
    if not chosen:
        for scope in scopes:
            members = [r for r, info in infos.items() if scope in info.groups]
            chosen += [members[0], members[-1]] if members else []
    return list(dict.fromkeys(r for r in chosen if r in infos))


def _focus(
    points: pd.DataFrame,
    infos: dict[str, RunInfo],
    fits_all: dict[str, FitResult],
    friction: pd.DataFrame,
    cfg: ForceBalanceConfig,
    scopes: list[str],
) -> Focus:
    s = cfg.settings
    global_beta = ladder.global_si_fit(points, s).params["beta"]
    calibrations, trajectories, errors = [], {}, []
    for run, info in infos.items():
        df = points[points["run"] == run]
        cal = ladder.calibrate(df, info, global_beta, s)
        calibrations.append({"run": run, "name": info.name, **asdict(cal)})
        geom = RunGeometry.from_points(df, info, s)
        for rung, traj in ladder.run_ladder(geom, cal, fits_all, df["T"].to_numpy(), s).items():
            trajectories[(run, rung)] = traj
        errors.append(ladder.phase_errors(df, run, {r: trajectories[(run, r)] for r in ladder.RUNGS}))
    calibration = pd.DataFrame(calibrations)
    beta_f = dict(zip(calibration["run"], calibration["beta_f"], strict=True))
    forces = decomposition.missing_forces(
        points, beta_f, fits_all["M4"].params["c_lub"], fits_all["M7"].params["lambda_eta"]
    )
    rng = np.random.default_rng(s.seed + 3)
    return Focus(
        calibration=calibration,
        ladder=trajectories,
        ladder_errors=pd.concat(errors, ignore_index=True),
        forces=forces,
        forces_by_phase=decomposition.by_phase(forces),
        geometry_laws=scaling.geometry_laws(friction, rng, s.bootstrap_samples),
        c_dependence=scaling.c_dependence(friction, rng, s.bootstrap_samples),
        lambda_phases=scaling.lambda_by_phase(points, infos),
        notes=scaling.identifiability(infos),
        representatives=_representatives(cfg, infos, [sc for sc in scopes if sc != OVERALL]),
    )


def _data_metrics(points: pd.DataFrame, infos: dict[str, RunInfo], cfg: ForceBalanceConfig) -> pd.DataFrame:
    rows = [
        {
            "run": run,
            "scope": "data",
            "model": "data",
            **data_metrics(points[points["run"] == run], info, cfg.settings.stop_ca),
        }
        for run, info in infos.items()
    ]
    return pd.DataFrame(rows)


def _log_settings(cfg: ForceBalanceConfig, log: AssumptionLog) -> None:
    s = cfg.settings
    log.add("global", "R0 = sqrt(Omega_D), Omega_D = inclusion area of the init field; X = (x - x_step)/R0")
    log.add("global", "eta = rho_l * (tau - 1/2)/3 (BGK, cs^2 = 1/3); rho_l = 1 in every run so eta = nu")
    log.add("global", "Bo uses the measured density contrast of the init field (the force the masked gravity injects)")
    log.add("global", "leading line = right contact line, trailing = left (gravity drives +x)")
    log.add(
        "global",
        "contact length A = cll_right - cll_left (base width at the first fluid row); no separate stored value exists",
    )
    log.add(
        "global",
        f"derivatives: Savitzky-Golay window {s.smoothing_window} samples, order {s.smoothing_polyorder}; "
        "noise sigma from the MAD of second differences, propagated through the central filter coefficients",
    )
    log.add(
        "global",
        f"fit points: Ca_cm > max({s.moving_sigma} sigma, {s.ca_min}); "
        f"CIs: {s.bootstrap_samples} run-level bootstrap resamples, seed {s.seed}",
    )
    log.add(
        "global",
        f"angle 'at bound' within {s.angle_tolerance_deg} deg; "
        f"line pinned below {s.line_pinned_sigma} sigma of its speed",
    )
    log.add("global", "line windows follow the solver: post-step window once a line is within edge_width of the step")
    log.add(
        "global",
        "regime 3 starts when the trailing line is 0.5 LU past the step (a held line sits within a cell of it)",
    )
    log.add(
        "global",
        "forward models start from rest at T = 0 (gravity onset = init snapshot timestep) "
        "at the first sample's smoothed X",
    )
    log.add(
        "global",
        "forward models M0-M5 read contact-line offsets, measured angles and line/centre speed ratios "
        "from the run as functions of X",
    )
    log.add("global", "M6 uses only the windows, Omega and X (circular cap, centre of mass at mid-base)")
    log.add(
        "global",
        "SI rung: closed form of SI Sec. S4 with w = measured initial base width and beta_f = (Bo - H1)/Ca_1 "
        "from the later half of the run's quasi-steady pre-step points; runs without one take the beta_f fitted "
        "to all runs. A short pre-step stretch has not fully reached Ca_inf, which biases beta_f slightly high",
    )
    log.add("global", "phases: pre | lead jump (until |inertia| falls to its noise) | straddle | hold | post")


def _fit_all(
    points: pd.DataFrame, scopes: list[str], infos: dict[str, RunInfo], cfg: ForceBalanceConfig
) -> dict[tuple[str, str, str], FitResult]:
    rng = np.random.default_rng(cfg.settings.seed)
    fits: dict[tuple[str, str, str], FitResult] = {}
    for scope in scopes:
        runs = list(infos) if scope == OVERALL else [r for r, i in infos.items() if scope in i.groups]
        sub = points[points["run"].isin(runs)]
        for model in MODELS:
            fits[(model.key, scope, "cm")] = fit_model(
                sub, model, scope, n_boot=cfg.settings.bootstrap_samples, rng=rng
            )
        if scope == OVERALL:
            for key in PROXY_MODELS:
                model = next(m for m in MODELS if m.key == key)
                for proxy in ("lead", "trail"):
                    fits[(key, scope, proxy)] = fit_model(
                        sub, model, scope, proxy=proxy, n_boot=cfg.settings.bootstrap_samples, rng=rng
                    )
    return fits


def _simulate_all(
    points: pd.DataFrame,
    scopes: list[str],
    infos: dict[str, RunInfo],
    fits: dict[tuple[str, str, str], FitResult],
    cfg: ForceBalanceConfig,
    progress: Callable[[str], None],
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], Trajectory]]:
    rows = []
    trajectories: dict[tuple[str, str, str], Trajectory] = {}
    s = cfg.settings
    for run, info in infos.items():
        df = points[points["run"] == run]
        geom = RunGeometry.from_points(df, info, s)
        tol = steady_tolerance(df)
        measured = data_metrics(df, info, s.stop_ca)
        rows.append({"run": run, "scope": "data", "model": "data", **measured})
        for scope in [OVERALL, *[g for g in info.groups if g in scopes]]:
            for model in MODELS:
                traj = simulate(geom, model, param_vector(fits[(model.key, scope, "cm")]), df["T"].to_numpy(), s)
                trajectories[(run, model.key, scope)] = traj
                rmse_x, rmse_t, coverage = ca_of_x_error(df, traj)
                rows.append(
                    {
                        "run": run,
                        "scope": scope,
                        "model": model.key,
                        "rmse_Ca_X": rmse_x,
                        "rmse_Ca_T": rmse_t,
                        "coverage": coverage,
                        **model_metrics(geom, traj, s.stop_ca, tol),
                    }
                )
        progress(f"simulated {info.name}")
    return pd.DataFrame(rows), trajectories


def _comparison_row(
    scope: str, model: FrictionModel, m: pd.DataFrame, data: pd.DataFrame, fit: FitResult
) -> dict[str, object]:
    """One model in one scope: parameters, fit quality and metric errors against *data*."""
    row: dict[str, object] = {
        "scope": scope,
        "model": model.key,
        "title": model.title,
        "n_runs": len(m),
        "n_params": len(model.params),
        "regression_rms": fit.rms,
        "aic": fit.aic,
        "rmse_Ca_X_median": float(m["rmse_Ca_X"].median()),
        "rmse_Ca_X_mean": float(m["rmse_Ca_X"].mean()),
        "rmse_Ca_T_median": float(m["rmse_Ca_T"].median()),
        "coverage_min": float(m["coverage"].min()),
    }
    for p in model.params:
        row[p] = fit.params[p]
        row[f"{p}_ci_low"] = fit.ci_low[p]
        row[f"{p}_ci_high"] = fit.ci_high[p]
    for metric in COMPARED_METRICS:
        diff = (m[metric] - data.loc[m.index, metric]).to_numpy(dtype=float)
        finite = np.isfinite(diff)
        row[f"{metric}_mae"] = float(np.mean(np.abs(diff[finite]))) if finite.any() else math.nan
        row[f"{metric}_n"] = int(finite.sum())
    stop_data = data.loc[m.index, "x_stop"].notna().to_numpy()
    row["stop_agreement"] = float(np.mean(stop_data == m["x_stop"].notna().to_numpy()))
    return row


def _comparison(
    run_metrics: pd.DataFrame, fits: dict[tuple[str, str, str], FitResult], scopes: list[str]
) -> pd.DataFrame:
    """M0..M6 per scope: fit quality, trajectory errors and metric errors against the data."""
    data = run_metrics[run_metrics["model"] == "data"].set_index("run")
    rows = []
    for scope in scopes:
        sub = run_metrics[(run_metrics["scope"] == scope) & (run_metrics["model"] != "data")]
        for model in MODELS:
            m = sub[sub["model"] == model.key].set_index("run")
            if not m.empty:
                rows.append(_comparison_row(scope, model, m, data, fits[(model.key, scope, "cm")]))
    return pd.DataFrame(rows)


def _inventory(infos: dict[str, RunInfo], hysteresis: pd.DataFrame, run_metrics: pd.DataFrame) -> pd.DataFrame:
    data = run_metrics[run_metrics["model"] == "data"].set_index("run")
    hyst = hysteresis.set_index("run")
    rows = []
    for run, info in infos.items():
        w = info.windows
        rows.append(
            {
                "run": run,
                "name": info.name,
                "groups": ";".join(info.groups),
                "duplicates": ";".join(info.duplicates),
                "R0": info.r0,
                "Omega_D": info.omega,
                "tau": info.tau,
                "eta": info.eta,
                "gamma": info.gamma,
                "gamma_source": info.gamma_source,
                "drho": info.drho,
                "g": info.g,
                "alpha_deg": info.alpha_deg,
                "Oh": info.oh,
                "Bo_par": info.bo,
                "Bo_par_rho_l": info.bo_rho_l,
                "theta_R_phob": w.receding_phob,
                "theta_A_phob": w.advancing_phob,
                "theta_R_phil": w.receding_phil,
                "theta_A_phil": w.advancing_phil,
                "H1": w.h1,
                "H2": w.h2,
                "H3": w.h3,
                "step_x_lu": info.step_x,
                "edge_width": info.edge_width,
                "save_interval": info.save_interval,
                "t_onset": info.t_onset,
                "starts_pinned": True,
                "observed": hyst.loc[run, "observed"],
                "predicted": hyst.loc[run, "predicted"],
                "x_stop": data.loc[run, "x_stop"],
            }
        )
    return pd.DataFrame(rows)


def compute(
    cfg: ForceBalanceConfig, groups: list[str] | None, progress: Callable[[str], None], *, detailed: bool = False
) -> Results:
    """Load, tabulate, fit, simulate and analyse; nothing is written."""
    log = AssumptionLog()
    _log_settings(cfg, log)
    loaded = load_runs(cfg, groups, log)
    if not loaded:
        msg = "no run could be loaded; see the assumption log"
        raise RuntimeError(msg)
    return analyse_runs(cfg, loaded, groups, log, progress, detailed=detailed)


def analyse_runs(
    cfg: ForceBalanceConfig,
    loaded: list[tuple[RunInfo, DropletSeries]],
    groups: list[str] | None,
    log: AssumptionLog,
    progress: Callable[[str], None],
    *,
    detailed: bool = False,
) -> Results:
    """Everything after loading: point tables, fits, the focused analysis, and (*detailed*) every model run."""
    infos = {info.run: info for info, _ in loaded}
    progress(f"loaded {len(infos)} unique runs")
    tables = [build_point_table(info, series, cfg.settings, log) for info, series in loaded]
    points = analyses.add_point_friction(pd.concat(tables, ignore_index=True))
    selected = groups if groups is not None else list(cfg.groups)
    scopes = [OVERALL, *[g for g in selected if any(g in i.groups for i in infos.values())]]
    fits = _fit_all(points, scopes, infos, cfg)
    progress("fitted M0..M6")
    if detailed:
        run_metrics, trajectories = _simulate_all(points, scopes, infos, fits, cfg, progress)
        comparison = _comparison(run_metrics, fits, scopes)
    else:
        run_metrics, trajectories, comparison = _data_metrics(points, infos, cfg), {}, pd.DataFrame()

    fits_all = {m.key: fits[(m.key, OVERALL, "cm")] for m in MODELS}
    rng = np.random.default_rng(cfg.settings.seed + 1)
    friction = analyses.friction_by_regime(points, infos)
    dependence = analyses.dependence_tests(friction, rng, cfg.settings.bootstrap_samples)
    hysteresis = analyses.hysteresis_table(points, infos, fits_all, cfg.settings)
    smooth = analyses.smooth_limit_test(points, fits_all["M4"], cfg.settings)
    overshoot = analyses.overshoot_table(points)
    common = []
    for scope in ("tau", "r"):
        runs = [r for r, i in infos.items() if scope in i.groups and (overshoot["run"] == r).any()]
        if len(runs) >= _MIN_RUNS_FOR_COMMON_H:
            part = analyses.common_hysteresis_test(points, infos, runs, fits_all["M4"], cfg.settings)
            part.insert(0, "group", scope)
            common.append(part)
    collapse = analyses.collapse_table(points, infos, fits_all["M1"], cfg.settings)
    progress("analyses D-G done")
    focus = _focus(points, infos, fits_all, friction, cfg, scopes)
    progress("model ladder, missing forces and beta scaling done")
    return Results(
        cfg=cfg,
        scopes=scopes,
        infos=infos,
        points=points,
        fits=fits,
        run_metrics=run_metrics,
        comparison=comparison,
        trajectories=trajectories,
        friction=friction,
        dependence=dependence,
        hysteresis=hysteresis,
        smooth_limit=smooth,
        overshoot=overshoot,
        common_heff=pd.concat(common, ignore_index=True) if common else pd.DataFrame(),
        collapse=collapse,
        log=log,
        focus=focus,
        detailed=detailed,
    )


def write_tables(res: Results, out: Path) -> None:
    """Every table of *res* under *out*."""
    (out / "points").mkdir(parents=True, exist_ok=True)
    for run, df in res.points.groupby("run", sort=False):
        df.to_csv(out / "points" / f"{run_slug(str(run))}.csv", index=False)
    res.points.to_csv(out / "points_all.csv", index=False)
    _inventory(res.infos, res.hysteresis, res.run_metrics).to_csv(out / "runs_inventory.csv", index=False)
    payload = [fit.as_dict() for fit in res.fits.values()]
    (out / "fits.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if res.detailed:
        res.comparison.to_csv(out / "model_comparison.csv", index=False)
    res.run_metrics.to_csv(out / "model_run_metrics.csv", index=False)
    res.friction.to_csv(out / "friction_by_regime.csv", index=False)
    res.dependence.to_csv(out / "friction_dependence.csv", index=False)
    res.hysteresis.to_csv(out / "hysteresis.csv", index=False)
    res.smooth_limit.to_csv(out / "smooth_limit.csv", index=False)
    res.overshoot.to_csv(out / "overshoot.csv", index=False)
    res.common_heff.to_csv(out / "common_heff.csv", index=False)
    res.collapse.to_csv(out / "collapse_points.csv", index=False)
    f = res.focus
    f.calibration.to_csv(out / "si_calibration.csv", index=False)
    f.ladder_errors.to_csv(out / "ladder_errors.csv", index=False)
    f.forces.to_csv(out / "missing_forces.csv", index=False)
    f.forces_by_phase.to_csv(out / "missing_by_phase.csv", index=False)
    f.geometry_laws.to_csv(out / "beta_geometry_laws.csv", index=False)
    f.c_dependence.to_csv(out / "beta_c_dependence.csv", index=False)
    f.lambda_phases.to_csv(out / "lambda_by_phase.csv", index=False)


def run_force_balance(
    cfg: ForceBalanceConfig, groups: list[str] | None, progress: Callable[[str], None], *, detailed: bool = False
) -> Path:
    """The CLI entry point: compute, write tables, figures and the report; return the report path."""
    from src.simulation_io.analysis.force_balance.plots import make_figures
    from src.simulation_io.analysis.force_balance.plots_focus import make_focus_figures
    from src.simulation_io.analysis.force_balance.report import write_report

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    res = compute(cfg, groups, progress, detailed=detailed)
    write_tables(res, out)
    progress("tables written")
    res.figures = make_focus_figures(res, out / "figures", progress)
    if detailed:
        res.figures += make_figures(res, out / "figures" / "detailed", progress)
    if cfg.source is not None:
        shutil.copyfile(cfg.source, out / "config.toml")
    (out / "assumptions.log").write_text("\n".join(res.log.lines()) + "\n", encoding="utf-8")
    return write_report(res, out)
