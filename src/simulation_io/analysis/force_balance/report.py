"""``report.md``: the numbers behind every finding, generated from the results.

Everything quoted here is computed; the interpretation that goes with it is
written separately so that re-running the analysis never overwrites it.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from src.simulation_io.analysis.force_balance.decomposition import PHASES
from src.simulation_io.analysis.force_balance.decomposition import TERMS
from src.simulation_io.analysis.force_balance.fits import MODELS
from src.simulation_io.analysis.force_balance.fits import PARAM_UNITS
from src.simulation_io.analysis.force_balance.ladder import RUNGS
from src.simulation_io.analysis.force_balance.pipeline import OVERALL

if TYPE_CHECKING:
    from pathlib import Path
    from src.simulation_io.analysis.force_balance.pipeline import Results

#: Column heading of an exponent with its 95% interval.
_EXPONENT_CI = "exponent [95% CI]"

#: A run is "not described" when its best trajectory RMSE exceeds this.
POOR_RMSE = 0.02


def _fmt(v: float, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f"{v:.{digits}g}"


def _interval(res: Results, model: str, param: str, scope: str = OVERALL) -> str:
    f = res.fit(model, scope)
    return f"{_fmt(f.params[param], 4)} [{_fmt(f.ci_low[param], 3)}, {_fmt(f.ci_high[param], 3)}]"


def _with_ci(df: pd.DataFrame, value: str) -> list[str]:
    """``value [low, high]`` strings from the columns *value*, *value*_ci_low and *value*_ci_high."""
    return [
        f"{_fmt(v)} [{_fmt(lo)}, {_fmt(hi)}]"
        for v, lo, hi in zip(df[value], df[f"{value}_ci_low"], df[f"{value}_ci_high"], strict=True)
    ]


def _table(df: pd.DataFrame, digits: int = 3) -> str:
    """A markdown table of *df* without a pandas extra dependency."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(_fmt(float(v), digits) if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _comparison_section(res: Results) -> list[str]:
    intro = (
        "All models are fitted by the same least squares on the per-point balance "
        "`Bo - H - Oh^-2 dCa/dT = beta Ca` over moving points, intervals are 95% percentile "
        f"bootstraps over runs ({res.cfg.settings.bootstrap_samples} resamples). Trajectory errors come "
        "from integrating each fitted model forward from rest and reading Ca at the measured X. "
        "`*_mae` columns are mean absolute errors against the same metric of the data."
    )
    lines = ["## C. Friction models compared", "", intro, "", "### Parameters (all runs)", ""]
    rows = [
        {
            "model": m.key,
            "description": m.title,
            "parameter": p,
            "units": PARAM_UNITS[p],
            "value [95% CI]": _interval(res, m.key, p),
        }
        for m in MODELS
        for p in m.params
    ]
    lines += [_table(pd.DataFrame(rows)), ""]
    cols = [
        "model",
        "n_runs",
        "n_params",
        "regression_rms",
        "aic",
        "rmse_Ca_X_median",
        "rmse_Ca_X_mean",
        "plateau_1_mae",
        "plateau_2_mae",
        "plateau_3_mae",
        "peak_mae",
        "decay_mae",
        "x_stop_mae",
        "stop_agreement",
    ]
    for scope in res.scopes:
        sub = res.comparison[res.comparison["scope"] == scope]
        if sub.empty:
            continue
        lines += [f"### Scope `{scope}`", "", _table(sub[cols]), ""]
    return lines


def _finding_best(rmse: pd.Series) -> str:
    best = str(rmse.idxmin())
    m0 = float(rmse["M0"])
    title = next(m.title for m in MODELS if m.key == best)
    others = ", ".join(f"{k} {_fmt(rmse[k])}" for k in ("M1", "M2", "M3", "M4", "M6", "M7"))
    return (
        f"1. Best trajectory model over all runs: **{best}** ({title}), median RMSE of Ca(X) "
        f"{_fmt(rmse[best])} against {_fmt(m0)} for M0 ({_fmt(m0 / rmse[best], 2)}x smaller); {others}."
    )


def _finding_area(res: Results) -> str:
    return (
        f"2. Area-weighted friction (M3, all runs): lambda_phob/eta = {_interval(res, 'M3', 'lambda_phob')} 1/LU, "
        f"lambda_phil/eta = {_interval(res, 'M3', 'lambda_phil')} 1/LU; with the stepwise H (M2): "
        f"{_interval(res, 'M2', 'lambda_phob')} and {_interval(res, 'M2', 'lambda_phil')} 1/LU."
    )


def _finding_lubrication(res: Results) -> str:
    per_group = ", ".join(f"{s} {_interval(res, 'M4', 'c_lub', s)}" for s in res.scopes if s != OVERALL)
    return (
        f"3. Lubrication friction beta = c A^2/Omega: c = {_interval(res, 'M4', 'c_lub')} with the measured H (M4), "
        f"{_interval(res, 'M4s', 'c_lub')} with the stepwise H (M4s), and {_interval(res, 'M6', 'c_geom')} in the "
        f"fully predictive closure M6. Per group (M4): {per_group}."
    )


def _finding_ingredients(rmse: pd.Series) -> str:
    m0 = _fmt(float(rmse["M0"]))
    return (
        "4. Effect of each ingredient on the median trajectory RMSE: friction law alone "
        f"(M0 -> M4s) {m0} -> {_fmt(rmse['M4s'])}; measured hysteresis alone (M0 -> M0m) "
        f"{m0} -> {_fmt(rmse['M0m'])}; both (M4) {_fmt(rmse['M4'])}."
    )


def _finding_proxy(res: Results) -> str:
    prox = ", ".join(f"{p} {_fmt(res.fit('M4', OVERALL, p).rms)}" for p in ("cm", "lead", "trail"))
    return f"5. Which velocity satisfies the balance (M4 regression residual RMS): {prox}."


def _finding_regimes(res: Results) -> str:
    m1 = res.fit("M1")
    betas = ", ".join(f"beta_{i} = {_interval(res, 'M1', f'beta_{i}')}" for i in (1, 2, 3))
    ratio = _fmt(m1.params["beta_3"] / m1.params["beta_1"])
    return f"6. Per-regime friction (M1): {betas}; ratio beta_3/beta_1 = {ratio}."


def _finding_hold(held: pd.DataFrame) -> str:
    med = held.median(numeric_only=True)
    return (
        f"7. Trailing line held at the step in {len(held)} runs: median hold length "
        f"{_fmt(med['hold_dX'])} R0 (closure predicts {_fmt(med['hold_dX_closure'])} R0), "
        f"median {_fmt(med['hold_dt_lattice'])} lattice steps; trailing angle "
        f"{_fmt(med['hold_theta_trail_start'])} -> {_fmt(med['hold_theta_trail_end'])} deg; "
        f"base width {_fmt(med['hold_A_start'])} -> {_fmt(med['hold_A_end'])} R0 "
        f"(closure {_fmt(med['A2_closure'])} -> {_fmt(med['A3_closure'])})."
    )


def _finding_onset(h: pd.DataFrame) -> str:
    wrong = h[~h["agrees"]]
    listed = ", ".join(
        f"{name} (Bo - H_start = {bo - h:+.3f}, {obs})"
        for name, bo, h, obs in zip(wrong["name"], wrong["Bo"], wrong["H_start"], wrong["observed"], strict=True)
    )
    return (
        "8. Onset Bo > H_start (H1, or H2 for a droplet released on the step) agrees with the observed motion in "
        f"{int(h['agrees'].sum())}/{len(h)} runs ({_fmt(100 * h['agrees'].mean(), 3)}%)."
        + (f" Disagreeing: {listed}." if len(wrong) else "")
    )


def _findings(res: Results) -> list[str]:
    rmse = res.comparison[res.comparison["scope"] == OVERALL].set_index("model")["rmse_Ca_X_median"]
    h = res.hysteresis
    held = h.dropna(subset=["hold_dX"]) if "hold_dX" in h else pd.DataFrame()
    items = [
        _finding_best(rmse),
        _finding_area(res),
        _finding_lubrication(res),
        _finding_ingredients(rmse),
        _finding_proxy(res),
        _finding_regimes(res),
        *([_finding_hold(held)] if not held.empty else []),
        _finding_onset(h),
    ]
    return ["## Findings (numbers)", "", *items]


def _dependence_section(res: Results) -> list[str]:
    intro = (
        "Per run and regime, the median of `beta_eff = (Bo - H_meas - Oh^-2 dCa/dT)/Ca` over quasi-steady "
        "moving points (`friction_by_regime.csv`). Below: Spearman correlation of `lambda/eta = beta/A` and of "
        "`c = beta Omega/A^2` with each run parameter, the log-log exponent with a bootstrap 95% interval, and "
        "the coefficient of variation over runs."
    )
    lines = ["## D. Does the friction per contact length depend on anything else?", "", intro, ""]
    dep = res.dependence
    if dep.empty:
        return [*lines, "No regime had enough runs.", ""]
    show = dep[dep["quantity"].isin(["lambda_per_lu", "c_lub"])].copy()
    show[_EXPONENT_CI] = _with_ci(show, "exponent")
    lines += [
        _table(show[["quantity", "regime", "parameter", "n_runs", "spearman_rho", "p_value", _EXPONENT_CI, "cv"]]),
        "",
    ]
    fr = res.friction
    summary = fr.groupby("regime").agg(
        n=("beta", "size"),
        beta=("beta", "median"),
        A_over_R0=("A_over_R0", "median"),
        lambda_per_lu=("lambda_per_lu", "median"),
        c_lub=("c_lub", "median"),
        theta_mean=("theta_mean", "median"),
    )
    lines += ["Medians over runs, per regime:", "", _table(summary.reset_index()), ""]
    return lines


def _hysteresis_section(res: Results) -> list[str]:
    h = res.hysteresis
    lines = ["## E. Hysteresis", ""]
    cols = [
        c
        for c in (
            "name",
            "observed",
            "predicted",
            "Bo",
            "H_start",
            "T_bounds_reached",
            "hold_dX",
            "hold_dX_closure",
            "hold_dt_lattice",
            "hold_theta_trail_start",
            "hold_theta_trail_end",
            "trail_ell",
            "trail_ell_ci",
            "trail_ell_resolved",
            "lead_ell",
            "lead_ell_ci",
            "lead_ell_resolved",
            "Lambda_3_M1",
            "trail_ell_over_Lambda3",
        )
        if c in h
    ]
    lines += [_table(h[cols]), ""]
    if not res.smooth_limit.empty:
        sl = res.smooth_limit
        summary = (
            "Smooth-variation limit over the hold window: median RMSE of the quasi-steady Ca_inf(X) "
            f"{_fmt(sl['rmse_quasi_steady'].median())}, with the -Lambda dCa_inf/dX correction "
            f"{_fmt(sl['rmse_smooth_limit'].median())} (median Lambda {_fmt(sl['Lambda_median'].median())} R0)."
        )
        lines += [summary, ""]
    return lines


def _overshoot_section(res: Results) -> list[str]:
    lines = ["## F. Overshoot", ""]
    if res.overshoot.empty:
        return [*lines, "No run had a measurable overshoot.", ""]
    ov = res.overshoot.merge(res.hysteresis[["run", "name", "Oh", "R0"]], on="run")
    for scope in ("tau", "r"):
        sub = ov[ov["run"].isin(res.runs_in(scope))].sort_values("Oh" if scope == "tau" else "R0")
        if not sub.empty:
            lines += [
                f"Group `{scope}`:",
                "",
                _table(sub[["name", "Oh", "R0", "peak", "settled", "overshoot", "A_at_peak", "A_settled"]]),
                "",
            ]
    if not res.common_heff.empty:
        lines += [
            "Forward integration with the group-mean H_eff(X) and each run's own Oh, Bo and A(X) (M4 friction):",
            "",
            _table(res.common_heff),
            "",
        ]
    return lines


def _poor_runs(res: Results) -> list[str]:
    m = res.run_metrics[(res.run_metrics["scope"] == OVERALL) & (res.run_metrics["model"] != "data")]
    best = m.groupby("run")["rmse_Ca_X"].min()
    poor = best[best > POOR_RMSE].sort_values(ascending=False)
    lines = [f"Runs whose best model (over M0..M6, all-run fits) has RMSE of Ca(X) above {POOR_RMSE}:", ""]
    if poor.empty:
        return [*lines, "none.", ""]
    best_model = m.sort_values("rmse_Ca_X").groupby("run")["model"].first()
    for run, v in poor.items():
        info = res.infos[str(run)]
        lines += [f"- {info.name} ({';'.join(info.groups)}): {_fmt(float(v))} ({best_model[run]})"]
    return [*lines, ""]


def write_detailed_report(res: Results, out: Path) -> Path:
    """Write ``report_detailed.md`` (every model, every table) under *out* and return its path."""
    n_runs = len(res.infos)
    lines = [
        "# Force-balance analysis of the chemical-step droplet runs",
        "",
        (
            f"{n_runs} unique runs, groups {', '.join(s for s in res.scopes if s != OVERALL)}. "
            "Generated by `tud-lbm force-balance --detailed`: every model and every table. "
            "The focused questions are answered in `report.md`."
        ),
        "",
        (
            "Model: `(1/Oh^2) dCa/dT = Bo - H - beta Ca`, `dX/dT = Ca`, with `X = (x - x_step)/R0`, "
            "`R0 = sqrt(Omega_D)`, `Ca = eta v/gamma`, `T = t gamma/(eta R0)`."
        ),
        "",
    ]
    lines += _findings(res)
    lines += [""]
    lines += _comparison_section(res)
    lines += _dependence_section(res)
    lines += _hysteresis_section(res)
    lines += _overshoot_section(res)
    lines += ["## What the models do not describe", ""]
    lines += _poor_runs(res)
    lines += ["## Assumptions and data problems", "", "Full list in `assumptions.log`:", ""]
    lines += [f"- {entry}" for entry in res.log.lines()]
    lines += ["", "## Figures", ""]
    lines += [
        f"- `figures/detailed/{name}.pdf` (`.png`)"
        for name in res.figures
        if name[:2] in ("02", "03", "04", "05", "06", "07", "08")
    ]
    lines += [f"- `figures/detailed/force_balance/` — one decomposition per run ({n_runs})"]
    path = out / "report_detailed.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ladder_section(res: Results) -> list[str]:
    f = res.focus
    cal = f.calibration
    from_plateau = int((cal["beta_source"] == "pre-step plateau").sum())
    errors = f.ladder_errors
    columns = ["all", *PHASES]
    med = errors.groupby("rung")[columns].median().reindex(list(RUNGS)).reset_index()
    si_all = float(med.set_index("rung").loc["SI", "all"])
    m4_all = float(med.set_index("rung").loc["M4", "all"])
    return [
        "## 1. The SI model against the data-filled model",
        "",
        (
            "Each rung of the ladder replaces one SI assumption with the run's own data, all integrated from rest "
            "at the run's start. The SI rung is the SI's closed form (Sec. S4) with `w` the measured initial base "
            "width and `beta_f = (Bo - H1)/Ca_1` from the run's own pre-step plateau "
            f"{_fallback_note(cal, from_plateau)}."
        ),
        "",
        "Median over runs of the RMS error of Ca(X), overall and per phase (`ladder_errors.csv`):",
        "",
        _table(med),
        "",
        (
            f"Filling in both ingredients from the data (M4) lowers the median error {_fmt(si_all / m4_all, 2)}x, "
            f"from {_fmt(si_all)} to {_fmt(m4_all)}. Per-run SI calibration: `si_calibration.csv`."
        ),
        "",
    ]


def _fallback_note(cal: pd.DataFrame, from_plateau: int) -> str:
    """Parenthetical on how many runs took ``beta_f`` from their plateau and what the rest used."""
    fallback = cal.loc[cal["beta_source"] != "pre-step plateau", "beta_f"]
    if fallback.empty:
        return f"(all {from_plateau} runs)"
    return (
        f"({from_plateau} runs; the other {fallback.size}, not moving (pinned or creeping) or released on the "
        f"step, use the value fitted to all runs, {_fmt(float(fallback.median()))})"
    )


def _missing_section(res: Results) -> list[str]:
    table = res.focus.forces_by_phase
    rms_cols = [f"rms_{t}" for t in TERMS]
    med = table.groupby("phase")[[*rms_cols, "rms_R_M7", "rms_F_res_SI"]].median().reindex(list(PHASES)).reset_index()
    dominant = med.set_index("phase")[rms_cols].idxmax(axis=1).str.removeprefix("rms_")
    return [
        "## 2. What the SI model misses, and when",
        "",
        (
            "The SI model's residual splits exactly into the cost of its constant base width "
            "(`dF_w = -(H_step - H_SI)`, H switched at the wrong X), the hysteresis dynamics it lacks "
            "(`dF_H = -(H_meas - H_step)`: Gibbs hold, lead jump), the contact-length friction it lacks "
            "(`dF_beta = -(beta_M4 - beta_f) Ca`) and what even M4 leaves (`R`); `R_M7` is what is left with the SI's "
            "own friction `(lambda/eta) A` instead of M4's (not part of the sum). "
            "Median over runs of the RMS of each, per phase (`missing_by_phase.csv`, per point `missing_forces.csv`):"
        ),
        "",
        _table(med),
        "",
        "Largest missing force per phase: "
        + ", ".join(f"{phase}: {term}" for phase, term in dominant.items() if isinstance(term, str))
        + ".",
        "",
    ]


def _scaling_section(res: Results) -> list[str]:
    f = res.focus
    if f.geometry_laws.empty or f.c_dependence.empty:
        return [
            "## 3. What beta depends on",
            "",
            "Too few runs with distinct parameter values for a scaling fit.",
            "",
            *[f"Note: {note}" for note in f.notes],
            "",
        ]
    laws = f.geometry_laws.copy()
    laws[_EXPONENT_CI] = _with_ci(laws, "exponent")
    laws["prefactor [95% CI]"] = _with_ci(laws, "prefactor")
    dep = f.c_dependence.copy()
    dep[_EXPONENT_CI] = _with_ci(dep, "exponent")
    medians = res.friction.groupby("regime")[["beta", "A_over_R0", "A2_over_Omega", "c_lub", "theta_mean"]].median()
    return [
        "## 3. What beta depends on",
        "",
        (
            "Per run and regime, `beta_eff = (Bo - H_meas - Oh^-2 dCa/dT)/Ca` over quasi-steady moving points. "
            "Free power laws, with 95% intervals from resampling runs (`beta_geometry_laws.csv`):"
        ),
        "",
        _table(laws[["law", "regime", "n_runs", _EXPONENT_CI, "prefactor [95% CI]", "rms_log_residual"]]),
        "",
        "Medians per regime:",
        "",
        _table(medians.reset_index()),
        "",
        (
            "What `c = beta Omega/A^2` still depends on: log-log exponent per regime, Spearman correlation over runs "
            "(`beta_c_dependence.csv`):"
        ),
        "",
        _table(dep[["regime", "parameter", "n_runs", "spearman_rho", "p_value", _EXPONENT_CI, "cv"]]),
        "",
        *[f"Note: {note}" for note in f.notes],
        "",
    ]


def _lambda_section(res: Results) -> list[str]:
    table = res.focus.lambda_phases
    m7, m4 = res.fit("M7"), res.fit("M4")
    lam = m7.params["lambda_eta"]
    per_phase = (
        table.groupby("phase")
        .agg(
            n_runs=("run", "size"),
            lambda_eta=("lambda_eta", "median"),
            lambda_eta_R0=("lambda_eta_R0", "median"),
            ratio_to_pre=("ratio_to_pre", "median"),
            ratio_q1=("ratio_to_pre", lambda v: v.quantile(0.25)),
            ratio_q3=("ratio_to_pre", lambda v: v.quantile(0.75)),
        )
        .reindex(list(PHASES))
        .reset_index()
    )
    return [
        "## 4. The SI friction coefficient lambda (M7)",
        "",
        (
            "The SI writes the friction as `F = lambda A v`, i.e. `beta_f = lambda A/eta` with `A` the base width. "
            f"Fitted with the measured H and A over all runs (M7): lambda/eta = {_interval(res, 'M7', 'lambda_eta')} "
            f"1/LU, a friction length eta/lambda = {_fmt(1 / lam)} LU. Regression residual RMS {_fmt(m7.rms)} against "
            f"{_fmt(m4.rms)} for M4 (`c A^2/Omega`); the trajectory errors are in the table of section 1."
        ),
        "",
        (
            "Per point, `lambda/eta = beta_eff/A`. Median over runs of each run's phase median, its product with R0 "
            "(dimensionless), and the ratio to the run's own pre-step value with quartiles over runs "
            "(`lambda_by_phase.csv`; in lattice units `lambda = (lambda/eta) eta`, column `lambda`):"
        ),
        "",
        _table(per_phase),
        "",
    ]


def write_report(res: Results, out: Path) -> Path:
    """Write the focused ``report.md`` under *out* (and ``report_detailed.md`` when detailed); return its path."""
    if res.detailed:
        write_detailed_report(res, out)
    lines = [
        "# Force balance of the chemical-step droplets: what the SI model misses",
        "",
        (
            f"{len(res.infos)} unique runs. Generated by `tud-lbm force-balance`; every number is recomputed on each "
            "run. `X = (x - x_step)/R0`, `R0 = sqrt(Omega_D)`; model `(1/Oh^2) dCa/dT = Bo - H - beta Ca`."
        ),
        "",
    ]
    lines += _ladder_section(res)
    lines += _missing_section(res)
    lines += _scaling_section(res)
    lines += _lambda_section(res)
    lines += ["## Assumptions and data problems", "", "Full list in `assumptions.log`:", ""]
    lines += [f"- {entry}" for entry in res.log.lines()]
    lines += ["", "## Figures", ""]
    lines += [
        f"- `figures/{name}.pdf` (`.png`)"
        for name in res.figures
        if name.startswith("F") and "decomposition" not in name
    ]
    lines += [f"- `figures/F2_decomposition/`: {len(res.focus.representatives)} representative runs"]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
