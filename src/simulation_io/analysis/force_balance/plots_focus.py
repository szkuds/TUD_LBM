"""The focused figures: what the SI model misses, when, and what beta depends on.

* ``F1a_ladder_si`` — Ca(X) of the data against the SI model and the SI model with
  one ingredient (friction, hysteresis) and then both (M4) taken from the run.
* ``F1b_ladder_predictive`` — the data-filled M4 against the predictive closure M6.
* ``F2_decomposition_<run>`` — per representative run: Ca, the force terms, and the
  forces the SI model misses (dF_H, dF_beta) with what is left (R).
* ``F3a_missing_heatmap`` — where (phase) and for which runs each missing force matters.
* ``F3b_si_error_vs_parameter`` — the SI model's Ca error per phase against each
  group's varied parameter.
* ``F4_beta_scaling`` — beta against A**2/Omega and the mean angle, and what the
  friction constant c still depends on.
* ``F5_ladder_summary`` — median Ca(X) error of every rung, per phase.
* ``F6a_lambda_time`` — the SI friction coefficient ``lambda/eta = beta_eff/A``
  against time for the representative runs, with the constant of M7.
* ``F6b_lambda_phases`` — its median per phase over the run's pre-step value, all runs.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from src.simulation_io.analysis.force_balance.decomposition import PHASES
from src.simulation_io.analysis.force_balance.decomposition import phase_labels
from src.simulation_io.analysis.force_balance.geometry import base_squared_over_area
from src.simulation_io.analysis.force_balance.ladder import RUNGS
from src.simulation_io.analysis.force_balance.pipeline import run_slug
from src.simulation_io.analysis.force_balance.plots import INK
from src.simulation_io.analysis.force_balance.plots import LABEL_CA
from src.simulation_io.analysis.force_balance.plots import LABEL_X
from src.simulation_io.analysis.force_balance.plots import MUTED
from src.simulation_io.analysis.force_balance.plots import REGIME_COLOURS
from src.simulation_io.analysis.force_balance.plots import REGIME_MARKERS
from src.simulation_io.analysis.force_balance.plots import SLOTS
from src.simulation_io.analysis.force_balance.plots import _save
from src.simulation_io.analysis.force_balance.plots import _scatter
from src.simulation_io.analysis.force_balance.plots import _style_axes
from src.simulation_io.analysis.force_balance.scaling import C_PARAMETERS

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    import matplotlib.axes
    import pandas as pd
    from src.simulation_io.analysis.force_balance.pipeline import Results

RUNG_STYLE = {
    "SI": (SLOTS[1], (0, (5, 2)), "SI model"),
    "SI+beta(A)": (SLOTS[2], "-.", r"SI + $\beta(A)$ from data"),
    "SI+H_meas": (SLOTS[3], ":", r"SI + $H$ from data"),
    "M7": (SLOTS[5], (0, (3, 1, 1, 1)), r"M7: $\beta_f = \lambda A/\eta$ + $H$ from data"),
    "M4": (SLOTS[0], "-", r"M4: $\beta(A)$ + $H$ from data"),
    "M6": (SLOTS[6], "-", "M6: predictive closure"),
}
PHASE_COLOURS = dict(zip(PHASES, (SLOTS[0], SLOTS[1], SLOTS[2], SLOTS[6], SLOTS[4]), strict=True))
#: Phase bands drawn behind the series: only the two transitions, as light tints.
PHASE_BANDS = {"lead jump": "#fdece4", "hold": "#e4eefb"}
TERM_STYLE = {
    "dF_w": (SLOTS[1], "o", r"$\Delta F_w$: constant base width"),
    "dF_H": (SLOTS[2], "s", r"$\Delta F_H$: hysteresis dynamics"),
    "dF_beta": (SLOTS[3], "D", r"$\Delta F_\beta$: friction the SI lacks"),
    "R": (SLOTS[6], "X", r"$R$: left after M4"),
}
LABEL_BO = r"$\mathrm{Bo}_\parallel$"
LABEL_OH = r"$\mathrm{Oh}$"

#: The parameter each group varies, with its axis label.
GROUP_PARAMETER = {
    "inc": ("Bo", LABEL_BO),
    "tau": ("Oh", LABEL_OH),
    "r": ("R0", r"$R_0$ [LU]"),
    "pre": ("H1", r"$H_1$"),
    "post": ("H3", r"$H_3$"),
    "regime": ("Bo", LABEL_BO),
    "si0": ("Oh", LABEL_OH),
}
#: Room reserved below the panels for a figure legend, in inches.
_LEGEND_INCHES = 0.9

#: Percentiles of the per-point lambda/eta shown in F6a; points beyond are clipped to the frame.
_LAMBDA_RANGE = (2.0, 98.0)

#: Upper percentile of the heatmap colour scale; cells above it saturate.
_HEATMAP_PERCENTILE = 98.0

BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#256abf", "#0d366b"])
_PARAM_LABELS = {
    "Oh": LABEL_OH,
    "R0": r"$R_0$ [LU]",
    "W_over_R0": r"$W/R_0$",
    "Bo": LABEL_BO,
    "Ca": r"$\mathrm{Ca}$",
    "theta_mean": r"$\bar\theta$ [deg]",
}


def _bands(ax: matplotlib.axes.Axes, x: np.ndarray, phases: np.ndarray) -> None:
    """Shade the lead-jump and hold intervals of *x*."""
    for phase, colour in PHASE_BANDS.items():
        sel = x[phases == phase]
        if sel.size:
            ax.axvspan(float(sel.min()), float(sel.max()), color=colour, zorder=0, label=phase)


def _run_df(res: Results, run: str) -> pd.DataFrame:
    return res.points[res.points["run"] == run].reset_index(drop=True)


def _ladder_panel(ax: matplotlib.axes.Axes, res: Results, run: str, rungs: tuple[str, ...]) -> None:
    df = _run_df(res, run)
    x = df["X_cm"].to_numpy()
    _bands(ax, x, phase_labels(df))
    _scatter(ax, x, df["Ca_cm"].to_numpy(), MUTED, "simulation")
    for rung in rungs:
        traj = res.focus.ladder[(run, rung)]
        colour, style, label = RUNG_STYLE[rung]
        n = traj.n_data
        ax.plot(traj.x[:n], traj.ca[:n], color=colour, linestyle=style, linewidth=1.5, label=label)
    info = res.infos[run]
    ax.set_title(f"{info.name}\nBo {info.bo:.2f}, Oh {info.oh:.2f}, $R_0$ {info.r0:.0f}", fontsize=8)
    pad = 0.05 * float(np.ptp(x))
    ax.set_xlim(float(x.min()) - pad, float(x.max()) + pad)
    _style_axes(ax)


def _grid_figure(n: int) -> tuple[plt.Figure, np.ndarray]:
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes


def _finish_grid(fig: plt.Figure, axes: np.ndarray, n: int, title: str, ylabel: str) -> None:
    for ax in axes.flat[:n]:
        ax.set_xlabel(r"$X_\mathrm{CM}$ [$-$]", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    fig.legend(list(unique.values()), list(unique.keys()), loc="lower center", ncol=len(unique), fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))


def ladder_figure(res: Results, out: Path, rungs: tuple[str, ...], name: str, title: str) -> str:
    """F1: Ca(X) of the representative runs against the chosen rungs."""
    runs = res.focus.representatives
    fig, axes = _grid_figure(len(runs))
    for ax, run in zip(axes.flat, runs, strict=False):
        _ladder_panel(ax, res, run, rungs)
    _finish_grid(fig, axes, len(runs), title, LABEL_CA)
    return _save(fig, out, name)


def decomposition_figure(res: Results, run: str, out: Path) -> str:
    """F2: per run, Ca against SI, M7 and M4, the force terms, the forces the SI model misses, and lambda/eta."""
    df = _run_df(res, run)
    forces = res.focus.forces[res.focus.forces["run"] == run]
    x = df["X_cm"].to_numpy()
    phases = phase_labels(df)
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 12.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.2, 1.2, 0.9]})
    for ax in axes:
        _bands(ax, x, phases)
    _ladder_panel(axes[0], res, run, ("SI", "M7", "M4"))
    axes[0].set_ylabel(LABEL_CA)

    ax = axes[1]
    ax.plot(x, forces["gravity"], color=SLOTS[1], linewidth=1.5, label=r"gravity $\mathrm{Bo}_\parallel$")
    _scatter(ax, x, forces["hysteresis"].to_numpy(), SLOTS[2], r"hysteresis $-H_\mathrm{meas}$", "s")
    _scatter(ax, x, forces["friction"].to_numpy(), SLOTS[3], r"friction $-\beta_{M4}\mathrm{Ca}$", "D")
    m7_friction = -res.fit("M7").params["lambda_eta"] * (df["A_lu"] * df["Ca_cm"]).to_numpy()
    _scatter(ax, x, m7_friction, SLOTS[5], r"friction $-\beta_{M7}\mathrm{Ca} = -(\lambda/\eta) A\,\mathrm{Ca}$", "P")
    _scatter(
        ax,
        x,
        forces["inertia"].to_numpy(),
        SLOTS[0],
        r"inertia $\mathrm{Oh}^{-2}\,d\mathrm{Ca}/dT$",
    )
    ax.axhline(0.0, color=INK, linewidth=0.6)
    ax.set_ylabel(r"force / $\gamma$ [$-$]")

    ax = axes[2]
    for term, (colour, marker, label) in TERM_STYLE.items():
        _scatter(ax, x, forces[term].to_numpy(), colour, label, marker)
    _scatter(ax, x, forces["R_M7"].to_numpy(), RUNG_STYLE["M7"][0], r"$R_{M7}$: left after M7", "P")
    ax.axhline(0.0, color=INK, linewidth=0.6)
    ax.set_ylabel(r"missing force / $\gamma$ [$-$]")

    _lambda_x_panel(axes[3], res, run, df)
    axes[3].set_xlabel(LABEL_X)
    for ax in axes:
        _style_axes(ax)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=True))
        ax.legend(list(unique.values()), list(unique.keys()), fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(f"{res.infos[run].name}: force balance and what the SI model misses", fontsize=10)
    fig.tight_layout()
    return _save(fig, out / "F2_decomposition", f"F2_decomposition_{run_slug(run)}")


def _lambda_x_panel(ax: matplotlib.axes.Axes, res: Results, run: str, df: pd.DataFrame) -> None:
    """``lambda/eta = beta_eff/A`` along the traversal, against M7's constant and M4's ``c A/Omega``."""
    x, lam = df["X_cm"].to_numpy(), df["lambda_eta_eff"].to_numpy()
    shown = np.isfinite(lam)
    _scatter(ax, x[shown], lam[shown], MUTED, r"data: $\beta_\mathrm{eff}/A$")
    m7 = res.fit("M7").params["lambda_eta"]
    ax.axhline(m7, color=RUNG_STYLE["M7"][0], linestyle="-.", linewidth=1.3, label=rf"M7: constant {m7:.3g}")
    c = res.fit("M4").params["c_lub"]
    m4 = (c * df["A_lu"] / df["Omega"]).to_numpy()
    ax.plot(x, m4, color=RUNG_STYLE["M4"][0], linewidth=1.3, label=r"M4: $c\,A/\Omega$")
    if shown.any():
        lo, hi = np.percentile(lam[shown], _LAMBDA_RANGE)
        lo, hi = min(lo, m7, float(m4.min())), max(hi, m7, float(m4.max()))
        pad = 0.1 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    medians = res.focus.lambda_phases.loc[res.focus.lambda_phases["run"] == run].set_index("phase")["lambda_eta"]
    text = ", ".join(f"{phase} {medians[phase]:.3g}" for phase in PHASES if phase in medians.index)
    ax.set_title(f"phase medians: {text}", fontsize=7, loc="left")
    ax.set_ylabel(r"$\lambda/\eta$ [LU$^{-1}$]")


def _ordered_runs(res: Results) -> list[tuple[str, str]]:
    """``(run, label)`` grouped by first group and ordered by that group's varied parameter."""
    out = []
    for group, (param, _) in GROUP_PARAMETER.items():
        members = [r for r, info in res.infos.items() if info.groups[0] == group]
        values = {r: _parameter(res, r, param) for r in members}
        out += [(r, f"{group}: {res.infos[r].name}") for r in sorted(members, key=values.__getitem__)]
    return out


def _parameter(res: Results, run: str, param: str) -> float:
    info = res.infos[run]
    w = info.windows
    return {"Bo": info.bo, "Oh": info.oh, "R0": info.r0, "H1": w.h1, "H2": w.h2, "H3": w.h3}[param]


def missing_heatmap(res: Results, out: Path) -> str:
    """F3a: RMS of each missing force per run and phase."""
    table = res.focus.forces_by_phase
    order = _ordered_runs(res)
    fig, axes = plt.subplots(
        1, len(TERM_STYLE), figsize=(16.0, 0.2 * len(order) + 2.0), sharey=True, layout="constrained"
    )
    # A common scale, so the terms compare; capped so one outlier cannot wash out the rest.
    vmax = float(np.nanpercentile(table[[f"rms_{t}" for t in TERM_STYLE]].to_numpy(), _HEATMAP_PERCENTILE))
    image = None
    for ax, term in zip(axes, TERM_STYLE, strict=True):
        grid = np.full((len(order), len(PHASES)), np.nan)
        for i, (run, _) in enumerate(order):
            sub = table[table["run"] == run].set_index("phase")
            for j, phase in enumerate(PHASES):
                if phase in sub.index:
                    grid[i, j] = sub.loc[phase, f"rms_{term}"]
        image = ax.imshow(grid, aspect="auto", cmap=BLUES, vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(PHASES)), PHASES, rotation=30, fontsize=8)
        ax.set_title(TERM_STYLE[term][2], fontsize=9)
    axes[0].set_yticks(range(len(order)), [label for _, label in order], fontsize=6)
    if image is not None:
        fig.colorbar(image, ax=axes, label=r"RMS missing force / $\gamma$ [$-$]", shrink=0.6, extend="max")
    fig.suptitle("Where the SI model's missing forces act (blank: phase not reached)", fontsize=10)
    return _save(fig, out, "F3a_missing_heatmap")


def si_error_figure(res: Results, out: Path) -> str:
    """F3b: the SI model's Ca(X) error per phase against each group's varied parameter."""
    errors = res.focus.ladder_errors
    si = errors[errors["rung"] == "SI"].set_index("run")
    groups = [g for g in GROUP_PARAMETER if any(info.groups[0] == g for info in res.infos.values())]
    cols = min(4, len(groups))
    rows = math.ceil(len(groups) / cols)
    height = 3.4 * rows + _LEGEND_INCHES
    fig, axes = plt.subplots(rows, cols, figsize=(max(4.0 * cols, 8.0), height), squeeze=False)
    for ax in axes.flat[len(groups) :]:
        ax.axis("off")
    for ax, group in zip(axes.flat, groups, strict=False):
        param, label = GROUP_PARAMETER[group]
        members = [r for r, info in res.infos.items() if info.groups[0] == group]
        values = np.array([_parameter(res, r, param) for r in members])
        for phase in PHASES:
            y = si.loc[members, phase].to_numpy(dtype=float)
            ax.scatter(values, y, s=22, color=PHASE_COLOURS[phase], label=phase)
        ax.set_xlabel(label)
        ax.set_title(f"group '{group}'", fontsize=9)
        _style_axes(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"SI model: RMS error of $\mathrm{Ca}(X)$ [$-$]", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(PHASES), fontsize=9, title="phase")
    fig.suptitle("When the SI model fails: its error per phase against the varied parameter", fontsize=10)
    fig.tight_layout(rect=(0, _LEGEND_INCHES / height, 1, 1 - 0.5 / height))
    return _save(fig, out, "F3b_si_error_vs_parameter")


def _c_panel(ax: matplotlib.axes.Axes, res: Results, param: str) -> None:
    fr = res.friction
    for regime in (1, 2, 3):
        sub = fr[fr["regime"] == regime]
        ax.scatter(sub[param], sub["c_lub"], s=14, color=REGIME_COLOURS[regime], marker=REGIME_MARKERS[regime])
    dep = res.focus.c_dependence
    lines = []
    for regime in (1, 2, 3) if not dep.empty else ():
        row = dep[(dep["regime"] == regime) & (dep["parameter"] == param)]
        if not row.empty:
            r = row.iloc[0]
            lines.append(f"{regime}: {r.exponent:+.2f} [{r.exponent_ci_low:+.2f}, {r.exponent_ci_high:+.2f}]")
    ax.set_title("log-log exponent per regime\n" + "\n".join(lines), fontsize=6, loc="left")
    ax.tick_params(labelsize=7)
    ax.set_xlabel(_PARAM_LABELS[param], fontsize=8)
    _style_axes(ax)


def beta_scaling_figure(res: Results, out: Path) -> str:
    """F4: beta against geometry and the residual dependence of ``c = beta Omega/A**2``."""
    fr = res.friction
    laws = res.focus.geometry_laws
    fig = plt.figure(figsize=(15.0, 8.5))
    grid = fig.add_gridspec(2, 5, height_ratios=[1.3, 1.0])
    ax_a = fig.add_subplot(grid[0, :3])
    ax_b = fig.add_subplot(grid[0, 3:])
    for regime in (1, 2, 3):
        sub = fr[fr["regime"] == regime]
        colour, marker, label = REGIME_COLOURS[regime], REGIME_MARKERS[regime], f"regime {regime}"
        ax_a.scatter(sub["A2_over_Omega"], sub["beta"], color=colour, marker=marker, s=24, label=label)
        ax_b.scatter(sub["theta_mean"], sub["beta"], color=colour, marker=marker, s=24, label=label)
    law = laws[(laws["regime"] == "all") & (laws["law"] == "beta ~ (A2_over_Omega)^p")].iloc[0]
    a2 = np.linspace(fr["A2_over_Omega"].min() * 0.9, fr["A2_over_Omega"].max() * 1.1, 100)
    ax_a.plot(
        a2,
        law.prefactor * a2**law.exponent,
        color=INK,
        linewidth=1.2,
        label=rf"fit $\beta = {law.prefactor:.2f}\,(A^2/\Omega)^{{{law.exponent:.2f}}}$, "
        rf"$p \in [{law.exponent_ci_low:.2f}, {law.exponent_ci_high:.2f}]$",
    )
    c = res.fit("M4").params["c_lub"]
    ax_a.plot(a2, c * a2, color=MUTED, linestyle="--", linewidth=1.0, label=rf"$p = 1$: $\beta = {c:.2f}\,A^2/\Omega$")
    lam = res.fit("M7").params["lambda_eta"]
    r0 = float(np.median([info.r0 for info in res.infos.values()]))
    m7_label = rf"M7: $\beta = (\lambda/\eta)\,A$, $\lambda/\eta$ = {lam:.3g} LU$^{{-1}}$, $R_0$ = {r0:.0f} LU"
    ax_a.plot(a2, lam * r0 * np.sqrt(a2), color=SLOTS[5], linestyle="-.", linewidth=1.2, label=m7_label)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"$A^2/\Omega$ [$-$]")
    ax_a.set_ylabel(r"$\beta_\mathrm{eff}$ [$-$]")
    theta = np.linspace(45.0, 120.0, 200)
    ax_b.plot(
        theta,
        c * np.array([base_squared_over_area(math.radians(t)) for t in theta]),
        color=INK,
        linewidth=1.2,
        label=r"circular cap: $\beta = c\,\frac{4\sin^2\bar\theta}{\bar\theta - \sin\bar\theta\cos\bar\theta}$",
    )
    cap = np.array([base_squared_over_area(math.radians(t)) for t in theta])
    ax_b.plot(theta, lam * r0 * np.sqrt(cap), color=SLOTS[5], linestyle="-.", linewidth=1.2, label=m7_label)
    ax_b.set_xlabel(r"mean contact angle $\bar\theta$ [deg]")
    ax_b.set_ylabel(r"$\beta_\mathrm{eff}$ [$-$]")
    for ax in (ax_a, ax_b):
        _style_axes(ax)
        ax.legend(fontsize=8)
    shown = [p for p in C_PARAMETERS if p != "W_over_R0"]  # W/R0 = 5/R0 here: the R0 panel says the same
    for k, param in enumerate(shown):
        ax = fig.add_subplot(grid[1, k])
        _c_panel(ax, res, param)
        if k == 0:
            ax.set_ylabel(r"$c = \beta\,\Omega/A^2$ [$-$]")
    note = " ".join(res.focus.notes)
    fig.suptitle("What beta depends on: geometry (top) and what is left in c (bottom)\n" + note, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out, "F4_beta_scaling")


def ladder_summary(res: Results, out: Path) -> str:
    """F5: median Ca(X) error over runs, per rung and phase."""
    errors = res.focus.ladder_errors
    columns = ("all", *PHASES)
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    width = 0.8 / len(RUNGS)
    for k, rung in enumerate(RUNGS):
        sub = errors[errors["rung"] == rung]
        med = [float(sub[col].median()) for col in columns]
        lo = [float(sub[col].quantile(0.25)) for col in columns]
        hi = [float(sub[col].quantile(0.75)) for col in columns]
        pos = np.arange(len(columns)) + (k - (len(RUNGS) - 1) / 2) * width
        yerr = np.array([np.subtract(med, lo), np.subtract(hi, med)])
        colour, _, label = RUNG_STYLE[rung]
        ax.bar(pos, med, width=width * 0.92, color=colour, label=label, yerr=yerr, capsize=2, ecolor=INK)
    ax.set_xticks(range(len(columns)), columns)
    ax.set_ylabel(r"median RMS error of $\mathrm{Ca}(X)$ [$-$]")
    _style_axes(ax)
    ax.legend(fontsize=8, ncol=len(RUNGS), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("What each data-filled ingredient buys, phase by phase (whiskers: quartiles over runs)", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "F5_ladder_summary")


def _lambda_panel(ax: matplotlib.axes.Axes, res: Results, run: str, m7: float) -> None:
    df = _run_df(res, run)
    phases = phase_labels(df)
    t, lam = df["T"].to_numpy(), df["lambda_eta_eff"].to_numpy()
    shown = np.isfinite(lam)
    for phase in PHASES:
        sel = shown & (phases == phase)
        ax.scatter(t[sel], lam[sel], s=8, color=PHASE_COLOURS[phase], label=phase)
    ax.axhline(m7, color=INK, linewidth=1.0, linestyle="--", label=rf"M7: $\lambda/\eta$ = {m7:.3g} LU$^{{-1}}$")
    if shown.any():
        lo, hi = np.percentile(lam[shown], _LAMBDA_RANGE)
        pad = 0.1 * (hi - lo)
        ax.set_ylim(min(lo, m7) - pad, max(hi, m7) + pad)
    info = res.infos[run]
    ax.set_title(f"{info.name}\nBo {info.bo:.2f}, Oh {info.oh:.2f}, $R_0$ {info.r0:.0f}", fontsize=8)
    ax.set_xlabel(r"$T$ [$-$]", fontsize=8)
    _style_axes(ax)


def lambda_time_figure(res: Results, out: Path) -> str:
    """F6a: ``lambda/eta = beta_eff/A`` against time, per representative run."""
    runs = res.focus.representatives
    m7 = res.fit("M7").params["lambda_eta"]
    fig, axes = _grid_figure(len(runs))
    for ax, run in zip(axes.flat, runs, strict=False):
        _lambda_panel(ax, res, run, m7)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\lambda/\eta = \beta_\mathrm{eff}/A$ [LU$^{-1}$]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
    fig.suptitle(r"The SI friction coefficient over time: $\beta_f = \lambda A/\eta$ at every point", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return _save(fig, out, "F6a_lambda_time")


def lambda_phase_figure(res: Results, out: Path) -> str:
    """F6b: median ``lambda/eta`` per phase over the run's own pre-step median, every run."""
    table = res.focus.lambda_phases.dropna(subset=["ratio_to_pre"])
    first_group = {run: info.groups[0] for run, info in res.infos.items()}
    present = set(table["run"].map(first_group))
    groups = [g for g in GROUP_PARAMETER if g in present]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    width = 0.6
    for k, group in enumerate(groups):
        sub = table[table["run"].map(first_group) == group]
        offset = (k - (len(groups) - 1) / 2) * width / max(len(groups), 1)
        pos = sub["phase"].map({p: i for i, p in enumerate(PHASES)}).to_numpy() + offset
        ax.scatter(pos, sub["ratio_to_pre"], s=16, color=SLOTS[k % len(SLOTS)], label=f"group '{group}'")
    medians = [float(table.loc[table["phase"] == p, "ratio_to_pre"].median()) for p in PHASES]
    ax.scatter(range(len(PHASES)), medians, s=160, marker="_", color=INK, linewidths=2.5, label="median over runs")
    ax.axhline(1.0, color=MUTED, linewidth=0.8)
    ax.set_xticks(range(len(PHASES)), PHASES)
    ax.set_ylabel(r"$(\lambda/\eta)_\mathrm{phase} / (\lambda/\eta)_\mathrm{pre}$ [$-$]")
    ax.set_title(r"If $\beta_f = \lambda A/\eta$ held with one $\lambda$, every phase would sit at 1", fontsize=10)
    _style_axes(ax)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return _save(fig, out, "F6b_lambda_phases")


def make_focus_figures(res: Results, out: Path, progress: Callable[[str], None]) -> list[str]:
    """Every focused figure; returns their names (without extension)."""
    names = [
        ladder_figure(
            res, out, ("SI", "SI+beta(A)", "SI+H_meas", "M7", "M4"), "F1a_ladder_si", "SI model, then data-filled"
        ),
        ladder_figure(
            res, out, ("SI", "M7", "M4", "M6"), "F1b_ladder_predictive", "SI friction (M7), M4 and predictive M6"
        ),
    ]
    names += [decomposition_figure(res, run, out) for run in res.focus.representatives]
    names += [
        missing_heatmap(res, out),
        si_error_figure(res, out),
        beta_scaling_figure(res, out),
        ladder_summary(res, out),
        lambda_time_figure(res, out),
        lambda_phase_figure(res, out),
    ]
    progress(f"{len(names)} focused figures")
    return names
