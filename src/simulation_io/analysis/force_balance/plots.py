"""Figures of the force-balance analysis, each saved as PDF and as a 220 dpi PNG.

Measured series are scatter, model curves are lines (the plotting rule of
this package). Colours: one categorical slot per entity in a fixed order,
never reused within a figure, and a single-hue blue ramp for ordered
quantities (Oh, R0). Every series is in a legend.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from src.simulation_io.analysis.force_balance.fits import MODEL_BY_KEY
from src.simulation_io.analysis.force_balance.fits import friction
from src.simulation_io.analysis.force_balance.fits import param_vector
from src.simulation_io.analysis.force_balance.geometry import StepClosure
from src.simulation_io.analysis.force_balance.geometry import base_squared_over_area
from src.simulation_io.analysis.force_balance.pipeline import OVERALL
from src.simulation_io.analysis.force_balance.pipeline import run_slug
from src.simulation_io.analysis.force_balance.points import AFTER
from src.simulation_io.analysis.force_balance.points import STRADDLING
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    import matplotlib.axes
    import matplotlib.figure
    import pandas as pd
    from src.simulation_io.analysis.force_balance.pipeline import Results

DPI = 220

#: Validated categorical palette (light surface), in fixed slot order.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
INK = "#0b0b0b"
MUTED = "#8a8985"
RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

#: Models drawn against the data in the Ca(X) figures, one slot each.
CURVE_MODELS = ("M0", "M1", "M2", "M3", "M4", "M6", "M7")
#: One colour per model, the same as in the focused figures (M6 and M7 take the last two slots).
MODEL_COLOURS = {**dict(zip(CURVE_MODELS[:5], SLOTS, strict=False)), "M6": SLOTS[6], "M7": SLOTS[5]}
MODEL_STYLES = {"M0": ":", "M1": "--", "M2": "-.", "M3": "-", "M4": "-", "M6": (0, (5, 1)), "M7": (0, (3, 1, 1, 1))}
GROUP_COLOURS = dict(zip(("inc", "tau", "r", "pre", "post", "regime", "si0"), SLOTS, strict=True))
REGIME_COLOURS = {1: SLOTS[0], 2: SLOTS[1], 3: SLOTS[2]}
REGIME_MARKERS = {1: "o", 2: "s", 3: "D"}
TERM_COLOURS = {
    "inertia": SLOTS[0],
    "gravity": SLOTS[1],
    "hysteresis": SLOTS[2],
    "friction": SLOTS[3],
    "residual": SLOTS[6],
}

LABEL_X = r"$X_\mathrm{CM} = (x_\mathrm{CM} - x_\mathrm{step})/R_0$ [$-$]"
LABEL_CA = r"$\mathrm{Ca}$ [$-$]"
MARKER = 10


def _style_axes(ax: matplotlib.axes.Axes) -> None:
    ax.grid(visible=True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=DEFAULT_STYLE.tick_label_fontsize - 3)


def _save(fig: matplotlib.figure.Figure, out: Path, name: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return name


def _step_marks(ax: matplotlib.axes.Axes, df: pd.DataFrame) -> None:
    """Dotted verticals where the leading line reaches and the trailing line leaves the step."""
    lead = df.index[df["regime"] >= STRADDLING]
    trail = df.index[df["regime"] == AFTER]
    if len(lead):
        ax.axvline(float(df.loc[lead[0], "X_cm"]), color=MUTED, linestyle=":", linewidth=0.8)
    if len(trail):
        ax.axvline(float(df.loc[trail[0], "X_cm"]), color=MUTED, linestyle=":", linewidth=0.8)


def _figure_legend(
    fig: matplotlib.figure.Figure, handles: list, labels: list[str], ncol: int, y: float, fontsize: int = 9
) -> None:
    """One legend for the whole figure, centred below the panels; repeated labels shown once."""
    unique = dict(zip(labels, handles, strict=True))
    fig.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="lower center",
        ncol=ncol,
        fontsize=fontsize,
        bbox_to_anchor=(0.5, y),
    )


def _scatter(
    ax: matplotlib.axes.Axes, x: np.ndarray, y: np.ndarray, colour: str, label: str, marker: str = "o"
) -> None:
    ax.scatter(
        x, y, s=MARKER, color=colour, label=label, marker=marker, alpha=DEFAULT_STYLE.scatter_alpha, linewidths=0
    )


def force_balance_figure(res: Results, run: str, out: Path) -> str:
    """(1) Per run: Ca of centre and lines, and the force terms of M1, M4 and M7."""
    df = res.points[res.points["run"] == run].reset_index(drop=True)
    info = res.infos[run]
    x = df["X_cm"].to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(10.0, 13.0), sharex=True)
    ax = axes[0]
    _scatter(ax, x, df["Ca_cm"], SLOTS[0], r"$\mathrm{Ca}_\mathrm{CM}$")
    _scatter(ax, x, df["Ca_lead"], SLOTS[1], r"$\mathrm{Ca}_\mathrm{lead}$", "^")
    _scatter(ax, x, df["Ca_trail"], SLOTS[2], r"$\mathrm{Ca}_\mathrm{trail}$", "v")
    ax.set_ylabel(LABEL_CA)
    ax.set_title(f"{info.name}: Bo = {info.bo:.2f}, Oh = {info.oh:.2f}, $R_0$ = {info.r0:.1f} LU", fontsize=10)
    for ax, key, hcol in ((axes[1], "M1", "H_step"), (axes[2], "M4", "H_meas"), (axes[3], "M7", "H_meas")):
        fit = res.fit(key)
        model = MODEL_BY_KEY[key]
        fr = friction(df, model, param_vector(fit))
        inertia = df["inertia_cm"].to_numpy()
        resid = inertia - (df["Bo"] - df[hcol] - fr).to_numpy()
        _scatter(
            ax,
            x,
            inertia,
            TERM_COLOURS["inertia"],
            r"inertia (mass $\times$ accel.) $\mathrm{Oh}^{-2}\,d\mathrm{Ca}/dT$",
        )
        ax.plot(x, df["Bo"], color=TERM_COLOURS["gravity"], linewidth=1.5, label=r"gravity $\mathrm{Bo}_\parallel$")
        _scatter(
            ax, x, -df[hcol].to_numpy(), TERM_COLOURS["hysteresis"], f"hysteresis $-H$ ({hcol.replace('_', ' ')})", "s"
        )
        _scatter(ax, x, -fr, TERM_COLOURS["friction"], r"friction $-\beta\,\mathrm{Ca}$", "D")
        _scatter(ax, x, resid, TERM_COLOURS["residual"], r"residual $F_\mathrm{res}$", "X")
        ax.axhline(0.0, color=INK, linewidth=0.6)
        ax.set_ylabel(f"{key}: force / $\\gamma$ [$-$]")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axes[-1].set_xlabel(LABEL_X)
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for ax in axes:
        _style_axes(ax)
        _step_marks(ax, df)
    return _save(fig, out / "force_balance", f"force_balance_{run_slug(run)}")


def friction_figure(res: Results, out: Path) -> str:
    """(2) beta against contact length, against Ca, and per contact length against the angle."""
    fr = res.friction
    c = res.fit("M4").params["c_lub"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    for regime in (1, 2, 3):
        sub = fr[fr["regime"] == regime]
        kw = {"color": REGIME_COLOURS[regime], "marker": REGIME_MARKERS[regime], "s": 24, "label": f"regime {regime}"}
        axes[0].scatter(sub["A_over_R0"], sub["beta"], **kw)
        axes[1].scatter(sub["Ca"], sub["beta"], **kw)
        axes[2].scatter(sub["theta_mean"], sub["lambda_per_lu"], **kw)
    a = np.linspace(0.9, 2.6, 100)
    axes[0].plot(a, c * a**2, color=INK, linewidth=1.2, label=rf"M4: $\beta = {c:.2f}\,A^2/\Omega$")
    theta = np.linspace(45.0, 120.0, 100)
    a2o = np.array([base_squared_over_area(math.radians(t)) for t in theta])
    r0 = float(np.median([i.r0 for i in res.infos.values()]))
    lam = res.fit("M7").params["lambda_eta"]
    m7_label = rf"M7: $\lambda/\eta$ = {lam:.3g} LU$^{{-1}}$ ($R_0$ = {r0:.1f} LU)"
    axes[0].plot(a, lam * r0 * a, color=MODEL_COLOURS["M7"], linestyle="-.", linewidth=1.2, label=m7_label)
    axes[2].axhline(lam, color=MODEL_COLOURS["M7"], linestyle="-.", linewidth=1.2, label=m7_label)
    axes[2].plot(
        theta, c * np.sqrt(a2o) / r0, color=INK, linewidth=1.2, label=rf"M4 on a circular cap, $R_0$ = {r0:.1f} LU"
    )
    axes[0].set_xlabel(r"contact length $A/R_0$ [$-$]")
    axes[0].set_ylabel(
        r"$\beta_\mathrm{eff} = (\mathrm{Bo} - H_\mathrm{meas} - \mathrm{Oh}^{-2}\,d\mathrm{Ca}/dT)/\mathrm{Ca}$ [$-$]"
    )
    axes[1].set_xlabel(r"$\mathrm{Ca}_\mathrm{CM}$ [$-$]")
    axes[1].set_ylabel(r"$\beta_\mathrm{eff}$ [$-$]")
    axes[2].set_xlabel(r"mean contact angle $(\theta_\mathrm{lead}+\theta_\mathrm{trail})/2$ [deg]")
    axes[2].set_ylabel(r"$\lambda/\eta = \beta_\mathrm{eff}/A$ [LU$^{-1}$]")
    for ax in axes:
        _style_axes(ax)
        ax.legend(fontsize=8)
    fig.suptitle("Per run and regime: median over quasi-steady moving points", fontsize=10)
    return _save(fig, out, "02_beta_vs_contact_length")


def _interval(
    ax: matplotlib.axes.Axes,
    xs: float,
    fit_val: float,
    lo: float,
    hi: float,
    colour: str,
    marker: str,
    label: str | None,
) -> None:
    yerr = np.array([[fit_val - lo], [hi - fit_val]]) if np.isfinite(lo) else None
    ax.errorbar([xs], [fit_val], yerr=yerr, fmt=marker, color=colour, capsize=3, markersize=6, label=label)


def lambda_figure(res: Results, out: Path) -> str:
    """(3) lambda/eta of M2, M3 (per surface) and M7 (one), and the M4/M6 constants per scope, with 95% intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    scopes = res.scopes
    entries = (
        ("M2", "lambda_phob", SLOTS[0], "o", -0.3),
        ("M2", "lambda_phil", SLOTS[1], "o", -0.15),
        ("M3", "lambda_phob", SLOTS[2], "s", 0.0),
        ("M3", "lambda_phil", SLOTS[3], "s", 0.15),
        ("M7", "lambda_eta", MODEL_COLOURS["M7"], "P", 0.3),
    )
    for k, scope in enumerate(scopes):
        for key, p, colour, marker, dx in entries:
            f = res.fit(key, scope)
            _interval(
                axes[0],
                k + dx,
                f.params[p],
                f.ci_low[p],
                f.ci_high[p],
                colour,
                marker,
                f"{key} {p.replace('_', ' ')}" if k == 0 else None,
            )
        for key, p, colour, marker, dx in (
            ("M4", "c_lub", MODEL_COLOURS["M4"], "D", -0.12),
            ("M6", "c_geom", MODEL_COLOURS["M6"], "^", 0.12),
        ):
            f = res.fit(key, scope)
            _interval(
                axes[1],
                k + dx,
                f.params[p],
                f.ci_low[p],
                f.ci_high[p],
                colour,
                marker,
                f"{key} {p.replace('_', ' ')}" if k == 0 else None,
            )
    axes[0].set_ylabel(r"$\lambda/\eta$ [LU$^{-1}$]")
    axes[1].set_ylabel(r"$c$ in $\beta = c\,A^2/\Omega$ [$-$]")
    axes[1].axhline(3.0, color=MUTED, linestyle=":", linewidth=1.0, label="flat-film lubrication, c = 3")
    for ax in axes:
        ax.set_xticks(range(len(scopes)), scopes)
        ax.set_xlabel("group (fit scope)")
        _style_axes(ax)
        ax.legend(fontsize=8)
    fig.suptitle("Fitted friction parameters, 95% run-level bootstrap intervals", fontsize=10)
    return _save(fig, out, "03_lambda_fits")


def _grid(n: int) -> tuple[int, int]:
    cols = min(4, n)
    return math.ceil(n / cols), cols


def ca_models_figure(res: Results, scope: str, out: Path) -> str:
    """(4) Per group: measured Ca(X) against the M0, M1, M2, M3, M4 and M6 trajectories (group fits)."""
    runs = res.runs_in(scope)
    rows, cols = _grid(len(runs))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows), squeeze=False)
    for ax, run in zip(axes.flat, runs, strict=False):
        df = res.points[res.points["run"] == run]
        _scatter(ax, df["X_cm"].to_numpy(), df["Ca_cm"].to_numpy(), MUTED, "data")
        for key in CURVE_MODELS:
            traj = res.trajectories.get((run, key, scope))
            if traj is None:
                continue
            n = traj.n_data
            ax.plot(
                traj.x[:n], traj.ca[:n], color=MODEL_COLOURS[key], linestyle=MODEL_STYLES[key], linewidth=1.3, label=key
            )
        info = res.infos[run]
        ax.set_title(f"{info.name}\nBo {info.bo:.2f}, Oh {info.oh:.2f}", fontsize=8)
        _style_axes(ax)
    for ax in axes.flat[len(runs) :]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(r"$X_\mathrm{CM}$ [$-$]")
    for ax in axes[:, 0]:
        ax.set_ylabel(LABEL_CA)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    _figure_legend(fig, handles, labels, ncol=len(labels), y=-0.01)
    fig.suptitle(f"Ca(X): data against models fitted to group '{scope}'", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return _save(fig, out, f"04_ca_x_models_{scope}")


def hysteresis_figure(res: Results, scope: str, out: Path) -> str:
    """(5) Per group: measured H against the stepwise, the solver-window and the closure H."""
    runs = res.runs_in(scope)
    rows, cols = _grid(len(runs))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows), squeeze=False)
    for ax, run in zip(axes.flat, runs, strict=False):
        df = res.points[res.points["run"] == run].sort_values("X_cm")
        x = df["X_cm"].to_numpy()
        _scatter(ax, x, df["H_meas"].to_numpy(), SLOTS[0], r"$H_\mathrm{meas}$")
        ax.plot(x, df["H_step"], color=SLOTS[1], linewidth=1.3, label=r"$H$ stepwise (regime)")
        ax.plot(x, df["H_window"], color=SLOTS[2], linewidth=1.0, linestyle="--", label=r"$H$ solver windows")
        xg = np.linspace(x.min(), x.max(), 300)
        _, hg = StepClosure(res.infos[run].windows).vectorised(xg)
        ax.plot(xg, hg, color=SLOTS[4], linewidth=1.0, linestyle="-.", label=r"$H$ closure (M6)")
        held = df[df["trail_held"]]
        if not held.empty:
            ax.axvspan(
                float(held["X_cm"].min()), float(held["X_cm"].max()), color="#f0efec", label="trailing line held"
            )
        ax.set_title(res.infos[run].name, fontsize=8)
        _style_axes(ax)
    for ax in axes.flat[len(runs) :]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(r"$X_\mathrm{CM}$ [$-$]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$H = \cos\theta_\mathrm{trail} - \cos\theta_\mathrm{lead}$ [$-$]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    _figure_legend(fig, handles, labels, ncol=len(labels), y=-0.01)
    fig.suptitle(f"Measured and modelled hysteresis, group '{scope}'", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return _save(fig, out, f"05_h_meas_{scope}")


def _onset_points(axes: np.ndarray, h: pd.DataFrame) -> None:
    """``Bo - H_start`` against Oh (left) and R0 (right): marker by observed state, colour by agreement."""
    markers = {"moving": "o", "creep": "s", "pinned": "X"}
    for state, marker in markers.items():
        for agrees in (True, False):
            sub = h[(h["observed"] == state) & (h["agrees"] == agrees)]
            if sub.empty:
                continue
            colour = SLOTS[0] if agrees else SLOTS[1]
            label = f"{state}, {'agrees' if agrees else 'disagrees'} with prediction"
            for ax, xcol in ((axes[0], "Oh"), (axes[1], "R0")):
                ax.scatter(sub[xcol], sub["Bo"] - sub["H_start"], marker=marker, color=colour, s=40, label=label)


def regime_map_figure(res: Results, out: Path) -> str:
    """(6) Onset of motion: Bo - H_start against Oh and against R0, marked by observed state and agreement."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    _onset_points(axes, res.hysteresis)
    for ax in axes:
        ax.axhline(
            0.0,
            color=INK,
            linewidth=1.0,
            linestyle="--",
            label=r"predicted onset $\mathrm{Bo}_\parallel = H_\mathrm{start}$",
        )
        ax.set_ylabel(r"$\mathrm{Bo}_\parallel - H_\mathrm{start}$ [$-$]")
        _style_axes(ax)
    axes[0].set_xlabel(r"$\mathrm{Oh}$ [$-$]")
    axes[1].set_xlabel(r"$R_0 = \sqrt{\Omega_D}$ [LU]")
    handles, labels = axes[0].get_legend_handles_labels()
    _figure_legend(fig, handles, labels, ncol=3, y=-0.06, fontsize=8)
    fig.suptitle("Onset of motion: every run, with the predicted threshold", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return _save(fig, out, "06_regime_map_onset")


def overshoot_figure(res: Results, out: Path) -> str:
    """(7) Inverse problem: H_eff(X) per run of the tau and R0 series, coloured by the varied quantity."""
    from src.simulation_io.analysis.force_balance.analyses import effective_hysteresis

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    heff = effective_hysteresis(res.points, res.fit("M4"))
    for ax, scope, key, label in ((axes[0], "tau", "Oh", "Oh"), (axes[1], "r", "R0", r"$R_0$")):
        runs = res.runs_in(scope)
        if not runs:
            ax.axis("off")
            continue
        values = [res.points.loc[res.points["run"] == r, key].iloc[0] for r in runs]
        lo, hi = min(values), max(values)
        for run, val in sorted(zip(runs, values, strict=True), key=lambda p: p[1]):
            sub = heff[heff["run"] == run]
            colour = RAMP((val - lo) / (hi - lo) if hi > lo else 0.5)
            ax.scatter(sub["X_cm"], sub["H_eff"], s=8, color=colour, label=f"{label} = {val:.2f}")
        ref = res.points[res.points["run"] == runs[0]].sort_values("X_cm")
        ax.plot(ref["X_cm"], ref["H_step"], color=INK, linestyle="--", linewidth=1.0, label=r"stepwise $H$")
        ax.set_title(f"group '{scope}'", fontsize=10)
        ax.set_xlabel(r"$X_\mathrm{CM}$ [$-$]")
        _style_axes(ax)
        ax.legend(fontsize=7)
    axes[0].set_ylabel(
        r"$H_\mathrm{eff} = \mathrm{Bo} - \mathrm{Oh}^{-2}\,d\mathrm{Ca}/dT - \beta_{M4}\mathrm{Ca}$ [$-$]"
    )
    fig.suptitle("Effective hysteresis implied by the data (friction from M4)", fontsize=10)
    return _save(fig, out, "07_overshoot_inverse_heff")


def _collapse_panel(ax: matplotlib.axes.Axes, col: pd.DataFrame, regime: int) -> None:
    """``Ca/Ca_inf`` against ``s = (X - X_entry)/|Lambda|`` in one regime, one colour per group."""
    sub = col[col["regime"] == regime]
    first_group = sub["groups"].str.split(";").str[0]
    for group, colour in GROUP_COLOURS.items():
        g = sub[first_group == group]
        if not g.empty:
            ax.scatter(g["s"], g["Ca_ratio"], s=5, color=colour, label=group, alpha=0.7)
    ax.axhline(1.0, color=INK, linewidth=0.8)
    ax.set_xlabel(rf"$(X - X_{{\mathrm{{entry}},{regime}}})/|\Lambda_{regime}|$ [$-$]")
    ax.set_ylabel(rf"$\mathrm{{Ca}}/\mathrm{{Ca}}_{{\infty,{regime}}}$ (M1) [$-$]")
    ax.set_title(f"regime {regime}", fontsize=10)
    right = min(30.0, float(sub["s"].quantile(0.99))) if not sub.empty else 30.0
    ax.set_xlim(left=-0.5, right=right)
    ax.set_ylim(-0.5, 3.0)


def _closure_plateau(res: Results, run: str, regime: int, c6: float) -> float:
    """``(Bo - H_i)/(c A_i**2)`` with the closure's cap base width."""
    info = res.infos[run]
    closure = StepClosure(info.windows)
    a = (closure.a1, closure.a2, closure.a3)[regime - 1]
    h = (info.windows.h1, info.windows.h2, info.windows.h3)[regime - 1]
    return (info.bo - h) / (c6 * a**2)


def _plateau_panel(ax: matplotlib.axes.Axes, res: Results, data: pd.DataFrame, hyst: pd.DataFrame) -> None:
    """Measured plateaus against M1 (filled) and M6 (open) predictions."""
    c6 = res.fit("M6").params["c_geom"]
    for regime in (1, 2, 3):
        pred_m1 = hyst[f"Ca_inf_{regime}_M1"]
        measured = data.loc[pred_m1.index, f"plateau_{regime}"]
        colour, marker = REGIME_COLOURS[regime], REGIME_MARKERS[regime]
        ax.scatter(pred_m1, measured, color=colour, marker=marker, s=20, label=f"regime {regime}, M1")
        pred_m6 = [_closure_plateau(res, run, regime, c6) for run in pred_m1.index]
        ax.scatter(
            pred_m6, measured, facecolors="none", edgecolors=colour, marker=marker, s=30, label=f"regime {regime}, M6"
        )
    lim = (0.0, 0.3)
    ax.plot(lim, lim, color=INK, linewidth=0.8, label="1:1")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"predicted $\mathrm{Ca}_{\infty,i} = (\mathrm{Bo} - H_i)/\beta_i$ [$-$]")
    ax.set_ylabel(r"measured plateau $\mathrm{Ca}_i$ [$-$]")
    ax.legend(fontsize=7)


def _length_panel(ax: matplotlib.axes.Axes, data: pd.DataFrame, hyst: pd.DataFrame) -> None:
    """Measured decay and rise lengths against the M1 coasting lengths."""
    lam3 = hyst["Lambda_3_M1"].abs()
    lam1 = hyst["Lambda_1_M1"].abs()
    ax.scatter(lam3, data.loc[lam3.index, "decay"], color=SLOTS[2], s=20, label=r"decay length vs $|\Lambda_3|$ (M1)")
    ax.scatter(
        lam1,
        data.loc[lam1.index, "rise_1"],
        color=SLOTS[0],
        marker="s",
        s=20,
        label=r"rise length vs $|\Lambda_1|$ (M1)",
    )
    ax.plot((0, 0.4), (0, 0.4), color=INK, linewidth=0.8, label="1:1")
    ax.set_xlabel(r"predicted coasting length $|\Lambda_i|$ [$-$]")
    ax.set_ylabel(r"measured length [$R_0$]")
    ax.legend(fontsize=7)


def _decay_panel(ax: matplotlib.axes.Axes, res: Results, data: pd.DataFrame) -> None:
    """Model against measured decay length, per model."""
    m = res.run_metrics
    for key in CURVE_MODELS:
        sub = m[(m["model"] == key) & (m["scope"] == OVERALL)].set_index("run")
        common = sub.index.intersection(data.index)
        ax.scatter(data.loc[common, "decay"], sub.loc[common, "decay"], color=MODEL_COLOURS[key], s=14, label=key)
    ax.plot((0, 0.4), (0, 0.4), color=INK, linewidth=0.8, label="1:1")
    ax.set_xlabel(r"measured decay length [$R_0$]")
    ax.set_ylabel(r"model decay length [$R_0$]")
    ax.legend(fontsize=7)


def collapse_figure(res: Results, out: Path) -> str:
    """(8) Ca/Ca_inf,i against (X - X_entry,i)/Lambda_i per regime, and plateaus against the predictions."""
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.0))
    for k, regime in enumerate((1, 2, 3)):
        _collapse_panel(axes[0, k], res.collapse, regime)
    handles, labels = [], []
    for ax in axes[0]:
        h, lab = ax.get_legend_handles_labels()
        handles += h
        labels += lab
    unique = dict(zip(labels, handles, strict=True))
    axes[0, 2].legend(
        list(unique.values()),
        list(unique.keys()),
        fontsize=7,
        markerscale=2,
        title="group (first listed)",
        title_fontsize=7,
    )
    data = res.run_metrics[res.run_metrics["model"] == "data"].set_index("run")
    hyst = res.hysteresis.set_index("run")
    _plateau_panel(axes[1, 0], res, data, hyst)
    _length_panel(axes[1, 1], data, hyst)
    _decay_panel(axes[1, 2], res, data)
    for ax in axes.flat:
        _style_axes(ax)
    fig.suptitle(
        "Scaling collapse (top) and plateaus, rise and decay lengths against the predictions (bottom)", fontsize=10
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out, "08_scaling_collapse")


def make_figures(res: Results, out: Path, progress: Callable[[str], None]) -> list[str]:
    """Every figure; returns their names (without extension)."""
    names = [force_balance_figure(res, run, out) for run in res.infos]
    progress(f"{len(names)} force-balance figures")
    names.append(friction_figure(res, out))
    names.append(lambda_figure(res, out))
    for scope in res.scopes:
        if scope == OVERALL:
            continue
        names.append(ca_models_figure(res, scope, out))
        names.append(hysteresis_figure(res, scope, out))
    names.append(regime_map_figure(res, out))
    names.append(overshoot_figure(res, out))
    names.append(collapse_figure(res, out))
    progress("summary figures")
    return names
