"""Why the density-ratio-1000 rising bubble diverges, and which change keeps it stable.

The reference case is ``~/TUD_LBM_data/bubble_simulation/2026-09-30/11-02-52_rising_bubble``:
128 x 256, BGK tau = 0.99, double-well kappa = 0.04, W = 5, rho_l = 1, rho_v = 1e-3,
g = 1e-6, bounce-back top/bottom, periodic sides. It hits NaN at t = 25000.

Phases (``python -m examples.density_ratio_study.run_study <phase> [--workers N]``):

* ``diagnose`` — the baseline, a static (g = 0) control, a stiffer double-well and a
  density-ratio ladder, to pin the mechanism.
* ``screen`` — ratio 1000, fixed geometry, g and liquid tau; every EOS against every
  collision variant (one-factor rows and the full cross product).
* ``matched`` — the screen's survivors rerun at the baseline's Bo and Oh.

A single case runs with ``case <name>``. Each case writes one ``run.npz`` under
``~/TUD_LBM_data/density_ratio_study/<phase>/<case>/`` holding its diagnostics time
series, density frames and parameters; ``report.py`` turns a phase directory into
``results.csv`` + ``report.md`` + figures. A case that ``SimulationConfig`` rejects
(e.g. a decoupled viscosity outside ``|A| < lambda_v - 1/2``) is recorded as such
and not run.
"""

from __future__ import annotations
import argparse
import json
import math
import multiprocessing
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.operators.protocols import EosOperator
    from src.pipeline.state.state import State

BASELINE_CONFIG = Path("~/TUD_LBM_data/bubble_simulation/2026-09-30/11-02-52_rising_bubble/config.toml").expanduser()
RESULTS = Path("~/TUD_LBM_data/density_ratio_study").expanduser()

#: Steps between diagnostic samples.
SAMPLE_INTERVAL = 250
#: A run that keeps less gas area than this fraction of its initial bubble lost the
#: bubble (it condensed into the liquid): numerically stable, physically void.
BUBBLE_LOST_FRACTION = 0.1
#: Steps between progress lines on stdout.
PROGRESS_INTERVAL = 5000
#: Density frames kept per run (evenly spaced).
N_FRAMES = 24
#: Screening horizon: 4x the baseline's failure time.
SCREEN_STEPS = 100_000

#: Collision variants. ``k_diag`` order: [rho, e, eps, jx, qx, jy, qy, pxx, pxy]; the
#: shear entries 7, 8 are rewritten to 1/relaxation_time by the config.
LAMBDA_V_OPT = 0.5 + math.sqrt(2.0) / 4.0  # cancels the leading error term (Zhang 2022)
COLLISIONS: dict[str, dict[str, Any]] = {
    "bgk": {"collision_scheme": "bgk"},
    # examples/config_MRT_simple.toml's free rates
    "mrt_ex": {"collision_scheme": "mrt", "k_diag": (0.0, 0.5, 0.6, 0.0, 1.2, 0.0, 1.2, 1.0, 1.0)},
    # Zhang 2022's free rates (all 1)
    "mrt_s1": {"collision_scheme": "mrt", "k_diag": (0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0)},
    # over-relaxed bulk/energy: less bulk viscosity
    "mrt_s15": {"collision_scheme": "mrt", "k_diag": (0.0, 1.5, 1.5, 0.0, 1.2, 0.0, 1.2, 1.0, 1.0)},
    "bgk_dec": {"collision_scheme": "bgk", "lambda_v": LAMBDA_V_OPT, "tau_gas": None},
    "bgk_dec_g12": {"collision_scheme": "bgk", "lambda_v": LAMBDA_V_OPT, "tau_gas": 1.2},
    "mrt_dec": {
        "collision_scheme": "mrt",
        "k_diag": (0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0),
        "lambda_v": LAMBDA_V_OPT,
        "tau_gas": None,
    },
}

#: Follow-up of the screen, which found the bulk rate ``s_e`` (entry 1) to be the lever:
#: s_e = 1.5 / 1.0 / 0.5 failed at 3000 / 21000 / >80000 steps for the double-well. The
#: bulk viscosity is ``cs2 (1/s_e - 1/2)``. These separate ``s_e`` from ``s_eps`` (entry 2),
#: go below 0.5, and combine the low bulk rate with the decoupled viscosity.
BULK_COLLISIONS: dict[str, dict[str, Any]] = {
    "mrt_e05": {"collision_scheme": "mrt", "k_diag": (0.0, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0)},
    "mrt_e03": {"collision_scheme": "mrt", "k_diag": (0.0, 0.3, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0)},
    "mrt_e02": {"collision_scheme": "mrt", "k_diag": (0.0, 0.2, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0)},
    "mrt_eps06": {"collision_scheme": "mrt", "k_diag": (0.0, 1.0, 0.6, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0)},
    "mrt_e03_dec": {
        "collision_scheme": "mrt",
        "k_diag": (0.0, 0.3, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0),
        "lambda_v": LAMBDA_V_OPT,
        "tau_gas": None,
    },
}
ALL_COLLISIONS = COLLISIONS | BULK_COLLISIONS

#: Liquid sound speed every cubic EOS is scaled to — the baseline double-well's.
C_L = 0.16
WIDTH = 5.0

#: Literature parameter sets at ratio 1000: ``label -> (eos, a, b, c_l)``. ``c_l = None``
#: keeps the published ``a`` (liquid sound speed ~0.95 for CS, ~1.13 for vdW, above the
#: lattice's 0.577); otherwise ``a`` is rescaled to that ``c_l`` at the same densities.
_CS_A, _CS_B = 1.0, 4.0  # Yuan & Schaefer
_VDW_A, _VDW_B = 9.0 / 392.0, 2.0 / 21.0  # Zhang, Guo & Wang 2022
LITERATURE_EOS: dict[str, tuple[str, float, float, float | None]] = {
    "cs_lit": ("carnahan-starling", _CS_A, _CS_B, None),
    "cs_lit_c03": ("carnahan-starling", _CS_A, _CS_B, 0.3),
    "cs_lit_c016": ("carnahan-starling", _CS_A, _CS_B, 0.16),
    "vdw_lit": ("van-der-waals", _VDW_A, _VDW_B, None),
    "vdw_lit_c03": ("van-der-waals", _VDW_A, _VDW_B, 0.3),
    "vdw_lit_c016": ("van-der-waals", _VDW_A, _VDW_B, 0.16),
}


@dataclass(frozen=True)
class Case:
    """One run: overrides on the baseline config plus its step count."""

    name: str
    phase: str
    nt: int
    overrides: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


def _eos_overrides(eos: str, ratio: float = 1000.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Config overrides and descriptive info for one EOS label at *ratio*."""
    from examples.density_ratio_study.eos_parameters import cubic_eos
    from examples.density_ratio_study.eos_parameters import double_well

    if eos == "dw":
        point = double_well(1.0, 1.0 / ratio, 0.04, 5)
    elif eos == "dw_stiff":
        point = double_well(1.0, 1.0 / ratio, 0.16, 5)
    elif eos == "dw_soft":
        point = double_well(1.0, 1.0 / ratio, 0.01, 5)
    elif eos == "dw_w8":
        # Wider interface at the baseline's c_l = sqrt(16 kappa / W^2) = 0.16.
        point = double_well(1.0, 1.0 / ratio, 0.04 * (8 / 5) ** 2, 8)
    elif eos == "vdw":
        point = cubic_eos("van-der-waals", ratio, 1.0, C_L, WIDTH)
    elif eos == "cs":
        point = cubic_eos("carnahan-starling", ratio, 1.0, C_L, WIDTH)
    elif eos in LITERATURE_EOS:
        name, a, b, c_l = LITERATURE_EOS[eos]
        # Literature densities (fixed b); a native (None -> use a) or rescaled to c_l.
        point = cubic_eos(name, ratio, None, c_l, WIDTH, a=None if c_l is not None else a, b=b)
    else:
        msg = f"unknown EOS label {eos!r}"
        raise ValueError(msg)
    return dict(point.config), point.row()


def _collision_overrides(label: str, tau: float) -> dict[str, Any]:
    overrides = dict(ALL_COLLISIONS[label])
    if "tau_gas" in overrides and overrides["tau_gas"] is None:
        overrides["tau_gas"] = tau
    return overrides


def _gravity(force_g: float) -> dict[str, Any]:
    return {"gravity_force": {"inclination_angle_deg": 0.0, "force_g": force_g}}


def diagnose_cases() -> list[Case]:
    nt = 60_000
    dw, dw_info = _eos_overrides("dw")
    stiff, stiff_info = _eos_overrides("dw_stiff")
    cases = [
        Case("dw_bgk_r1000", "diagnose", nt, dw, dw_info),
        Case("dw_bgk_r1000_g0", "diagnose", nt, dw | _gravity(0.0), dw_info),
        Case("dw_stiff_bgk_r1000", "diagnose", nt, stiff, stiff_info),
    ]
    for label in ("dw_soft", "dw_w8"):
        eos, info = _eos_overrides(label)
        cases.append(Case(f"{label}_bgk_r1000", "diagnose", nt, eos, info))
    for ratio in (10, 30, 100, 300):
        eos, info = _eos_overrides("dw", ratio)
        cases.append(Case(f"dw_bgk_r{ratio}", "diagnose", nt, eos, info))
    return cases


def screen_cases(tau: float = 0.99) -> list[Case]:
    cases = []
    for eos in ("dw", "dw_stiff", "vdw", "cs"):
        eos_over, info = _eos_overrides(eos)
        for collision in COLLISIONS:
            overrides = eos_over | _collision_overrides(collision, tau)
            cases.append(
                Case(f"{eos}__{collision}", "screen", SCREEN_STEPS, overrides, info | {"collision": collision})
            )
    return cases


def bulk_cases(tau: float = 0.99) -> list[Case]:
    """The bulk-rate follow-up at the screen's conditions, for the three base EOSs."""
    cases = []
    for eos in ("dw", "vdw", "cs"):
        eos_over, info = _eos_overrides(eos)
        for collision in BULK_COLLISIONS:
            overrides = eos_over | _collision_overrides(collision, tau)
            cases.append(Case(f"{eos}__{collision}", "bulk", SCREEN_STEPS, overrides, info | {"collision": collision}))
    return cases


def init_cases() -> list[Case]:
    """Does the start-up transient decide the EOS comparison?

    Every screen case starts from ``tanh((d - R) / 5)``, twice the double-well's
    equilibrium ``tanh(2x / 5)`` and further still from a cubic EOS's asymmetric
    profile. vdW and CS die within 1000-7000 steps with a 6-30% area loss first; if
    their failure time moves with the initial width, the screen's EOS axis measured
    the initialisation, not the scheme.
    """
    cases = []
    for eos in ("dw", "vdw", "cs"):
        eos_over, info = _eos_overrides(eos)
        for width in (3, 8):
            overrides = eos_over | {"interface_width": width}
            cases.append(Case(f"{eos}__bgk__iw{width}", "init", 30_000, overrides, info | {"init_width": width}))
    return cases


def _mrt_bulk(s_e: float) -> dict[str, Any]:
    return {"collision_scheme": "mrt", "k_diag": (0.0, s_e, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0)}


def refine_cases() -> list[Case]:
    """Double-well only: map the bulk-rate window, then stack the single-factor gains.

    The bulk phase found ``s_e = 0.5`` best (77750 steps), ``s_e <= 0.3`` diverging in
    the first 250 steps and ``s_e = 1`` failing at 21000. The init phase found a wider
    initial profile (``interface_width = 8``) to help the double-well, and the
    diagnose phase a lower surface tension (``dw_soft``) and no gravity.
    """
    dw, info = _eos_overrides("dw")
    cases = [
        Case(f"dw__mrt_e{s_e:g}", "refine", SCREEN_STEPS, dw | _mrt_bulk(s_e), info | {"s_e": s_e})
        for s_e in (0.4, 0.45, 0.6, 0.7)
    ]
    soft, soft_info = _eos_overrides("dw_soft")
    stacked = {
        "dw__mrt_e0.5__iw8": (dw | {"interface_width": 8}, info),
        "dw_soft__mrt_e0.5": (soft, soft_info),
        "dw_soft__mrt_e0.5__iw8": (soft | {"interface_width": 8}, soft_info),
        "dw__mrt_e0.5__g0": (dw | _gravity(0.0), info),
    }
    for name, (eos_over, eos_info) in stacked.items():
        cases.append(Case(name, "refine", SCREEN_STEPS, eos_over | _mrt_bulk(0.5), eos_info | {"s_e": 0.5}))
    return cases


def matched_cases(survivors: list[str], bond: float = 0.123003, ohnesorge: float = 0.442476) -> list[Case]:
    """Rerun *survivors* (screen case names) at the baseline's Bo and Oh.

    With ``L = sqrt(A/pi)`` of the initial bubble and ``rho_l = 1``:
    ``g = Bo sigma / (drho L^2)``, ``nu = Oh sqrt(rho_l sigma L)``, ``tau = 1/2 + 3 nu``.
    """
    base = _load_baseline()
    # The initialiser scales radii by min(nx, ny); L_eff = sqrt(A/pi) is that radius.
    length = base["initialisation"]["radii"][0] * min(base["grid_shape"][:2])
    cases = []
    for name in survivors:
        eos, collision = name.split("__")
        eos_over, info = _eos_overrides(eos)
        sigma = info["sigma"]
        drho = info["rho_l"] - info["rho_v"]
        g = bond * sigma / (drho * length**2)
        tau = 0.5 + 3.0 * ohnesorge * math.sqrt(info["rho_l"] * sigma * length)
        overrides = eos_over | _collision_overrides(collision, tau) | {"tau": tau} | _gravity(g)
        cases.append(
            Case(
                f"{name}__matched",
                "matched",
                SCREEN_STEPS,
                overrides,
                info | {"collision": collision, "g": g, "tau": tau},
            )
        )
    return cases


def _load_baseline() -> dict[str, Any]:
    from src.config import TomlAdapter

    raw = TomlAdapter().load_raw(str(BASELINE_CONFIG))
    for key in ("results_dir", "save_fields", "plot_fields", "output_format", "save_interval", "skip_interval"):
        raw.pop(key, None)
    return raw


def _phase_label(rho: float, rho_v: float, rho_l: float) -> str:
    frac = (rho - rho_v) / (rho_l - rho_v)
    if frac < 0.1:  # noqa: PLR2004
        return "vapour"
    if frac > 0.9:  # noqa: PLR2004
        return "liquid"
    return "interface"


def run_case(case: Case) -> dict[str, Any]:  # noqa: PLR0915 - one linear driver loop
    """Run *case*, write its ``run.npz``, return its summary row."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.config import DictAdapter
    from src.config.jax_config import configure_jax
    from src.operators.macroscopic.eos import build_pressure_fn
    from src.pipeline.runner import init_state
    from src.pipeline.setup import build_setup
    from src.simulation_io.analysis.stability import compute_stability_metrics
    from src.simulation_io.analysis.stability import local_mach

    configure_jax()
    out = RESULTS / case.phase / case.name
    out.mkdir(parents=True, exist_ok=True)
    raw = _load_baseline() | {"nt": case.nt} | case.overrides
    summary: dict[str, Any] = {"case": case.name, "phase": case.phase, "nt": case.nt} | case.info
    try:
        config = DictAdapter().load(raw)
    except ValueError as exc:
        summary |= {"status": "rejected", "reason": str(exc)}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return summary

    setup = build_setup(config)
    mp = setup.multiphase_params
    assert mp is not None  # noqa: S101 - every case is multiphase
    step_fn = setup.step_fn
    assert step_fn is not None  # noqa: S101 - every sim_type resolves a step operator
    pressure_fn = build_pressure_fn(mp)
    rho_v, rho_l = float(mp.rho_v), float(mp.rho_l)
    rho_mid = 0.5 * (rho_v + rho_l)

    advance = jax.jit(lambda s: jax.lax.fori_loop(0, SAMPLE_INTERVAL, lambda _, s: step_fn(setup, s), s))
    single = jax.jit(lambda s: step_fn(setup, s))

    @jax.jit
    def sample(state: State) -> dict[str, jnp.ndarray]:
        metrics = compute_stability_metrics(
            state, gradient_density=setup.gradient_density, mp=mp, pressure_fn=pressure_fn
        )
        rho = state.rho[:, :, 0, 0, 0]
        gas = rho < rho_mid
        n_gas = jnp.maximum(jnp.sum(gas), 1)
        x, y = jnp.meshgrid(jnp.arange(rho.shape[0]), jnp.arange(rho.shape[1]), indexing="ij")
        pure_vapour = rho < rho_v + 0.1 * (rho_l - rho_v)
        uy = state.u[:, :, 0, 0, 1]
        return {
            "metrics": metrics,
            "mass": jnp.sum(state.rho),
            "gas_area": jnp.sum(gas),
            "gas_mass": jnp.sum(jnp.where(gas, rho, 0.0)),
            "gas_xc": jnp.sum(jnp.where(gas, x, 0)) / n_gas,
            "gas_yc": jnp.sum(jnp.where(gas, y, 0)) / n_gas,
            "gas_uy": jnp.sum(jnp.where(gas, uy, 0.0)) / n_gas,
            "vapour_rho_mean": jnp.sum(jnp.where(pure_vapour, rho, 0.0)) / jnp.maximum(jnp.sum(pure_vapour), 1),
            "vapour_rho_min": jnp.min(jnp.where(pure_vapour, rho, jnp.inf)),
            "mach_all_max": jnp.max(local_mach(state.rho, state.u, pressure_fn)),
        }

    keys = ("t", "max_u", "max_grad_mu", "rho_min", "rho_max", "cb_amp", "n_wake", "stripe_amp", "vapour_mach")
    extra = (
        "mass",
        "gas_area",
        "gas_mass",
        "gas_xc",
        "gas_yc",
        "gas_uy",
        "vapour_rho_mean",
        "vapour_rho_min",
        "mach_all_max",
    )
    series: dict[str, list[float]] = {k: [] for k in keys + extra}

    def record(t: int, s: dict[str, Any]) -> bool:
        m = np.asarray(s["metrics"], dtype=np.float64)
        for k, v in zip(keys, [float(t), *m], strict=True):
            series[k].append(v)
        for k in extra:
            series[k].append(float(s[k]))
        return bool(np.isfinite(m).all()) and float(m[2]) > 0.0

    state = init_state(setup)
    frame_every = max(1, case.nt // SAMPLE_INTERVAL // N_FRAMES)
    frames, frame_t = [np.asarray(state.rho)[:, :, 0, 0, 0].astype(np.float32)], [0]
    record(0, sample(state))
    t, status, failure = 0, "survived", {}
    start = time.time()
    n_chunks = case.nt // SAMPLE_INTERVAL
    for k in range(1, n_chunks + 1):
        prev = state
        state = advance(state)
        t = k * SAMPLE_INTERVAL
        if not record(t, sample(state)):
            status = "diverged"
            failure = _locate_failure(prev, t - SAMPLE_INTERVAL, single, rho_v, rho_l, pressure_fn)
            frames.append(np.asarray(prev.rho)[:, :, 0, 0, 0].astype(np.float32))
            frame_t.append(t - SAMPLE_INTERVAL)
            break
        if t % PROGRESS_INTERVAL == 0:
            print(
                f"[{case.name}] t={t} stripe={series['stripe_amp'][-1]:.3g} "
                f"Ma_vap={series['vapour_mach'][-1]:.3g} rho_min={series['rho_min'][-1]:.3g} "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )
        if k % frame_every == 0:
            frames.append(np.asarray(state.rho)[:, :, 0, 0, 0].astype(np.float32))
            frame_t.append(t)
    elapsed = time.time() - start

    arrays = {k: np.asarray(v) for k, v in series.items()}
    # Samples up to the last healthy one: finite metrics and a positive density.
    finite = np.isfinite(arrays["rho_min"]) & (arrays["rho_min"] > 0.0) & np.isfinite(arrays["stripe_amp"])
    summary |= {
        "status": status
        if status == "diverged" or arrays["gas_area"][finite][-1] >= BUBBLE_LOST_FRACTION * arrays["gas_area"][0]
        else "bubble_lost",
        "t_end": t if status == "diverged" else case.nt,
        "elapsed_s": round(elapsed, 1),
        "max_vapour_mach": float(np.nanmax(arrays["vapour_mach"][finite])),
        "max_u": float(np.nanmax(arrays["max_u"][finite])),
        "rho_min": float(np.nanmin(arrays["rho_min"][finite])),
        "max_stripe": float(np.nanmax(arrays["stripe_amp"][finite])),
        "mass_drift": float(arrays["mass"][finite][-1] / arrays["mass"][0] - 1.0),
        "gas_area_ratio": float(arrays["gas_area"][finite][-1] / arrays["gas_area"][0]),
        "gas_rise": float(arrays["gas_yc"][finite][-1] - arrays["gas_yc"][0]),
        "tau": config.tau,
        "relaxation_time": config.relaxation_time,
    } | {f"fail_{k}": v for k, v in failure.items()}
    payload: dict[str, Any] = {f"series_{k}": v for k, v in arrays.items()}
    payload |= {
        "rho": np.stack(frames),
        "frame_t": np.asarray(frame_t),
        "summary": json.dumps(summary, default=str),
        "config": json.dumps(raw, default=str),
    }
    np.savez_compressed(out / "run.npz", **payload)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def _locate_failure(
    state: State,
    t0: int,
    single: Callable[[State], State],
    rho_v: float,
    rho_l: float,
    pressure_fn: EosOperator,
) -> dict[str, Any]:
    """Step one at a time from the last good sample to the first non-finite or non-positive density.

    Reports the worst cell of the *last* healthy state — the cell whose local
    Mach number is highest — since by the first bad step the defect has spread.
    """
    import numpy as np
    from src.simulation_io.analysis.stability import local_mach

    healthy_steps = 0
    while healthy_steps < SAMPLE_INTERVAL:
        nxt = single(state)
        rho = np.asarray(nxt.rho)[:, :, 0, 0, 0]
        if not np.isfinite(rho).all() or rho.min() <= 0.0:
            break
        state = nxt
        healthy_steps += 1
    rho = np.asarray(state.rho)[:, :, 0, 0, 0]
    u = np.asarray(state.u)[:, :, 0, 0, :]
    mach = np.asarray(local_mach(state.rho, state.u, pressure_fn))[:, :, 0, 0, 0]
    i, j = np.unravel_index(int(np.argmax(mach)), mach.shape)
    iu, ju = np.unravel_index(int(np.argmax(np.hypot(u[..., 0], u[..., 1]))), rho.shape)
    imin, jmin = np.unravel_index(int(np.argmin(rho)), rho.shape)
    ny = rho.shape[1]
    return {
        "t_last_ok": t0 + healthy_steps,
        "mach_cell": (int(i), int(j)),
        "mach_max": float(mach[i, j]),
        "mach_cell_rho": float(rho[i, j]),
        "mach_cell_phase": _phase_label(float(rho[i, j]), rho_v, rho_l),
        "umax_cell": (int(iu), int(ju)),
        "umax_cell_phase": _phase_label(float(rho[iu, ju]), rho_v, rho_l),
        "rhomin_cell": (int(imin), int(jmin)),
        "rhomin": float(rho[imin, jmin]),
        "rhomin_near_wall": bool(min(jmin, ny - 1 - jmin) < 3),  # noqa: PLR2004
    }


def _run_all(cases: list[Case], workers: int) -> list[dict[str, Any]]:
    rows = []
    if workers <= 1:
        for case in cases:
            rows.append(_safe_run(case))
            print(json.dumps(rows[-1], default=str), flush=True)
        return rows
    ctx = multiprocessing.get_context("spawn")  # JAX is not fork-safe
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(_safe_run, case): case for case in cases}
        for future in as_completed(futures):
            rows.append(future.result())
            print(json.dumps(rows[-1], default=str), flush=True)
    return rows


def _safe_run(case: Case) -> dict[str, Any]:
    try:
        return run_case(case)
    except Exception as exc:  # noqa: BLE001 - one broken case must not stop the phase
        return {
            "case": case.name,
            "phase": case.phase,
            "status": "error",
            "reason": repr(exc),
            "tb": traceback.format_exc(),
        }


def _survivors() -> list[str]:
    names = []
    for summary in sorted([*(RESULTS / "screen").glob("*/summary.json"), *(RESULTS / "bulk").glob("*/summary.json")]):
        row = json.loads(summary.read_text())
        if row.get("status") == "survived":
            names.append(row["case"])
    return names


def literature_cases() -> list[Case]:
    """Van der Waals and Carnahan-Starling at their published ``(a, b)``, and rescaled ``a``."""
    collisions = {"bgk": COLLISIONS["bgk"], "mrt_e05": _mrt_bulk(0.5), "mrt_e06": _mrt_bulk(0.6)}
    cases = []
    for label in LITERATURE_EOS:
        eos_over, info = _eos_overrides(label)
        for name, collision in collisions.items():
            overrides = eos_over | collision
            cases.append(Case(f"{label}__{name}", "literature", SCREEN_STEPS, overrides, info | {"collision": name}))
    return cases


#: Phases whose case list does not depend on earlier results.
PHASES = {
    "diagnose": diagnose_cases,
    "screen": screen_cases,
    "bulk": bulk_cases,
    "init": init_cases,
    "refine": refine_cases,
    "literature": literature_cases,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Density-ratio 1000 rising-bubble stability study.")
    parser.add_argument("phase", choices=[*PHASES, "matched", "case", "list"])
    parser.add_argument("names", nargs="*", help="case names (for 'case'), or a filter on the phase's cases")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nt", type=int, default=None, help="override the step count (smoke tests)")
    args = parser.parse_args()

    every = [case for build in PHASES.values() for case in build()]
    if args.phase == "list":
        for case in every:
            print(case.phase, case.name)
        return
    if args.phase == "matched":
        cases = matched_cases(args.names or _survivors())
    else:
        cases = every if args.phase == "case" else PHASES[args.phase]()
        if args.names or args.phase == "case":
            cases = [c for c in cases if c.name in args.names]
    if args.nt is not None:
        cases = [Case(c.name, c.phase, args.nt, c.overrides, c.info) for c in cases]
    _run_all(cases, args.workers)


if __name__ == "__main__":
    main()
