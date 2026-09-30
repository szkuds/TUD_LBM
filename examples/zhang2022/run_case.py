"""Validation runs of Zhang, Guo & Wang, Phys. Fluids 34, 012110 (2022).

Every lattice parameter is derived from the paper's dimensionless numbers, so a
case is fully stated by the constants below. Van der Waals fluid,
``a = 9/392``, ``b = 2/21``, ``R = 1``, ``T = 0.8 T_c``, ``kappa = 0.01``,
``sigma = 0.08489`` (the paper's Laplace-law value; the planar-interface
integral of this EOS gives 0.08556).

Cases (``python examples/zhang2022/run_case.py <case> [--lambda-v X] [--mo X]``):

* ``collision1`` / ``collision2`` / ``collision3`` — Table I, 801 x 1601, ``r = 100``,
  ``lambda_v = 0.8``, ``rho_l nu_l / (rho_g nu_g) = 100``, fully periodic.
* ``collision3_re2000`` — case 3 at ``Re = 2000`` (Fig. 8).
* ``bubble`` — rising bubble, 8R x 14R, ``R = 30`` at ``(4R, 3R)``, ``Eo = 1``,
  ``Mo = 1e-5`` (override with ``--mo``), ``rho_g nu_g / (rho_l nu_l) = 0.01``,
  ``rho_0 = rho_l``, ``lambda_v = 0.78`` (override with ``--lambda-v``), periodic
  sides and no-slip (bounce-back) top/bottom. ``d_e = 2R``.

No explicit spatial filter is applied (the paper filters rho and u in the
collision cases, Sec. III C, without specifying it).

The CLI's run directories always store ``f`` (~125 MB per 801 x 1601
snapshot), so this driver keeps only float32 density frames and the scalar
diagnostics in one compressed ``.npz`` per run, plus a PNG montage.
"""

from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import jax
import numpy as np
from src.config import DictAdapter
from src.config.jax_config import configure_jax
from src.pipeline.runner import init_state
from src.pipeline.setup import build_setup

configure_jax()

A_EOS, B_EOS = 9.0 / 392.0, 2.0 / 21.0
T = 0.8 / 14.0
KAPPA = 0.01
SIGMA = 0.08489
RHO_L, RHO_G = 6.7645, 0.8388
CS2 = 1.0 / 3.0
MRT_FREE_RATES = (0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0)
RESULTS = Path("~/TUD_LBM_data/zhang2022").expanduser()

#: Table I: (We, Re, B).
COLLISIONS = {
    "collision1": (0.2, 14.8, 0.2),
    "collision2": (8.6, 105.9, 0.08),
    "collision3": (60.8, 313.7, 0.68),
    "collision3_re2000": (60.8, 2000.0, 0.68),
}
#: Steps per case, in units of r / U0 (collision) — long enough for the regime to show.
COLLISION_DURATION = {"collision1": 1.1, "collision2": 5.0, "collision3": 12.0, "collision3_re2000": 12.0}


def _tau(nu: float) -> float:
    return 0.5 + nu / CS2


def _base(grid_shape: tuple[int, int], tau_l: float, tau_g: float, lambda_v: float) -> dict[str, object]:
    return {
        "sim_type": "multiphase",
        "grid_shape": grid_shape,
        "tau": tau_l,
        "tau_gas": tau_g,
        "lambda_v": lambda_v,
        "collision_scheme": "mrt",
        "k_diag": MRT_FREE_RATES,
        "eos": "van-der-waals",
        "a_eos": A_EOS,
        "b_eos": B_EOS,
        "r_eos": 1.0,
        "t_eos": T,
        "kappa": KAPPA,
        "rho_l": RHO_L,
        "rho_v": RHO_G,
        "interface_width": 2,  # tanh((d - R)/2) == the paper's tanh(2(R - d)/W), W = 4
    }


def collision_case(name: str) -> tuple[dict[str, object], dict[str, float], int]:
    we, re, b = COLLISIONS[name]
    r = 100.0
    u0 = math.sqrt(we * SIGMA / (8.0 * r * RHO_L))
    nu_l = 4.0 * r * u0 / re
    nu_g = RHO_L * nu_l / (100.0 * RHO_G)
    config = _base((1601, 801), _tau(nu_l), _tau(nu_g), 0.8)
    config |= {
        "init_type": "binary_collision",
        "initialisation": {"radius": r, "speed": u0, "impact_parameter": b, "gap": 20.0},
    }
    params = {"We": we, "Re": re, "B": b, "r": r, "U0": u0, "nu_l": nu_l, "nu_g": nu_g}
    return config, params, round(COLLISION_DURATION[name] * r / u0)


def bubble_case(mo: float, lambda_v: float) -> tuple[dict[str, object], dict[str, float], int]:
    radius = 30
    d_e = 2.0 * radius
    eo = 1.0
    g = eo * SIGMA / ((RHO_L - RHO_G) * d_e**2)
    re = eo**0.75 / mo**0.25
    nu_l = math.sqrt(g * (1.0 - RHO_G / RHO_L) * d_e**3) / re
    nu_g = 0.01 * RHO_L * nu_l / RHO_G
    nx, ny = 8 * radius, 14 * radius
    config = _base((nx, ny), _tau(nu_l), _tau(nu_g), lambda_v)
    config |= {
        "init_type": "multiphase_bubbles",
        "initialisation": {
            "centres": [[0.5, 3 * radius / (ny - 1)]],
            "radii": [radius / min(nx, ny)],
            "dispersed": "vapour",
        },
        "bc_config": {"top": "bounce-back", "bottom": "bounce-back", "left": "periodic", "right": "periodic"},
        "gravity_force": {"force_g": g, "reference_density": RHO_L},
    }
    t_star = math.sqrt(2.0 * radius / g)
    params = {
        "Eo": eo,
        "Mo": mo,
        "Re": re,
        "R": radius,
        "d_e": d_e,
        "g": g,
        "nu_l": nu_l,
        "nu_g": nu_g,
        "t_star": t_star,
    }
    return config, params, round(12.0 * t_star)


def bubble_diagnostics(rho: np.ndarray) -> tuple[float, float, float]:
    """Area, centroid ``(x, y)`` of the gas region ``rho < (rho_l + rho_g)/2``."""
    gas = rho < (RHO_L + RHO_G) / 2.0
    x, y = np.nonzero(gas)
    return float(gas.sum()), float(x.mean()), float(y.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Zhang, Guo & Wang (2022) validation runs.")
    parser.add_argument("case", choices=[*COLLISIONS, "bubble"])
    parser.add_argument("--lambda-v", type=float, default=0.78)
    parser.add_argument("--mo", type=float, default=1e-5)
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()

    if args.case == "bubble":
        raw, params, nt = bubble_case(args.mo, args.lambda_v)
        tag = f"bubble_mo{args.mo:.0e}_lv{args.lambda_v:g}"
    else:
        raw, params, nt = collision_case(args.case)
        tag = args.case
    out = RESULTS / tag
    out.mkdir(parents=True, exist_ok=True)

    config = DictAdapter().load(raw)
    setup = build_setup(config)
    state = init_state(setup)
    chunk = max(1, nt // args.frames)
    step_fn = setup.step_fn
    assert step_fn is not None  # noqa: S101 - every sim_type resolves a step operator
    advance = jax.jit(lambda s: jax.lax.fori_loop(0, chunk, lambda _, s: step_fn(setup, s), s))

    frames, times, diagnostics = [np.asarray(state.rho)[:, :, 0, 0, 0].astype(np.float32)], [0], []
    mass0 = float(np.sum(np.asarray(state.f)))
    print(f"[{tag}] nt={nt} chunk={chunk} tau_l={config.tau:.5f} tau_g={config.tau_gas:.5f} {json.dumps(params)}")
    start = time.time()
    for k in range(1, args.frames + 1):
        state = advance(state)
        rho = np.asarray(state.rho)[:, :, 0, 0, 0]
        u = np.asarray(state.u)[:, :, 0, 0, :]
        if not np.isfinite(rho).all():
            print(f"[{tag}] diverged by t={k * chunk}")
            break
        frames.append(rho.astype(np.float32))
        times.append(k * chunk)
        row = {
            "t": k * chunk,
            "max_u": float(np.abs(u).max()),
            "mass_drift": float(np.sum(np.asarray(state.f))) / mass0 - 1.0,
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
        }
        if args.case == "bubble":
            row["area"], row["x_c"], row["y_c"] = bubble_diagnostics(rho)
            gas = rho < (RHO_L + RHO_G) / 2.0
            row["u_bubble"] = float(u[..., 1][gas].mean())
        diagnostics.append(row)
        print(f"[{tag}] {k}/{args.frames} {json.dumps(row)} ({time.time() - start:.0f}s)", flush=True)

    np.savez_compressed(
        out / "run.npz",
        rho=np.stack(frames),
        t=np.asarray(times),
        diagnostics=json.dumps(diagnostics),
        params=json.dumps(params | {"tau_l": config.tau, "tau_g": config.tau_gas, "lambda_v": config.relaxation_time}),
    )
    print(f"[{tag}] done in {time.time() - start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
