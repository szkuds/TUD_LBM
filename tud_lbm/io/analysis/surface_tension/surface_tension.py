"""Numerical surface-tension calibration via the Young-Laplace relation.

Some equations of state have no closed-form surface tension in terms of the
simulation parameters (Carnahan-Starling). For those, the lattice surface
tension is measured directly: periodic droplets of several radii are
equilibrated, the Laplace pressure jump is read from each, and a line is
fitted to ``dP = sigma / R`` (2-D Young-Laplace).

During ``tud-lbm run`` the measurement is triggered automatically only for
EOS without a closed form (Carnahan-Starling); ``tud-lbm analyse
CONFIG.toml --surface-tension`` forces it for any supported multiphase EOS,
e.g. to verify the closed-form double-well sigma numerically.

The measurement is expensive, so results are cached on disk keyed by the
thermodynamic parameters that determine sigma. The cache file lives at
``tud_lbm/io/analysis/surface_tension/data/surface_tension_cache.json`` — inside the repo, so
it's shared with the team via the normal git workflow rather than re-measured
by everyone individually (commit it after adding a new entry). The
calibration figure is written into the active run directory on every run,
whether measured or served from cache, so each run keeps its own copy.
"""

from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast
import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable
    from tud_lbm.config import SimulationConfig
    from tud_lbm.operators.macroscopic import MultiphaseParams
    from tud_lbm.pipeline.setup import SimulationSetup
    from tud_lbm.pipeline.state.state import State

# EOS whose surface tension must be measured rather than derived analytically.
_EOS_REQUIRING_CALIBRATION = frozenset({"carnahan-starling"})

_KNOWN_EOS = frozenset({"double-well", "carnahan-starling"})

_N_RADII = 5
_N_ITERATIONS = 200_000
_PERIODIC_BC = {"top": "periodic", "bottom": "periodic", "left": "periodic", "right": "periodic"}

_CACHE_FILENAME = "surface_tension_cache.json"
_PLOT_FILENAME = "surface_tension_calibration.png"

# Git-tracked, shared across the team: a measured sigma committed here is
# picked up by everyone on the next `git pull`, instead of each person
# re-running the ~40-minute droplet sweep. Sharing a new entry still requires
# an explicit `git add/commit/push` — writing to this file only updates your
# local working tree.
_SHARED_CACHE_PATH = Path(__file__).resolve().parent / "data" / _CACHE_FILENAME

# Parameters that uniquely determine the measured surface tension.
_CACHE_KEYS = ("eos", "kappa", "rho_l", "rho_v", "interface_width", "a_eos", "b_eos", "r_eos", "t_eos")

console = Console()


def record_surface_tension(config: SimulationConfig, run_dir: str | Path) -> SimulationConfig:
    """Measure sigma when the EOS needs it, refresh the parameter file, return updated config.

    For an EOS with a closed-form surface tension the config is returned
    unchanged. Otherwise sigma is measured (or read from cache), stored in
    ``config.extra['surface_tension']``, and ``physical_parameters.txt`` is
    rewritten in *run_dir* with the measured value.
    """
    if not (config.is_multiphase and config.eos in _EOS_REQUIRING_CALIBRATION):
        return config

    from tud_lbm.io.analysis.physical_parameters import write_physical_parameters

    sigma = calibrate_surface_tension(config, run_dir)
    updated = replace(config, extra={**config.extra, "surface_tension": sigma})
    write_physical_parameters(updated, Path(run_dir) / "physical_parameters.txt")
    return updated


def calibrate_surface_tension(config: SimulationConfig, run_dir: str | Path) -> float:
    """Return the measured lattice surface tension and write the calibration figure.

    Looks up a cached value keyed by the EOS thermodynamic parameters; on a
    miss, runs the droplet sweep and caches the result. The figure is always
    written into *run_dir*.
    """
    run_dir = Path(run_dir)
    cache_path = _cache_path()
    key = _cache_key(config)

    cached = _load_cache(cache_path).get(key)
    if cached is not None:
        radii = np.asarray(cached["radii"], dtype=float)
        delta_p = np.asarray(cached["delta_p"], dtype=float)
        sigma = float(cached["sigma"])
    else:
        console.print(
            f"[dim]No cached σ for these EOS parameters — running "
            f"Young–Laplace calibration ({_N_RADII} droplets)...[/dim]"
        )
        radii, delta_p = _measure_pressure_jumps(config)
        sigma = _fit_sigma(radii, delta_p)
        _store_cache(key, radii, delta_p, sigma)
        console.print(f"[bold green]Surface tension calibrated: σ = {sigma:.6g}[/bold green]")

    _save_plot(run_dir / _PLOT_FILENAME, radii, delta_p, sigma)
    return sigma


# ── Measurement ───────────────────────────────────────────────────────


def _bulk_pressure_fn(mp: MultiphaseParams) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``pressure(rho) -> p_0`` for the EOS bound in *mp*.

    Only the bulk pressure differs between EOS in the Young-Laplace
    measurement; everything else in the droplet sweep is EOS-agnostic.
    """
    if mp.eos == "carnahan-starling":
        from tud_lbm.operators.macroscopic.eos import carnahan_starling_pressure

        if mp.a_eos is None or mp.b_eos is None or mp.r_eos is None or mp.t_eos is None:
            msg = "a_eos, b_eos, r_eos, t_eos are required for Carnahan-Starling calibration"
            raise ValueError(msg)
        a_eos, b_eos, r_eos, t_eos = mp.a_eos, mp.b_eos, mp.r_eos, mp.t_eos
        return lambda rho: np.asarray(carnahan_starling_pressure(rho, a_eos, b_eos, r_eos, t_eos))

    if mp.eos == "double-well":
        from tud_lbm.operators.macroscopic.eos import double_well_pressure

        beta = 8.0 * mp.kappa / (float(mp.interface_width) ** 2 * (mp.rho_l - mp.rho_v) ** 2)
        return lambda rho: np.asarray(double_well_pressure(rho, beta, mp.rho_l, mp.rho_v))

    supported = ", ".join(sorted(_KNOWN_EOS))
    msg = f"surface-tension calibration supports EOS {supported}; got '{mp.eos}'"
    raise ValueError(msg)


def _measure_pressure_jumps(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Equilibrate one droplet per radius and return ``(radii, delta_p)``."""
    from tud_lbm.operators.initialise import build_initialise_fn
    from tud_lbm.operators.macroscopic import build_multiphase_params
    from tud_lbm.pipeline.runner import init_state
    from tud_lbm.pipeline.setup import build_setup

    if config.interface_width is None or config.rho_l is None or config.rho_v is None:
        msg = "interface_width, rho_l, rho_v are required for surface-tension calibration"
        raise ValueError(msg)

    nx, ny = int(config.grid_shape[0]), int(config.grid_shape[1])
    min_dim = min(nx, ny)
    radii = np.linspace(min_dim / 4.0, min_dim / 2.0, _N_RADII)
    width = int(config.interface_width)
    rho_l, rho_v = float(config.rho_l), float(config.rho_v)

    calib_config = _calibration_config(config)
    mp = build_multiphase_params(calib_config)
    pressure_fn = _bulk_pressure_fn(mp)

    setup = build_setup(calib_config)
    grid_shape = cast("tuple[int, int, int]", setup.grid_shape)
    init_fn = build_initialise_fn("multiphase_bubbles")

    delta_p = np.empty(_N_RADII)
    for i, radius in enumerate(radii):
        console.print(f"[dim]Calibration running ({i + 1}/{_N_RADII})...[/dim]")
        f0 = init_fn(
            grid_shape,
            setup.lattice,
            rho_l=rho_l,
            rho_v=rho_v,
            interface_width=width,
            centres=[[0.5, 0.5]],
            radii=[float(radius) / min_dim],
            dispersed="liquid",
        )
        final_state = _run_to_final_state(setup, init_state(setup, f=f0), _N_ITERATIONS)
        rho_2d = _density_2d(final_state)
        delta_p[i] = _pressure_jump(pressure_fn(rho_2d), width)

    return radii, delta_p


def _run_to_final_state(setup: SimulationSetup, state: State, nt: int) -> State:
    """Advance *nt* steps without stacking a trajectory.

    ``pipeline.runner.run()`` without an ``io_handler`` materializes the full
    per-step trajectory in memory (one stacked ``State`` per step) — for the
    droplet sweep's large ``nt`` that overflows memory long before it
    finishes. Only the final state is needed here, so the scan body discards
    its per-step output instead.
    """
    import jax

    if setup.step_fn is None:
        msg = "step_fn is required in SimulationSetup to run simulation"
        raise TypeError(msg)
    step_fn = setup.step_fn

    @jax.jit
    def scan_body(state: State, _t: int) -> tuple[State, None]:
        return step_fn(setup, state), None

    final_state, _ = jax.lax.scan(scan_body, state, jax.numpy.arange(nt))
    return final_state


def _calibration_config(config: SimulationConfig) -> SimulationConfig:
    """An isolated single-droplet config: periodic, no forces, no wetting."""
    return replace(
        config,
        sim_type="multiphase",
        bc_config=dict(_PERIODIC_BC),
        nt=_N_ITERATIONS,
        save_interval=0,
        skip_interval=0,
        save_fields=None,
        plot_fields=None,
        animate_fields=None,
        g=None,
        gravity_force=None,
        gravity_masked_force=None,
        electric_force=None,
        wetting_config=None,
        hysteresis_config=None,
        chemical_step_config=None,
        init_type="multiphase_bubbles",
        initialisation={"centres": [[0.5, 0.5]], "radii": [0.2], "dispersed": "liquid"},
        simulation_name=f"{config.simulation_name}_surface_tension",
    )


def _density_2d(state: State) -> np.ndarray:
    """Extract the 2-D density slice from a final :class:`State`."""
    import jax.numpy as jnp

    rho = state.rho if state.rho is not None else jnp.sum(state.f, axis=3, keepdims=True)
    return np.asarray(rho)[:, :, 0, 0, 0]


def _pressure_jump(pressure: np.ndarray, width: int) -> float:
    """Laplace jump: centre (liquid) minus the mean of four vapour corners."""
    nx, ny = pressure.shape
    p_inside = pressure[nx // 2, ny // 2]
    margin = 3 * width
    p_outside = np.mean(
        [
            pressure[margin, margin],
            pressure[margin, ny - margin - 1],
            pressure[nx - margin - 1, margin],
            pressure[nx - margin - 1, ny - margin - 1],
        ]
    )
    return float(p_inside - p_outside)


def _fit_sigma(radii: np.ndarray, delta_p: np.ndarray) -> float:
    """Surface tension is the slope of ``dP`` against ``1/R``."""
    slope, _ = np.polyfit(1.0 / radii, delta_p, deg=1)
    return float(slope)


# ── Cache ─────────────────────────────────────────────────────────────


def _cache_path() -> Path:
    return _SHARED_CACHE_PATH


def _cache_key(config: SimulationConfig) -> str:
    values = {k: getattr(config, k, None) for k in _CACHE_KEYS}
    return json.dumps(values, sort_keys=True)


def _sanitize_key(raw_key: str) -> str | None:
    """Rebuild a cache key from validated primitives, or reject it.

    Valid keys are the canonical JSON produced by :func:`_cache_key`: exactly
    the ``_CACHE_KEYS`` fields, with numeric or ``None`` values and an EOS
    name from ``_KNOWN_EOS``. The returned key is re-serialized from coerced
    primitives so nothing read from the cache file is echoed back verbatim.
    """
    try:
        values = json.loads(raw_key)
    except ValueError:
        return None
    if not isinstance(values, dict) or set(values) != set(_CACHE_KEYS):
        return None
    clean: dict[str, str | int | float | None] = {}
    for field in _CACHE_KEYS:
        value = values[field]
        if field == "eos":
            known = next((eos for eos in _KNOWN_EOS if eos == value), None)
            if value is not None and known is None:
                return None
            clean[field] = known
        elif value is None:
            clean[field] = None
        elif isinstance(value, int) and not isinstance(value, bool):
            clean[field] = int(value)
        elif isinstance(value, float):
            clean[field] = float(value)
        else:
            return None
    return json.dumps(clean, sort_keys=True)


def _sanitize_entry(raw_entry: dict) -> dict | None:
    """Coerce a stored measurement to floats, or reject it."""
    try:
        return {
            "sigma": float(raw_entry["sigma"]),
            "radii": [float(x) for x in raw_entry["radii"]],
            "delta_p": [float(x) for x in raw_entry["delta_p"]],
        }
    except (KeyError, TypeError, ValueError):
        return None


def _load_cache(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    cache: dict[str, dict] = {}
    for raw_key, raw_entry in raw.items():
        if not isinstance(raw_entry, dict):
            continue
        key = _sanitize_key(raw_key)
        entry = _sanitize_entry(raw_entry)
        if key is not None and entry is not None:
            cache[key] = entry
    return cache


def _store_cache(key: str, radii: np.ndarray, delta_p: np.ndarray, sigma: float) -> None:
    path = _cache_path().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(path)
    cache[key] = {
        "sigma": float(sigma),
        "radii": [float(x) for x in radii],
        "delta_p": [float(x) for x in delta_p],
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic; concurrent sweep workers never see a partial file


# ── Plot ──────────────────────────────────────────────────────────────


def _save_plot(path: Path, radii: np.ndarray, delta_p: np.ndarray, sigma: float) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    inv_r = 1.0 / radii
    predicted = sigma * inv_r
    ss_res = np.sum((delta_p - predicted) ** 2)
    ss_tot = np.sum((delta_p - np.mean(delta_p)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(inv_r, delta_p, s=80, color="tab:blue", label="Droplet measurements")
    x_fit = np.linspace(inv_r.min() * 0.9, inv_r.max() * 1.1, 100)
    ax1.plot(x_fit, sigma * x_fit, "r--", lw=2, label=f"Fit: σ = {sigma:.6g}")
    ax1.set_xlabel("1/R [lattice units]")
    ax1.set_ylabel("ΔP [lattice units]")
    ax1.set_title(f"Young–Laplace fit (R² = {r_squared:.4f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.scatter(radii, delta_p, s=80, color="tab:blue", label="Droplet measurements")
    r_fit = np.linspace(radii.min() * 0.9, radii.max() * 1.1, 100)
    ax2.plot(r_fit, sigma / r_fit, "r--", lw=2, label="ΔP = σ/R")
    ax2.set_xlabel("R [lattice units]")
    ax2.set_ylabel("ΔP [lattice units]")
    ax2.set_title("Pressure jump vs radius")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
