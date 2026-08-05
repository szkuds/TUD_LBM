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
thermodynamic parameters and calibration grid size that determine sigma. The
cache file lives at
``src/simulation_io/analysis/surface_tension/data/surface_tension_cache.json`` — inside the repo, so
it's shared with the team via the normal git workflow rather than re-measured
by everyone individually (commit it after adding a new entry). The equilibrated
density field of every droplet is cached beside it under ``data/fields/`` and
must be committed together with the JSON: it is what lets a cache hit still
draw the snapshot figures below without re-running the sweep.

Every artefact of a calibration is grouped under ``<run_dir>/surface_tension/``
rather than dropped flat into the run directory, in the same ``data/`` +
``plots/`` shape a run directory itself has:

``plots/calibration.png``
    The Young-Laplace fit. Written on every run, whether measured or served
    from cache.
``data/data.json``
    The fitted ``(radii, delta_p, sigma)``. Written alongside the figure.
``plots/snapshots/R_<R>.png``
    One figure per droplet showing its equilibrated density, bulk pressure and
    total pressure, with markers on the pixels entering the Laplace jump.
    Written whenever the density fields are available — from the sweep, or
    from the field cache.
``data/radius_<R>_{init,final}.npz``
    The full initial and equilibrated ``State`` of every droplet, saved only
    when the sweep actually runs. Living in ``data/`` is what lets
    ``tud-lbm visualise <that file> --single`` write its figure into
    ``plots/snapshots/`` beside the calibration output.
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast
import numpy as np
from rich.console import Console
from src.operators.macroscopic.eos import PRESSURE_EOS as _KNOWN_EOS
from src.operators.macroscopic.eos import build_pressure_fn

if TYPE_CHECKING:
    from src.config import SimulationConfig
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State

# EOS whose surface tension must be measured rather than derived analytically.
_EOS_REQUIRING_CALIBRATION = frozenset({"carnahan-starling"})

_MIN_GRID_SHAPE_DIMS = 2
_N_RADII = 5
_N_ITERATIONS = 200_000
_PERIODIC_BC = {"top": "periodic", "bottom": "periodic", "left": "periodic", "right": "periodic"}

_CACHE_FILENAME = "surface_tension_cache.json"
# Every per-run artefact is grouped under this subdirectory of the run
# directory, so the names below need no further "surface_tension" prefix.
_OUTPUT_DIRNAME = "surface_tension"
# The tree mirrors a run directory — data/ for saved arrays and the fitted
# numbers, plots/ for figures — so the same tooling (notably
# ``visualise <snapshot>.npz --single``) resolves it the same way.
_OUTPUT_DATA_DIRNAME = "data"
_OUTPUT_PLOTS_DIRNAME = "plots"
_PLOT_FILENAME = "calibration.png"
_DATA_FILENAME = "data.json"
_SNAPSHOTS_DIRNAME = "snapshots"

# Equilibrated density fields, cached beside the JSON so a cache hit can still
# draw the snapshot figures. One file per cache key, named by its digest
# because the key itself is a JSON blob.
_FIELDS_DIRNAME = "fields"
_FIELDS_KEY_DIGEST_LEN = 16
_FIELD_STACK_DIMS = 3  # (n_radii, nx, ny)

# Git-tracked, shared across the team: a measured sigma committed here is
# picked up by everyone on the next `git pull`, instead of each person
# re-running the ~40-minute droplet sweep. Sharing a new entry still requires
# an explicit `git add/commit/push` — writing to this file only updates your
# local working tree.
_SHARED_CACHE_PATH = Path(__file__).resolve().parent / "data" / _CACHE_FILENAME

# Parameters that uniquely determine the measured surface tension.
_CACHE_KEYS = (
    "eos",
    "kappa",
    "rho_l",
    "rho_v",
    "interface_width",
    "a_eos",
    "b_eos",
    "r_eos",
    "t_eos",
    "grid_shape",
)

console = Console()


def surface_tension_dir(run_dir: str | Path) -> Path:
    """Return the subdirectory of *run_dir* holding the surface-tension artefacts."""
    return Path(run_dir) / _OUTPUT_DIRNAME


def surface_tension_data_dir(run_dir: str | Path) -> Path:
    """Return the directory holding the saved droplet states and ``data.json``."""
    return surface_tension_dir(run_dir) / _OUTPUT_DATA_DIRNAME


def surface_tension_plots_dir(run_dir: str | Path) -> Path:
    """Return the directory holding the calibration and per-droplet figures."""
    return surface_tension_dir(run_dir) / _OUTPUT_PLOTS_DIRNAME


def record_surface_tension(config: SimulationConfig, run_dir: str | Path) -> SimulationConfig:
    """Measure sigma when the EOS needs it, refresh the parameter file, return updated config.

    For an EOS with a closed-form surface tension the config is returned
    unchanged. Otherwise sigma is measured (or read from cache), stored in
    ``config.extra['surface_tension']``, and ``physical_parameters.txt`` is
    rewritten in *run_dir* with the measured value.
    """
    if not (config.is_multiphase and config.eos in _EOS_REQUIRING_CALIBRATION):
        return config

    from src.simulation_io.analysis.physical_parameters import write_physical_parameters

    sigma = calibrate_surface_tension(config, run_dir)
    updated = replace(config, extra={**config.extra, "surface_tension": sigma})
    write_physical_parameters(updated, Path(run_dir) / "physical_parameters.txt")
    return updated


def calibrate_surface_tension(config: SimulationConfig, run_dir: str | Path) -> float:
    """Return the measured lattice surface tension and write the calibration figure.

    Looks up a cached value keyed by the EOS thermodynamic parameters; on a
    miss, runs the droplet sweep and caches the result. The calibration figure
    and the fitted ``(radii, delta_p, sigma)`` data file are always written
    into ``run_dir/surface_tension/``, as are the per-droplet snapshot figures
    whenever the equilibrated density fields are available — freshly measured
    or restored from the field cache. On a fresh measurement the initial and
    equilibrated state of every droplet is additionally saved under
    ``run_dir/surface_tension/data/`` (a cache hit runs no droplets, so no
    states are written).
    """
    data_dir = surface_tension_data_dir(run_dir)
    plots_dir = surface_tension_plots_dir(run_dir)
    cache_path = _cache_path()
    key = _cache_key(config)

    cached = _load_cache(cache_path).get(key)
    if cached is not None:
        radii = np.asarray(cached["radii"], dtype=float)
        delta_p = np.asarray(cached["delta_p"], dtype=float)
        sigma = float(cached["sigma"])
        densities = _load_fields(key, radii.size)
        console.print(
            f"[dim]Using cached σ = {sigma:.6g} — droplet states are only "
            f"saved when the calibration sweep actually runs.[/dim]"
        )
    else:
        console.print(
            f"[dim]No cached σ for these EOS parameters — running "
            f"Young–Laplace calibration ({_N_RADII} droplets)...[/dim]"
        )
        radii, delta_p, densities = _measure_pressure_jumps(config, states_dir=data_dir)
        sigma = _fit_sigma(radii, delta_p)
        _store_cache(key, radii, delta_p, sigma, config.grid_shape)
        _store_fields(key, densities)
        console.print(f"[bold green]Surface tension calibrated: σ = {sigma:.6g}[/bold green]")

    _save_plot(plots_dir / _PLOT_FILENAME, radii, delta_p, sigma)
    _save_data(data_dir / _DATA_FILENAME, radii, delta_p, sigma)
    _save_snapshots(config, plots_dir / _SNAPSHOTS_DIRNAME, radii, delta_p, densities)
    return sigma


# ── Measurement ───────────────────────────────────────────────────────


def _measure_pressure_jumps(
    config: SimulationConfig, states_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Equilibrate one droplet per radius and return ``(radii, delta_p, densities)``.

    *densities* holds the equilibrated 2-D density field of each droplet — the
    very field the pressure jump was read from, handed back so the snapshot
    figures and the field cache need no second pass over the states.

    When *states_dir* is given, the initial and final :class:`State` of each
    droplet is saved there as ``radius_<R>_init.npz`` / ``radius_<R>_final.npz``
    so the fields entering the Young-Laplace fit can be inspected afterwards.
    """
    from src.operators.initialise import build_initialise_fn
    from src.operators.macroscopic import build_multiphase_params
    from src.pipeline.runner import init_state
    from src.pipeline.setup import build_setup

    if config.interface_width is None or config.rho_l is None or config.rho_v is None:
        msg = "interface_width, rho_l, rho_v are required for surface-tension calibration"
        raise ValueError(msg)

    nx, ny = int(config.grid_shape[0]), int(config.grid_shape[1])
    min_dim = min(nx, ny)
    radii = np.linspace(min_dim / 5.0, min_dim / 3.0, _N_RADII)
    width = int(config.interface_width)
    rho_l, rho_v = float(config.rho_l), float(config.rho_v)

    calib_config = _calibration_config(config)
    mp = build_multiphase_params(calib_config)
    pressure_fn = build_pressure_fn(mp)

    setup = build_setup(calib_config)
    grid_shape = cast("tuple[int, int, int]", setup.grid_shape)
    init_fn = build_initialise_fn("multiphase_bubbles")

    if states_dir is not None:
        states_dir.mkdir(parents=True, exist_ok=True)

    delta_p = np.empty(_N_RADII)
    densities: list[np.ndarray] = []
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
        initial_state = init_state(setup, f=f0)
        if states_dir is not None:
            _save_state(states_dir / f"radius_{radius:.2f}_init.npz", initial_state)
        final_state = _run_to_final_state(setup, initial_state, _N_ITERATIONS)
        if states_dir is not None:
            _save_state(states_dir / f"radius_{radius:.2f}_final.npz", final_state)
        rho_2d = _density_2d(final_state)
        densities.append(rho_2d)
        delta_p[i] = _pressure_jump(pressure_fn(rho_2d), width)

    return radii, delta_p, densities


def _save_state(path: Path, state: State) -> None:
    """Save every array field of *state* to an ``.npz`` snapshot."""
    arrays = {name: np.asarray(value) for name, value in state._asdict().items() if hasattr(value, "shape")}
    np.savez(path, **arrays)  # ty: ignore[invalid-argument-type]


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


def sample_points(nx: int, ny: int, width: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Return the ``(inside, outside)`` array indices the Laplace jump reads.

    *inside* is the domain centre, where the droplet sits; *outside* are four
    corner pixels inset by three interface widths, far enough from both the
    droplet and the periodic wrap to be bulk vapour.

    This is the single definition of the sample geometry: the measurement in
    :func:`_pressure_jump` and the markers the snapshot figures draw both read
    it, so the plot cannot drift from what was actually measured.
    """
    margin = 3 * width
    inside = (nx // 2, ny // 2)
    outside = [
        (margin, margin),
        (margin, ny - margin - 1),
        (nx - margin - 1, margin),
        (nx - margin - 1, ny - margin - 1),
    ]
    return inside, outside


def _pressure_jump(pressure: np.ndarray, width: int) -> float:
    """Laplace jump: centre (liquid) minus the mean of four vapour corners."""
    nx, ny = pressure.shape
    inside, outside = sample_points(nx, ny, width)
    p_inside = pressure[inside]
    p_outside = np.mean([pressure[point] for point in outside])
    return float(p_inside - p_outside)


def _fit_sigma(radii: np.ndarray, delta_p: np.ndarray) -> float:
    """Surface tension is the slope of ``dP`` against ``1/R``."""
    slope, _ = np.polyfit(1.0 / radii, delta_p, deg=1)
    return float(slope)


# ── Cache ─────────────────────────────────────────────────────────────


def _cache_path() -> Path:
    return _SHARED_CACHE_PATH


def _cache_grid_shape(config: SimulationConfig) -> list[int]:
    return [int(dim) for dim in config.grid_shape]


def _cache_key(config: SimulationConfig) -> str:
    values = {k: getattr(config, k, None) for k in _CACHE_KEYS}
    values["grid_shape"] = _cache_grid_shape(config)
    return json.dumps(values, sort_keys=True)


def _sanitize_key(raw_key: str) -> str | None:
    """Rebuild a cache key from validated primitives, or reject it.

    Valid keys are the canonical JSON produced by :func:`_cache_key`: exactly
    the ``_CACHE_KEYS`` fields, with numeric or ``None`` values, a validated
    ``grid_shape``, and an EOS name from ``_KNOWN_EOS``. The returned key is
    re-serialized from coerced primitives so nothing read from the cache file
    is echoed back verbatim.
    """
    try:
        values = json.loads(raw_key)
    except ValueError:
        return None
    if not isinstance(values, dict) or set(values) != set(_CACHE_KEYS):
        return None
    clean: dict[str, str | int | float | list[int] | None] = {}
    for field in _CACHE_KEYS:
        clean_value = _sanitize_key_field(field, values[field])
        if clean_value is None and values[field] is not None:
            return None
        clean[field] = clean_value
    return json.dumps(clean, sort_keys=True)


def _sanitize_key_field(field: str, value: object) -> str | int | float | list[int] | None:
    clean: str | int | float | list[int] | None
    if field == "eos":
        known = next((eos for eos in _KNOWN_EOS if eos == value), None)
        if value is not None and known is None:
            return None
        clean = known
    elif field == "grid_shape":
        try:
            clean = _sanitize_grid_shape(value)
        except ValueError:
            return None
    elif value is None:
        clean = None
    elif isinstance(value, int) and not isinstance(value, bool):
        clean = int(value)
    elif isinstance(value, float):
        clean = float(value)
    else:
        clean = None
    return clean


def _sanitize_entry(raw_entry: dict) -> dict | None:
    """Coerce a stored measurement to floats, or reject it."""
    try:
        return {
            "sigma": float(raw_entry["sigma"]),
            "radii": [float(x) for x in raw_entry["radii"]],
            "delta_p": [float(x) for x in raw_entry["delta_p"]],
            "grid_shape": _sanitize_grid_shape(raw_entry["grid_shape"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _sanitize_grid_shape(raw_grid_shape: object) -> list[int]:
    if not isinstance(raw_grid_shape, list) or len(raw_grid_shape) < _MIN_GRID_SHAPE_DIMS:
        msg = "grid_shape must be a list with at least two positive integer dimensions"
        raise ValueError(msg)
    grid_shape: list[int] = []
    for dim in raw_grid_shape:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            msg = "grid_shape must contain only positive integer dimensions"
            raise ValueError(msg)
        grid_shape.append(dim)
    return grid_shape


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


def _store_cache(
    key: str, radii: np.ndarray, delta_p: np.ndarray, sigma: float, grid_shape: tuple[int, ...] | list[int]
) -> None:
    path = _cache_path().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(path)
    cache[key] = {
        "sigma": float(sigma),
        "radii": [float(x) for x in radii],
        "delta_p": [float(x) for x in delta_p],
        "grid_shape": [int(dim) for dim in grid_shape],
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic; concurrent sweep workers never see a partial file


def _fields_path(key: str) -> Path:
    """Path of the cached density fields for *key*."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:_FIELDS_KEY_DIGEST_LEN]
    return _cache_path().parent / _FIELDS_DIRNAME / f"{digest}.npz"


def _store_fields(key: str, densities: list[np.ndarray]) -> None:
    """Cache the equilibrated density fields beside the shared JSON cache.

    Stored as one stacked ``(n_radii, nx, ny)`` array so a later cache hit can
    redraw the snapshot figures without re-running the sweep. Like the JSON
    cache this is git-tracked: commit it alongside the new cache entry.
    """
    if not densities:
        return
    path = _fields_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, rho=np.stack([np.asarray(rho) for rho in densities]))
    tmp.replace(path)  # atomic; a reader never sees a half-written archive


def _load_fields(key: str, n_radii: int) -> list[np.ndarray] | None:
    """Return the cached density fields for *key*, or ``None`` if unusable.

    A missing, corrupt or stale file is simply a miss — the snapshot figures
    are then skipped, never an error, since the measurement itself does not
    depend on them.
    """
    path = _fields_path(key)
    try:
        with np.load(path) as raw:
            stacked = np.asarray(raw["rho"], dtype=float)
    except (OSError, ValueError, KeyError):
        return None
    if stacked.ndim != _FIELD_STACK_DIMS or stacked.shape[0] != n_radii:
        return None
    return list(stacked)


# ── Per-run output ────────────────────────────────────────────────────


def _save_data(path: Path, radii: np.ndarray, delta_p: np.ndarray, sigma: float) -> None:
    """Write the fitted measurement data into the run directory."""
    payload = {
        "sigma": float(sigma),
        "radii": [float(x) for x in radii],
        "delta_p": [float(x) for x in delta_p],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_snapshots(
    config: SimulationConfig,
    out_dir: Path,
    radii: np.ndarray,
    delta_p: np.ndarray,
    densities: list[np.ndarray] | None,
) -> None:
    """Write one snapshot figure per droplet, when the density fields are known.

    A cache hit whose fields predate the field cache has nothing to draw; that
    is reported rather than raised, since sigma itself is already measured.
    """
    if densities is None:
        console.print(
            "[dim]No cached droplet fields for these parameters — snapshot "
            "figures need a re-measurement (delete the cache entry to force one).[/dim]"
        )
        return
    from src.simulation_io.analysis.surface_tension.snapshot_figures import save_snapshot_figures

    save_snapshot_figures(_calibration_config(config), out_dir, radii, delta_p, densities, timestep=_N_ITERATIONS)


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
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
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
