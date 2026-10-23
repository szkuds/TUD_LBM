"""Numerical surface-tension calibration via the Young-Laplace relation.

Some equations of state have no closed-form surface tension in terms of the
simulation parameters (Carnahan-Starling). For those, the lattice surface
tension is measured directly: periodic droplets of several radii are
equilibrated, the Laplace pressure jump is read from each, and a line is
fitted to ``dP = sigma / R`` (2-D Young-Laplace).

The droplets are **ordinary simulations**, not something this module runs.
:func:`calibration_configs` turns a config into one run config per radius
(``tud-lbm calibration stage`` writes them as TOMLs, under
``$TUD_LBM_DATA_DIR/surface_tension/<fluid>/configs/``); they are run like any
other config — in practice on DelftBlue through ``scripts/db_pipeline.sh`` —
and :func:`collect_calibration` assembles their final snapshots into sigma
(``tud-lbm calibration collect``). Nothing here advances a simulation, so a
``tud-lbm run`` whose fluid has no cached sigma starts immediately and says how
to stage the sweep, instead of equilibrating five droplets first.

The measurement is expensive, so results are cached on disk keyed by the
thermodynamic parameters and calibration grid size that determine sigma. The
cache is split in two by size, and only the small half lives in the repo:

``src/simulation_io/analysis/surface_tension/data/surface_tension_cache.json``
    The fitted numbers — a few hundred bytes per entry. Git-tracked on
    purpose, so a measured sigma is shared with the team via the normal git
    workflow rather than re-measured by everyone individually (commit it after
    adding a new entry).
``$TUD_LBM_DATA_DIR/surface_tension/fields/<digest>.npz``
    The equilibrated density field of every droplet, ~4 MB per entry. This is
    simulation output, so it is written under the user data root
    (:data:`~src.config.config_overview.BASE_RESULTS_DIR`) and never inside
    the repository — a run must not dirty the working tree. It is what lets a
    run with a cached sigma draw the snapshot figures below; a machine that has
    the JSON entry but not the fields simply skips those figures.
    :func:`_store_fields` refuses to write inside the package tree, so the
    split cannot silently regress.

Every artefact of a calibration is grouped under ``<run_dir>/surface_tension/``
rather than dropped flat into the run directory, in the same ``data/`` +
``plots/`` shape a run directory itself has:

``plots/calibration.png``
    The Young-Laplace fit. Written on every run whose fluid is calibrated.
``data/data.json``
    The fitted ``(radii, delta_p, sigma)``. Written alongside the figure.
``plots/snapshots/R_<R>.png``
    One figure per droplet showing its equilibrated density, bulk pressure and
    total pressure, with markers on the pixels entering the Laplace jump.
    Written whenever the density fields are in the field cache.
"""

from __future__ import annotations
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from typing import NamedTuple
import numpy as np
from rich.console import Console
from src.config.config_overview import BASE_RESULTS_DIR
from src.config.run_config import CONFIG_FILENAME
from src.config.run_config import DATA_DIRNAME
from src.config.run_config import PHYSICAL_PARAMETERS_FILENAME
from src.config.run_config import PLOTS_DIRNAME
from src.config.run_config import SNAPSHOT_GLOB
from src.config.run_config import SNAPSHOTS_DIRNAME
from src.operators.macroscopic import eos as _eos  # noqa: F401  registers the pressure operators
from src.operators.macroscopic.eos import has_analytical_surface_tension
from src.registry import get_operator_names

if TYPE_CHECKING:
    from collections.abc import Iterable
    from src.config import SimulationConfig

_MIN_GRID_SHAPE_DIMS = 2
_N_RADII = 5
_N_ITERATIONS = 200_000

# The sweep's droplet radii, as fractions of the smaller grid dimension.
_RADIUS_MIN_FRACTION = 1.0 / 4.0
_RADIUS_MAX_FRACTION = 1.0 / 3.0
# The vapour corner samples are inset by this fraction of the smaller grid
# dimension. Tying the inset to the grid rather than to the interface width is
# what keeps the sample geometry valid at every resolution: the radii above are
# themselves fractions of ``min(nx, ny)``, so a fixed fraction leaves the same
# clearance between the corners and the largest droplet's interface on any
# grid. An inset measured in interface widths does not — on a 32² grid the old
# ``3 * W = 12`` put the corners 5.7 lattice units from the centre, i.e. deep
# inside the largest (R = 10.7) droplet, silently measuring liquid as vapour.
_SAMPLE_MARGIN_FRACTION = 1.0 / 8.0

# Side of the square box every droplet sweep runs in.
#
# The sweep is deliberately NOT run in the run's own domain. Young-Laplace
# assumes a circular droplet in an effectively unbounded bath, and a
# non-square box breaks that: in a 201x101 domain the largest sweep droplet
# (R = 33.7) leaves only 33.7 lattice units to its own vertical periodic image,
# against 100+ in a square box, and the squeeze varies across the five radii
# (50.5 units of clearance at the smallest, 33.7 at the largest). That biases
# the five dP samples unequally, which is what a dP-vs-1/R slope fit cannot
# survive -- the shipped 201x101 entry fitted sigma = -0.0754 while four square
# boxes spanning a 2.3x range of sizes agreed at +0.0720 to +0.0731.
#
# 301 is the smallest square in that agreeing set, so it is the smallest side
# with evidence behind it -- and it is fixed, not a floor. The box used to grow
# with the run's grid (``max(nx, ny, 301)``), which put the run's size back into
# the cache key: a 401x101 run missed the entry a 201x101 run of the very same
# fluid had measured, for a sigma those four boxes show does not depend on it.
_CALIBRATION_SIDE = 301
_CALIBRATION_GRID_SHAPE = (_CALIBRATION_SIDE, _CALIBRATION_SIDE, 1)

# Everything a calibration leaves on disk lives under one directory of the
# results root: a folder per fluid holding its staged sweep configs, the sweep's
# run directories and the fitted result, plus the density-field cache.
SURFACE_TENSION_ROOT = Path(BASE_RESULTS_DIR) / "surface_tension"
_SWEEP_CONFIGS_DIRNAME = "configs"

# A sweep run is named ``surface_tension_<fluid>_R<radius>``. The name is what
# marks a config as a sweep run -- here, and in ``scripts/db_new_job.sh``, which
# keys the job command on the same prefix -- but it is a label, not an identity:
# a run's fluid is its cache key and its place in the sweep is its radius, both
# read from its own ``config.toml``.
_SWEEP_NAME_PREFIX = "surface_tension"
_SWEEP_NAME_PATTERN = re.compile(rf"(?:^|_){_SWEEP_NAME_PREFIX}_.+_R\d+$")


_PERIODIC_BC = {"top": "periodic", "bottom": "periodic", "left": "periodic", "right": "periodic"}

_CACHE_FILENAME = "surface_tension_cache.json"
# Every per-run artefact is grouped under this subdirectory of the run
# directory, so the names below need no further "surface_tension" prefix.
_OUTPUT_DIRNAME = "surface_tension"
# The tree mirrors a run directory — data/ for saved arrays and the fitted
# numbers, plots/ for figures — so the same tooling (notably
# ``visualise <snapshot>.npz --single``) resolves it the same way.
_PLOT_FILENAME = "calibration.png"
_DATA_FILENAME = "data.json"

# Equilibrated density fields, cached so a cache hit can still draw the
# snapshot figures. One file per cache key, named by its digest because the key
# itself is a JSON blob.
_FIELDS_KEY_DIGEST_LEN = 16
_FIELD_STACK_DIMS = 3  # (n_radii, nx, ny)

# Git-tracked, shared across the team: a measured sigma committed here is
# picked up by everyone on the next `git pull`, instead of each person
# re-running the ~40-minute droplet sweep. Sharing a new entry still requires
# an explicit `git add/commit/push` — writing to this file only updates your
# local working tree. Only the JSON qualifies: it is small, diffable and
# reviewable, which the multi-megabyte density fields are not.
_SHARED_CACHE_PATH = Path(__file__).resolve().parent / "data" / _CACHE_FILENAME

# The density fields are simulation output, so they go under the user data root
# — the same ``$TUD_LBM_DATA_DIR`` (default ``~/TUD_LBM_data``) that run
# directories default to — and never into the checkout. Machine-local by
# design: an absent field cache costs snapshot figures, never sigma.
_FIELDS_CACHE_DIR = SURFACE_TENSION_ROOT / "fields"

# Anything at or below the import root is the checkout; a calibration writing
# there would dirty the working tree. Guards :func:`_store_fields`.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]

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
    return surface_tension_dir(run_dir) / DATA_DIRNAME


def surface_tension_plots_dir(run_dir: str | Path) -> Path:
    """Return the directory holding the calibration and per-droplet figures."""
    return surface_tension_dir(run_dir) / PLOTS_DIRNAME


def cached_surface_tension(config: SimulationConfig) -> float | None:
    """The measured sigma for *config*'s fluid, or ``None`` when there is none.

    This is the link from a config to the cache: a run's ``config.toml`` never
    stores sigma, so every reader that wants the measured value asks here. A
    value already on ``config.extra`` wins; an EOS with a closed form is never
    looked up, since its sigma is not a measurement.
    """
    stored = config.extra.get("surface_tension")
    if stored is not None:
        return float(stored)
    if not needs_calibration(config):
        return None
    cached = _load_cache(_cache_path()).get(_cache_key(config))
    return None if cached is None else float(cached["sigma"])


def needs_calibration(config: SimulationConfig) -> bool:
    """Whether *config*'s sigma can only come from a Young-Laplace measurement.

    That is registry membership: an EOS registered under the
    ``"surface_tension"`` kind has a closed form.
    """
    return config.is_multiphase and not has_analytical_surface_tension(config.eos)


def is_calibrated(config: SimulationConfig) -> bool:
    """Whether the cache holds a measurement for *config*'s fluid, whatever its EOS."""
    return _cache_key(config) in _load_cache(_cache_path())


def record_surface_tension(config: SimulationConfig, run_dir: str | Path) -> SimulationConfig:
    """Attach the cached sigma when the EOS needs one, refresh the parameter file, return the config.

    A fluid with a closed form, or one not yet calibrated, returns *config*
    unchanged. Otherwise sigma is read from the cache, stored in
    ``config.extra['surface_tension']``, and ``physical_parameters.txt`` is
    rewritten in *run_dir* with the measured value.
    """
    # A sweep run *is* the measurement; it has no sigma to look up yet.
    if not needs_calibration(config) or is_sweep_config(config):
        return config

    sigma = calibrate_surface_tension(config, run_dir)
    if sigma is None:
        return config

    from src.simulation_io.analysis.physical_parameters import write_physical_parameters

    updated = replace(config, extra={**config.extra, "surface_tension": sigma})
    write_physical_parameters(updated, Path(run_dir) / PHYSICAL_PARAMETERS_FILENAME)
    return updated


def calibrate_surface_tension(config: SimulationConfig, run_dir: str | Path) -> float | None:
    """Return the cached lattice surface tension and write the calibration artefacts.

    On a hit the calibration figure and the fitted ``(radii, delta_p, sigma)``
    data file are written into ``run_dir/surface_tension/``, as are the
    per-droplet snapshot figures whenever the equilibrated density fields are in
    the field cache. On a miss nothing is measured and nothing is written: the
    sweep is a set of runs of its own (:func:`calibration_configs`), so this
    reports how to stage it and returns ``None``.
    """
    key = _cache_key(config)
    cached = _load_cache(_cache_path()).get(key)
    if cached is None:
        console.print(
            "[yellow]No cached σ for this fluid — continuing without it.[/yellow]\n"
            "[dim]Stage the Young–Laplace sweep with `tud-lbm calibration stage <config.toml>`, run the "
            "staged configs (scripts/db_pipeline.sh), then `tud-lbm calibration collect`.[/dim]"
        )
        return None

    radii = np.asarray(cached["radii"], dtype=float)
    delta_p = np.asarray(cached["delta_p"], dtype=float)
    sigma = float(cached["sigma"])
    console.print(f"[dim]Using cached σ = {sigma:.6g}[/dim]")

    data_dir = surface_tension_data_dir(run_dir)
    plots_dir = surface_tension_plots_dir(run_dir)
    _save_plot(plots_dir / _PLOT_FILENAME, radii, delta_p, sigma)
    _save_data(data_dir / _DATA_FILENAME, radii, delta_p, sigma)
    _save_snapshots(config, plots_dir / SNAPSHOTS_DIRNAME, radii, delta_p, _load_fields(key, radii.size))
    return sigma


# ── The sweep, as ordinary runs ───────────────────────────────────────


def sweep_radii() -> np.ndarray:
    """The droplet radii of a sweep, in lattice units of the calibration box."""
    return np.linspace(_CALIBRATION_SIDE * _RADIUS_MIN_FRACTION, _CALIBRATION_SIDE * _RADIUS_MAX_FRACTION, _N_RADII)


def fluid_label(config: SimulationConfig) -> str:
    """Readable name of *config*'s fluid: EOS initials, kappa and the density pair.

    ``cs_kappa0.015_rho12.18_0.015`` for a Carnahan-Starling fluid. It names the
    fluid's folder and its sweep runs; two fluids differing only in parameters
    the label leaves out share it, which is why nothing reads identity off it.
    """
    initials = "".join(word[0] for word in str(config.eos).split("-"))
    return f"{initials}_kappa{config.kappa:g}_rho{config.rho_l:g}_{config.rho_v:g}"


def sweep_run_name(config: SimulationConfig, radius: float) -> str:
    """The ``simulation_name`` of the sweep run of *config*'s fluid at *radius*."""
    return f"{_SWEEP_NAME_PREFIX}_{fluid_label(config)}_R{round(radius)}"


def is_sweep_config(config: SimulationConfig) -> bool:
    """Whether *config* is itself a run of a calibration sweep."""
    return _SWEEP_NAME_PATTERN.search(str(config.simulation_name)) is not None


def sweep_config_path(sweep_config: SimulationConfig) -> Path:
    """Where the staged TOML of *sweep_config* lives: ``<fluid dir>/configs/R<radius>.toml``."""
    radius_tag = str(sweep_config.simulation_name).rsplit("_", maxsplit=1)[-1]
    return Path(sweep_config.results_dir) / _SWEEP_CONFIGS_DIRNAME / f"{radius_tag}.toml"


def calibration_configs(config: SimulationConfig) -> list[SimulationConfig]:
    """One run config per sweep radius, for the fluid of *config*.

    Each is a plain ``multiphase`` run — periodic, force-free, a single liquid
    droplet — that saves only its final state and writes into the fluid's own
    folder under :data:`SURFACE_TENSION_ROOT`, where :func:`find_sweep_runs`
    gathers the finished run directories. Every one of them, and *config*
    itself, has the same cache key: that is what lets
    :func:`collect_calibration` file the result where *config* will look.
    """
    base = _calibration_config(config)
    fluid_dir = SURFACE_TENSION_ROOT / fluid_label(config)
    return [
        replace(
            base,
            results_dir=str(fluid_dir),
            # Only the final state enters the fit, and `pressure` is the very
            # field the macroscopic operator computed it from.
            save_interval=_N_ITERATIONS,
            save_fields=["rho", "pressure"],
            initialisation={**base.initialisation, "radii": [float(radius) / _CALIBRATION_SIDE]},
            simulation_name=sweep_run_name(config, radius),
        )
        for radius in sweep_radii()
    ]


def find_sweep_runs(root: str | Path) -> dict[Path, list[Path]]:
    """Sweep run directories under *root*, grouped by their fluid folder.

    A sweep run sits at ``<root>/<fluid>/<date>/<time>_<sweep name>``. Each
    group is sorted, so a later run of the same radius follows an earlier one.
    Run directories placed directly under *root* (``<root>/<date>/<run>``) are
    one level too shallow to match, whatever they are named.
    """
    groups: dict[Path, list[Path]] = {}
    for run_dir in sorted(Path(root).glob("*/*/*")):
        if _SWEEP_NAME_PATTERN.search(run_dir.name) is not None and (run_dir / CONFIG_FILENAME).is_file():
            groups.setdefault(run_dir.parent.parent, []).append(run_dir)
    return groups


class _SweepSample(NamedTuple):
    """What one finished sweep run contributes to the fit."""

    key: str
    index: int
    radius: float
    delta_p: float
    density: np.ndarray


def collect_calibration(run_dirs: Iterable[str | Path], out_dir: str | Path | None = None) -> float | None:
    """Fit sigma from the finished sweep runs of one fluid and cache it.

    Reads the final snapshot of each run, takes the Laplace jump from its
    ``pressure`` field and fits ``dP = sigma / R`` over the radii. The cache
    key is recomputed from the runs' own ``config.toml``, so the entry lands
    exactly where the originating config looks it up, and a run's place in the
    sweep is its configured radius — neither is read off the directory name.
    When a radius was run more than once the last directory in sorted order —
    the newest — wins. With *out_dir* (the fluid's folder) the fit figure and
    its data are written there too.

    Returns ``None``, storing nothing, until every radius of the sweep has a
    finished run.

    Raises:
        ValueError: If the runs do not all describe the same fluid.
    """
    samples: dict[int, _SweepSample] = {}
    for run_dir in sorted(Path(d) for d in run_dirs):
        sample = _read_sweep_run(run_dir)
        if sample is not None:
            samples[sample.index] = sample

    keys = {sample.key for sample in samples.values()}
    if len(keys) > 1:
        msg = "the sweep runs do not share one set of fluid parameters; collect one fluid at a time"
        raise ValueError(msg)
    if sorted(samples) != list(range(_N_RADII)):
        console.print(f"[dim]Sweep incomplete: {len(samples)}/{_N_RADII} radii finished.[/dim]")
        return None

    ordered = [samples[index] for index in range(_N_RADII)]
    key = ordered[0].key
    radii = np.array([sample.radius for sample in ordered])
    delta_p = np.array([sample.delta_p for sample in ordered])
    sigma = _fit_sigma(radii, delta_p)
    _store_cache(key, radii, delta_p, sigma, _CALIBRATION_GRID_SHAPE)
    _store_fields(key, [sample.density for sample in ordered])
    if out_dir is not None:
        _save_plot(Path(out_dir) / _PLOT_FILENAME, radii, delta_p, sigma)
        _save_data(Path(out_dir) / _DATA_FILENAME, radii, delta_p, sigma)
    console.print(f"[bold green]Surface tension calibrated: σ = {sigma:.6g}[/bold green]")
    return sigma


def _read_sweep_run(run_dir: Path) -> _SweepSample | None:
    """The fit sample of one sweep run, or ``None`` while it has not finished."""
    from src.config.adapter_toml import TomlAdapter
    from src.simulation_io.analysis.droplet_metrics import parse_timestep_from_path

    config_path = run_dir / CONFIG_FILENAME
    snapshots = sorted((run_dir / DATA_DIRNAME).glob(SNAPSHOT_GLOB), key=parse_timestep_from_path)
    if not config_path.is_file() or not snapshots:
        return None

    config = TomlAdapter().load(str(config_path))
    nx, ny = int(config.grid_shape[0]), int(config.grid_shape[1])
    radius = float(config.initialisation["radii"][0]) * min(nx, ny)
    matches = np.flatnonzero(np.isclose(sweep_radii(), radius))
    if matches.size != 1:
        return None
    # A run cut short by its time limit has snapshots, but not an equilibrated one.
    if parse_timestep_from_path(snapshots[-1]) < config.nt:
        return None
    with np.load(snapshots[-1]) as snapshot:
        pressure = _field_2d(snapshot["pressure"])
        density = _field_2d(snapshot["rho"])

    return _SweepSample(
        key=_cache_key(config),
        index=int(matches[0]),
        radius=radius,
        delta_p=_pressure_jump(pressure),
        density=density,
    )


def _field_2d(field: np.ndarray) -> np.ndarray:
    """The ``(nx, ny)`` slice of a saved ``(nx, ny, 1, 1, 1)`` scalar field."""
    return np.asarray(field, dtype=float)[:, :, 0, 0, 0]


def _calibration_config(config: SimulationConfig) -> SimulationConfig:
    """An isolated single-droplet config: periodic, no forces, no wetting."""
    return replace(
        config,
        sim_type="multiphase",
        bc_config=dict(_PERIODIC_BC),
        grid_shape=_CALIBRATION_GRID_SHAPE,
        nt=_N_ITERATIONS,
        save_interval=0,
        skip_interval=0,
        save_fields=None,
        plot_fields=None,
        animate_fields=None,
        overlay_fields=None,
        g=None,
        gravity_force=None,
        gravity_masked_force=None,
        electric_force=None,
        wetting_config=None,
        hysteresis_config=None,
        chemical_step_config=None,
        init_type="multiphase_bubbles",
        init_dir=None,
        initialisation={"centres": [[0.5, 0.5]], "radii": [0.2], "dispersed": "liquid"},
        simulation_name=f"{config.simulation_name}_surface_tension",
    )


def sample_points(nx: int, ny: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Return the ``(inside, outside)`` array indices the Laplace jump reads.

    *inside* is the domain centre, where the droplet sits; *outside* are four
    corner pixels inset by :data:`_SAMPLE_MARGIN_FRACTION` of the smaller grid
    dimension, far enough from both the droplet and the periodic wrap to be
    bulk vapour.

    The geometry is tied to the grid, not to the interface width. Because the
    sweep's radii are themselves grid fractions, the clearance between a corner
    pixel and the largest droplet's interface is then the same at every
    resolution — which an inset measured in interface widths cannot guarantee.

    This is the single definition of the sample geometry: the measurement in
    :func:`_pressure_jump` and the markers the snapshot figures draw both read
    it, so the plot cannot drift from what was actually measured.
    """
    margin = max(1, round(min(nx, ny) * _SAMPLE_MARGIN_FRACTION))
    inside = (nx // 2, ny // 2)
    outside = [
        (margin, margin),
        (margin, ny - margin - 1),
        (nx - margin - 1, margin),
        (nx - margin - 1, ny - margin - 1),
    ]
    return inside, outside


def _pressure_jump(pressure: np.ndarray) -> float:
    """Laplace jump: centre (liquid) minus the mean of four vapour corners."""
    nx, ny = pressure.shape
    inside, outside = sample_points(nx, ny)
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


def _cache_key(config: SimulationConfig) -> str:
    """The fluid parameters that determine sigma, as canonical JSON.

    ``grid_shape`` is the fixed calibration box rather than the run's grid, so
    the key depends on the fluid alone; it stays in the key because the stored
    entries carry it.
    """
    values = {k: getattr(config, k, None) for k in _CACHE_KEYS}
    values["grid_shape"] = list(_CALIBRATION_GRID_SHAPE)
    return json.dumps(values, sort_keys=True)


def _sanitize_key(raw_key: str) -> str | None:
    """Rebuild a cache key from validated primitives, or reject it.

    Valid keys are the canonical JSON produced by :func:`_cache_key`: exactly
    the ``_CACHE_KEYS`` fields, with numeric or ``None`` values, a validated
    ``grid_shape``, and an EOS registered under the ``"pressure"`` kind. The returned key is
    re-serialised from coerced primitives so nothing read from the cache file
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
        known = next((eos for eos in get_operator_names("pressure") if eos == value), None)
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
    """Path of the cached density fields for *key*, under the user data root."""
    return _FIELDS_CACHE_DIR / f"{_key_digest(key)}.npz"


def _key_digest(key: str) -> str:
    """Short stable name for a cache key, which is itself a JSON blob."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:_FIELDS_KEY_DIGEST_LEN]


def _store_fields(key: str, densities: list[np.ndarray]) -> None:
    """Cache the equilibrated density fields under the user data root.

    Stored as one stacked ``(n_radii, nx, ny)`` array so a run with a cached
    sigma can draw the snapshot figures without the sweep's run directories. Unlike the JSON
    cache these are machine-local and never committed: multi-megabyte binaries
    do not belong in the checkout, and a peer missing them loses only the
    snapshot figures.
    """
    if not densities:
        return
    path = _fields_path(key).resolve()
    if path.is_relative_to(_PACKAGE_ROOT):
        msg = f"refusing to write the density-field cache inside the repository: {path}"
        raise RuntimeError(msg)
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

    A cache entry measured on another machine has no fields here; that
    is reported rather than raised, since sigma itself is already measured.
    """
    if densities is None:
        console.print(
            "[dim]No cached droplet fields for these parameters — snapshot figures are "
            "skipped (the fields are machine-local; `tud-lbm calibration collect` writes them).[/dim]"
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
