"""The init density field of an ``init_from_file`` run, read and measured once.

Part of the configuration rather than the analysis layer: the phase densities
measured here are a property of the run's input, consumed both by the
configuration (:attr:`SimulationConfig.phase_references`, which bands the
masked-gravity force) and by ``physical_parameters`` (the reported Bond and
Archimedes numbers). Keeping one reader under ``config`` lets the force and
the report share it without the configuration importing ``simulation_io``.
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from typing import NamedTuple
import numpy as np

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig

# Prefix remaps for analysing runs whose init NPZ is no longer where the config
# recorded it. Tried in order, first existing file wins:
#
# * Runs downloaded off DelftBlue: data stored under /scratch/<user>/LBM/26_TUD_LBM/
#   on the cluster lives under ~/ locally, so .../TUD_LBM_data/<run>/ resolves to
#   ~/TUD_LBM_data/<run>/.
# * Runs since archived: finished sweeps are moved into ~/TUD_LBM_data/old/ (with
#   the DelftBlue ones under old/DB/), which leaves every init_dir in their configs
#   pointing one or two levels above where the file now is. Without these, an
#   archived run silently falls back to the nominal config radius and the
#   prescribed rho_l - rho_v, i.e. a different Bo/Oh than the run was measured with.
_ARCHIVE_ROOTS: tuple[str, ...] = (
    f"{Path.home()}/TUD_LBM_data/old/DB/",
    f"{Path.home()}/TUD_LBM_data/old/",
)
_INIT_PATH_REMAPS: tuple[tuple[str, str], ...] = (
    ("/scratch/sbszkudlarek/LBM/26_TUD_LBM/", f"{Path.home()}/"),
    *((f"{Path.home()}/TUD_LBM_data/DB/", root) for root in _ARCHIVE_ROOTS),
    *(("/scratch/sbszkudlarek/LBM/26_TUD_LBM/TUD_LBM_data/", root) for root in _ARCHIVE_ROOTS),
    *((f"{Path.home()}/TUD_LBM_data/", root) for root in _ARCHIVE_ROOTS),
)


def resolve_npz_path(path: str | None) -> str | None:
    """Return an existing path for an init NPZ, remapping known prefixes.

    Falls back to the remapped location when the literal path is absent, so
    downloaded DelftBlue runs read their real init file instead of defaulting
    to the nominal config radius. Returns None when no existing file is found.
    """
    if not path:
        return None
    if Path(path).exists():
        return path
    for old, new in _INIT_PATH_REMAPS:
        if path.startswith(old):
            remapped = new + path[len(old) :]
            if Path(remapped).exists():
                return remapped
    return None


class InitField(NamedTuple):
    """An init density field with the phase densities measured off it.

    ``rho_mean`` and ``drho`` come from the field's own extrema rather than from
    ``config.rho_l``/``rho_v``: an equilibrated droplet relaxes away from the
    prescribed coexistence densities, so the config midpoint is not the
    mid-interface contour of the field and the config contrast is not the
    buoyancy contrast the run actually has.
    """

    rho: np.ndarray
    rho_min: float
    rho_max: float
    rho_mean: float
    drho: float


def load_init_field(config: SimulationConfig) -> InitField | None:
    """Load the init rho field from NPZ and measure its densities, for init_from_file.

    Returns ``None`` when no file resolves, it holds no ``rho``, or the field is
    empty or non-finite.
    """
    npz_path = resolve_npz_path(config.init_dir or config.initialisation.get("npz_path"))
    if not npz_path:
        return None

    try:
        stat = Path(npz_path).stat()
    except OSError:
        return None
    return _load_field_cached(npz_path, (stat.st_mtime_ns, stat.st_size))


#: Init fields held at once. Sized for a *set* of runs, not one: ``compare``
#: and ``regime-map`` walk N run directories, and each run's numbers are
#: resolved after every run's CSV has already read the same field, so a cache
#: that only spans one run re-reads a multi-megabyte NPZ per run. Matches
#: ``droplet_metrics._MAX_CACHED_RUNS``, which bounds the parallel snapshot
#: cache for the same walk.
_MAX_CACHED_INIT_FIELDS = 8


@lru_cache(maxsize=_MAX_CACHED_INIT_FIELDS)
def _load_field_cached(npz_path: str, _stat_key: tuple[int, int]) -> InitField | None:
    """Read and measure an init NPZ, memoized on the file's identity.

    Several resolvers (area, buoyancy contrast, contact-line spacing) each need
    the same multi-megabyte field, so it is read once. ``_stat_key`` carries the
    file's mtime and size purely so that rewriting a path invalidates the entry.
    """
    try:
        with np.load(npz_path) as data:
            if "rho" not in data:
                return None
            rho = np.asarray(data["rho"])
    except (KeyError, OSError, TypeError, ValueError):
        return None

    return measure_field(rho)


def measure_field(rho: np.ndarray) -> InitField | None:
    """Return *rho* with its extrema, midpoint and contrast, or None when unusable."""
    try:
        values = np.asarray(rho, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            return None
        rho_min = float(np.min(values))
        rho_max = float(np.max(values))
    except (TypeError, ValueError):
        return None
    return InitField(
        rho=rho,
        rho_min=rho_min,
        rho_max=rho_max,
        rho_mean=0.5 * (rho_max + rho_min),
        drho=rho_max - rho_min,
    )


def measure_init_phase_densities(config: SimulationConfig) -> tuple[float, float] | None:
    """Return ``(rho_min, rho_max)`` measured off the run's init NPZ, or None.

    :attr:`SimulationConfig.phase_references
    <src.config.simulation_config.SimulationConfig.phase_references>` bands the
    masked-gravity phase indicator on these, and ``physical_parameters.txt``
    reports the buoyancy contrast off the same :func:`load_init_field`, so the
    contrast a run injects and the one its Bond number quotes cannot diverge.
    Returns None when the run has no init file, or it holds no usable ``rho``.
    """
    field = load_init_field(config)
    if field is None:
        return None
    return field.rho_min, field.rho_max


#: Wetting-state scalars a hysteresis run carries across a restart: the wall
#: parameters the optimiser has accumulated and the contact-line anchors.
#: Contact angles are not among them; they are re-measured off the field.
RESTORED_WETTING_KEYS: tuple[str, ...] = (
    "phi_left",
    "phi_right",
    "d_rho_left",
    "d_rho_right",
    "cll_left",
    "cll_right",
)


def load_init_wetting(config: SimulationConfig) -> dict[str, float] | None:
    """The wetting state saved in the run's init NPZ, or ``None``.

    Returns ``None`` when no init file resolves or it lacks any of
    :data:`RESTORED_WETTING_KEYS` (a snapshot from a run without wetting).
    """
    npz_path = resolve_npz_path(config.init_dir or config.initialisation.get("npz_path"))
    if not npz_path:
        return None
    try:
        with np.load(npz_path) as data:
            if not all(key in data for key in RESTORED_WETTING_KEYS):
                return None
            return {key: float(data[key]) for key in RESTORED_WETTING_KEYS}
    except (KeyError, OSError, TypeError, ValueError):
        return None
