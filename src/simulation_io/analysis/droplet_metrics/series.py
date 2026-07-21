"""Per-run droplet metric time series.

``compute_droplet_series`` is the only place in the codebase that reads
``.npz`` snapshots for droplet metrics. Every consumer — the CSV export, the
Ca/theta plots, the contact-angle and contact-line-speed plots, and the regime
map — derives from the :class:`DropletSeries` it returns.
"""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from src.simulation_io.analysis.droplet_metrics._scales import MetricScales
from src.simulation_io.analysis.droplet_metrics._scales import resolve_scales
from src.simulation_io.analysis.droplet_metrics._snapshot import avg_x_location
from src.simulation_io.analysis.droplet_metrics._snapshot import center_of_mass
from src.simulation_io.analysis.droplet_metrics._snapshot import contact_angles_from_rho
from src.simulation_io.analysis.droplet_metrics._snapshot import contact_lines_from_rho
from src.simulation_io.analysis.droplet_metrics._snapshot import extract_rho_2d
from src.simulation_io.analysis.droplet_metrics._snapshot import extract_velocity_components_2d
from src.simulation_io.analysis.droplet_metrics._snapshot import mean_velocity_in_liquid
from src.simulation_io.analysis.droplet_metrics._snapshot import optional_contact_metrics
from src.simulation_io.analysis.droplet_metrics._snapshot import parse_timestep
from src.simulation_io.analysis.droplet_metrics._snapshot import parse_timestep_from_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    import pandas as pd
    from src.config import SimulationConfig

#: Column order of ``simulation_data.csv``. Downstream consumers select by
#: name, but the order is pinned so the file stays diffable across runs.
CSV_COLUMNS: tuple[str, ...] = (
    "iteration",
    "normalised_iteration",
    "avg_x_location",
    "avg_x_location_norm",
    "avg_u_x",
    "avg_u_y",
    "cll_left",
    "cll_right",
    "v_left",
    "v_right",
    "v_cm",
    "ca_left",
    "ca_right",
    "cm_x",
    "cm_y",
    "Ca",
    "Ca_cll_left",
    "Ca_cll_right",
    "Ca_cm",
    "Ca_norm",
    "Re",
    # Appended after the historical 21. Consumers select by name, so appending
    # is safe; reordering would not be.
    "sigma_lg",
    "sigma_lg_source",
    "Ca_analytical",
)


def backward_diff(values: np.ndarray, iterations: np.ndarray, fallback_interval: int) -> np.ndarray:
    """Backward difference of *values* with respect to *iterations*.

    Divides by the **actual** gap between consecutive samples rather than the
    nominal save interval. The two differ whenever a run was resumed, used a
    ``skip_interval``, or had snapshots pruned — in which case a fixed-interval
    difference over-reports the rate by the ratio of the gaps.

    The leading element is 0.0 (no earlier sample), as are any samples sharing
    an iteration with their predecessor.
    """
    vals = np.asarray(values, dtype=float)
    iters = np.asarray(iterations, dtype=float)
    if vals.size == 0:
        return np.zeros_like(vals)

    deltas = np.diff(vals, prepend=vals[0])
    gaps = np.diff(iters, prepend=iters[0] - max(fallback_interval, 1))
    return np.divide(deltas, gaps, out=np.zeros_like(deltas), where=gaps > 0)


@dataclass(frozen=True)
class DropletSeries:
    """Per-snapshot droplet metrics for one run, ordered by timestep.

    Naming: ``theta_*`` are contact ANGLES in degrees — stored in the ``.npz``
    files under the keys ``ca_left``/``ca_right``, and written to the CSV under
    those same historical names. Every ``ca*`` property on this class is a
    capillary NUMBER. See :meth:`to_dataframe` for the rename.
    """

    iteration: np.ndarray
    avg_u_x: np.ndarray
    avg_u_y: np.ndarray
    avg_x_location: np.ndarray
    cll_left: np.ndarray
    cll_right: np.ndarray
    theta_left: np.ndarray
    theta_right: np.ndarray
    cm_x: np.ndarray
    cm_y: np.ndarray
    scales: MetricScales

    def __len__(self) -> int:
        """Number of snapshots in the series."""
        return int(self.iteration.size)

    @cached_property
    def normalised_iteration(self) -> np.ndarray:
        """Iteration mapped onto ``[0, 1]``; all zeros when the run has no span."""
        it = self.iteration.astype(float)
        span = it.max() - it.min() if it.size else 0.0
        if span <= 0:
            return np.zeros_like(it)
        return (it - it.min()) / span

    @cached_property
    def avg_x_location_norm(self) -> np.ndarray:
        """Average x-position in units of the initial droplet radius."""
        if self.scales.r_zero <= 0:
            return np.zeros_like(self.avg_x_location)
        return self.avg_x_location / self.scales.r_zero

    @cached_property
    def v_left(self) -> np.ndarray:
        """Left contact-line velocity."""
        return backward_diff(self.cll_left, self.iteration, self.scales.save_interval)

    @cached_property
    def v_right(self) -> np.ndarray:
        """Right contact-line velocity."""
        return backward_diff(self.cll_right, self.iteration, self.scales.save_interval)

    @cached_property
    def v_cm(self) -> np.ndarray:
        """Centre-of-mass velocity in x."""
        return backward_diff(self.cm_x, self.iteration, self.scales.save_interval)

    def _capillary(self, velocity: np.ndarray, sigma: float | None) -> np.ndarray:
        """``Ca = v * nu / sigma``, all zeros when *sigma* is unavailable."""
        if not sigma:
            return np.zeros_like(velocity)
        return (velocity * self.scales.nu) / sigma

    @cached_property
    def ca(self) -> np.ndarray:
        """Capillary number from the mean liquid velocity, using the primary sigma."""
        return self._capillary(self.avg_u_x, self.scales.sigma_primary)

    @cached_property
    def ca_analytical(self) -> np.ndarray:
        """:attr:`ca` forced onto the closed-form sigma, for comparison.

        Equals :attr:`ca` unless the run carries a measured calibration.
        """
        return self._capillary(self.avg_u_x, self.scales.sigma_analytical)

    @cached_property
    def ca_cll_left(self) -> np.ndarray:
        """Capillary number of the left (trailing) contact line."""
        return self._capillary(self.v_left, self.scales.sigma_primary)

    @cached_property
    def ca_cll_right(self) -> np.ndarray:
        """Capillary number of the right (leading) contact line."""
        return self._capillary(self.v_right, self.scales.sigma_primary)

    @cached_property
    def ca_cm(self) -> np.ndarray:
        """Capillary number of the centre of mass."""
        return self._capillary(self.v_cm, self.scales.sigma_primary)

    @cached_property
    def ca_norm(self) -> np.ndarray:
        """Capillary number normalised by the gravity inclination, when inclined."""
        import math

        if self.scales.incl_deg > 0:
            return self.ca / math.sin(math.radians(self.scales.incl_deg))
        return self.ca

    @cached_property
    def re(self) -> np.ndarray:
        """Reynolds number based on the initial droplet diameter."""
        return (self.avg_u_x * (2.0 * self.scales.r_zero)) / self.scales.nu

    def to_dataframe(self) -> pd.DataFrame:
        """Serialise to the ``simulation_data.csv`` column layout.

        The ``theta_left``/``theta_right`` contact angles are written under the
        historical column names ``ca_left``/``ca_right``. Those columns hold
        DEGREES, not capillary numbers, despite the ``ca`` prefix.
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "iteration": self.iteration,
                "normalised_iteration": self.normalised_iteration,
                "avg_x_location": self.avg_x_location,
                "avg_x_location_norm": self.avg_x_location_norm,
                "avg_u_x": self.avg_u_x,
                "avg_u_y": self.avg_u_y,
                "cll_left": self.cll_left,
                "cll_right": self.cll_right,
                "v_left": self.v_left,
                "v_right": self.v_right,
                "v_cm": self.v_cm,
                # Historical names: these are contact ANGLES in degrees.
                "ca_left": self.theta_left,
                "ca_right": self.theta_right,
                "cm_x": self.cm_x,
                "cm_y": self.cm_y,
                "Ca": self.ca,
                "Ca_cll_left": self.ca_cll_left,
                "Ca_cll_right": self.ca_cll_right,
                "Ca_cm": self.ca_cm,
                "Ca_norm": self.ca_norm,
                "Re": self.re,
                "sigma_lg": self.scales.sigma_primary,
                "sigma_lg_source": self.scales.sigma_source,
                "Ca_analytical": self.ca_analytical,
            },
            columns=list(CSV_COLUMNS),
        )


#: Per-snapshot metric memo. Animation builds a figure per frame from a growing
#: prefix of the run's snapshots, so every frame re-derives all preceding ones —
#: without this, a run of N snapshots costs N(N+1)/2 reads instead of N. Entries
#: are 9-float tuples, so the cache holds a long run for negligible memory.
_MAX_CACHED_SNAPSHOTS = 4096
_SnapshotKey = tuple[str, int, int, MetricScales]
_SnapshotMetrics = tuple[float, float, float, float, float, float, float, float, float]
_SNAPSHOT_CACHE: OrderedDict[_SnapshotKey, _SnapshotMetrics | None] = OrderedDict()


def _read_snapshot(path: Path, scales: MetricScales) -> _SnapshotMetrics | None:
    """Memoized :func:`_read_snapshot_uncached`, keyed on file identity and *scales*.

    The key carries the file's size and mtime, so a rewritten snapshot is
    re-read rather than served stale.
    """
    stat = path.stat()
    key: _SnapshotKey = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, scales)
    if key in _SNAPSHOT_CACHE:
        _SNAPSHOT_CACHE.move_to_end(key)
        return _SNAPSHOT_CACHE[key]

    metrics = _read_snapshot_uncached(path, scales)
    _SNAPSHOT_CACHE[key] = metrics
    while len(_SNAPSHOT_CACHE) > _MAX_CACHED_SNAPSHOTS:
        _SNAPSHOT_CACHE.popitem(last=False)
    return metrics


def _read_snapshot_uncached(path: Path, scales: MetricScales) -> _SnapshotMetrics | None:
    """Extract one snapshot's metrics, deriving any absent from the density field."""
    with np.load(path) as raw:
        if "rho" not in raw:
            return None
        rho_2d = extract_rho_2d(np.asarray(raw["rho"]))
        theta_l, theta_r, cll_l, cll_r = optional_contact_metrics(raw)
        u_x, u_y = extract_velocity_components_2d(np.asarray(raw["u"])) if "u" in raw else (None, None)

    if cll_l is None or cll_r is None:
        cll_l, cll_r = contact_lines_from_rho(rho_2d, scales.rho_mean, scales.wall_edge)
    if theta_l is None or theta_r is None:
        theta_l, theta_r = contact_angles_from_rho(rho_2d, scales.rho_mean, scales.wall_edge)

    if u_x is None or u_y is None:
        avg_ux, avg_uy = 0.0, 0.0
    else:
        avg_ux, avg_uy = mean_velocity_in_liquid(u_x, u_y, rho_2d, scales.rho_mean)

    cm_x, cm_y = center_of_mass(rho_2d, scales.rho_mean)
    avg_x = avg_x_location(rho_2d, scales.rho_mean, scales.offset_x)
    return avg_ux, avg_uy, avg_x, cll_l, cll_r, theta_l, theta_r, cm_x, cm_y


def compute_droplet_series(
    files: Sequence[Path],
    config: SimulationConfig,
) -> DropletSeries | None:
    """Compute droplet metrics across *files*.

    Returns ``None`` on a capability failure — no parseable snapshots, or a
    config without the multiphase parameters needed to define an interface.
    This function has no opinion about ``sim_type``; gating which runs get
    analysed is the caller's decision.
    """
    scales = resolve_scales(config)
    if scales is None:
        return None

    ordered = sorted(files, key=parse_timestep_from_path)
    rows: list[tuple[float, ...]] = []
    iterations: list[int] = []
    for path in ordered:
        step = parse_timestep(path.stem)
        if step is None:
            continue
        metrics = _read_snapshot(path, scales)
        if metrics is None:
            continue
        iterations.append(step)
        rows.append(metrics)

    if not rows:
        return None

    columns = np.asarray(rows, dtype=float).T
    return DropletSeries(
        iteration=np.asarray(iterations, dtype=int),
        avg_u_x=columns[0],
        avg_u_y=columns[1],
        avg_x_location=columns[2],
        cll_left=columns[3],
        cll_right=columns[4],
        theta_left=columns[5],
        theta_right=columns[6],
        cm_x=columns[7],
        cm_y=columns[8],
        scales=scales,
    )


_MAX_CACHED_RUNS = 8
_Key = tuple[tuple[str, ...], tuple[object, ...]]
_CACHE: OrderedDict[_Key, DropletSeries | None] = OrderedDict()


def _config_fingerprint(config: SimulationConfig) -> tuple[object, ...]:
    """Exactly the config inputs that can change a computed series.

    Anything absent here provably cannot affect the result, so including more
    would only cause spurious cache misses.
    """
    chem = config.chemical_step_config
    step_location = chem.get("chemical_step_location") if isinstance(chem, dict) else None
    gravity = config.gravity_force
    incl = gravity.get("inclination_angle_deg") if isinstance(gravity, dict) else None
    return (
        config.sim_type,
        config.rho_l,
        config.rho_v,
        config.tau,
        config.kappa,
        config.interface_width,
        config.save_interval,
        tuple(config.grid_shape),
        config.init_dir,
        step_location,
        incl,
        config.extra.get("surface_tension"),
    )


def series_for_files(files: Sequence[Path], config: SimulationConfig) -> DropletSeries | None:
    """Cached :func:`compute_droplet_series` keyed on file set and config."""
    key: _Key = (tuple(str(Path(f).resolve()) for f in files), _config_fingerprint(config))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    series = compute_droplet_series(files, config)
    _CACHE[key] = series
    _CACHE.move_to_end(key)
    while len(_CACHE) > _MAX_CACHED_RUNS:
        _CACHE.popitem(last=False)
    return series


def droplet_series_for_run(run_dir: str | Path, config: SimulationConfig) -> DropletSeries | None:
    """Cached droplet series for every ``data/timestep_*.npz`` under *run_dir*."""
    data_dir = Path(run_dir) / "data"
    if not data_dir.exists():
        return None
    files = sorted(data_dir.glob("timestep_*.npz"), key=parse_timestep_from_path)
    if not files:
        return None
    return series_for_files(files, config)


def clear_series_cache() -> None:
    """Drop every cached series and per-snapshot metric. Intended for tests."""
    _CACHE.clear()
    _SNAPSHOT_CACHE.clear()
