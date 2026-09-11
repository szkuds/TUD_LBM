"""Write a human-readable physical parameter overview to disk.

Generates ``physical_parameters.txt`` in the run directory whenever a
:class:`~src.simulation_io.SimulationIO` is created with a config.  Unlike the
saved TOML (machine-readable, round-trippable), this file is intended to
be read directly by a human and includes derived quantities such as
kinematic viscosity.

Public API::

    from src.simulation_io.analysis.physical_parameters import write_physical_parameters
    write_physical_parameters(config, "/path/to/run/physical_parameters.txt")
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast
import numpy as np
from src.registry import get_operators
from src.simulation_io.analysis.physical_parameters import (
    numbers as _numbers,  # noqa: F401  (registers the dimensionless operators)
)
from src.simulation_io.analysis.physical_parameters._inputs import DimensionlessInputs
from src.simulation_io.analysis.physical_parameters.numbers._bond import BondNumbers
from src.simulation_io.analysis.physical_parameters.numbers._bond import compute_bond_numbers
from src.simulation_io.analysis.physical_parameters.numbers._buoyancy import compute_archimedes_number
from src.simulation_io.analysis.physical_parameters.numbers._buoyancy import compute_reynolds_number

if TYPE_CHECKING:
    from collections.abc import Mapping
    from src.config.simulation_config import SimulationConfig
    from src.registry import OperatorEntry
    from src.simulation_io.analysis.physical_parameters._inputs import DimensionlessNumberOperator

# Re-exported so the formulas keep their historical import site while living
# beside the operators that register them.
__all__ = [
    "BondNumbers",
    "DimensionlessNumbers",
    "build_overview",
    "compute_archimedes_number",
    "compute_bond_numbers",
    "compute_dimensionless_numbers",
    "compute_reynolds_number",
    "dimensionless_keys",
    "dimensionless_label",
    "inclusion_mask_from_rho",
    "measure_init_phase_densities",
    "resolve_dimensionless_inputs",
    "write_physical_parameters",
]

_CS2 = 1.0 / 3.0  # Speed of sound squared for D2Q9/D3Q19

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


def _resolve_npz_path(path: str | None) -> str | None:
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


def _nu(tau: float) -> float:
    """Kinematic viscosity from relaxation time: nu = cs2 * (tau - 0.5)."""
    return _CS2 * (tau - 0.5)


def _section(title: str) -> str:
    return f"\n{title}\n" + "-" * len(title)


def _row(label: str, value: object, indent: int = 2) -> str:
    pad = " " * indent
    return f"{pad}{label:<26}{value}"


def _add_simulation_section(lines: list[str], config: SimulationConfig) -> None:
    lines.append(_section("Simulation"))
    if config.simulation_name:
        lines.append(_row("Name:", config.simulation_name))
    lines.append(_row("Type:", config.sim_type))
    lines.append(_row("Lattice:", config.lattice_type))


def _add_grid_section(lines: list[str], config: SimulationConfig) -> None:
    lines.append(_section("Grid"))
    shape = config.grid_shape
    display_shape = shape[:2] if shape[2] == 1 else shape
    lines.append(_row("Shape:", " x ".join(str(n) for n in display_shape)))
    lines.append(_row("Timesteps:", config.nt))
    lines.append(_row("Save interval:", config.save_interval))


def _add_collision_section(lines: list[str], config: SimulationConfig) -> None:
    lines.append(_section("Collision"))
    tau = config.tau
    lines.append(_row("Collision scheme:", config.collision_scheme))
    lines.append(_row("tau:", tau))
    lines.append(_row("nu (kinematic viscosity):", f"{_nu(tau):.6g}  [cs2*(tau-0.5)]"))


def _get_setup_contact_line_length(config: SimulationConfig) -> float | None:
    """Calculate the distance between the two contact lines at setup."""
    if config.init_type == "init_from_file":
        return _get_contact_line_length_from_file(config)

    init = config.initialisation
    if not init or not isinstance(init, dict):
        return None
    try:
        centres = init.get("centres", [])
        radii = init.get("radii", [])
        if not centres or not radii:
            return None

        nx = float(config.grid_shape[0])
        ny = float(config.grid_shape[1])
        min_dim = min(nx, ny)

        fx, fy = float(centres[0][0]), float(centres[0][1])
        r = float(radii[0])

        _r = r * min_dim
        # Compute distance to the closest bounding wall (0, nx) or (0, ny)
        dist_x = min(fx * nx, (1.0 - fx) * nx)
        dist_y = min(fy * ny, (1.0 - fy) * ny)
        wall_dist = min(dist_x, dist_y)

        val = _r**2 - wall_dist**2
        if val > 0:
            return 2.0 * (val**0.5)
    except (IndexError, ValueError, TypeError):
        pass
    return None


#: A wall row must cross ``rho_mean`` at least twice to bracket a contact line.
_MIN_CROSSINGS = 2

#: Smallest single-cell density step across which the ``rho_mean`` crossing is
#: interpolated. A flatter step makes the interpolation below explode, so the
#: row yields no usable length. An explicit absolute tolerance is required:
#: ``math.isclose(x, 0.0)`` defaults to ``abs_tol=0.0`` and so reduces to
#: ``x == 0.0``, which is the one case that never needed guarding.
_MIN_CROSSING_SLOPE = 1e-9

#: Order in which ``bc_config`` is scanned for the wetting wall. Matches
#: :func:`src.operators.wetting._edge_config._resolve_wetting_edges`, so the wall
#: reported here is the one the solver actually measured at.
_EDGE_SCAN_ORDER = ("bottom", "top", "left", "right")


def resolve_wall_edge(config: SimulationConfig) -> str:
    """The single wall marked ``"wetting"`` in ``bc_config``, else ``"bottom"``.

    Config validation guarantees exactly one wetting wall for wetting runs;
    non-wetting runs fall back to ``"bottom"``, for which the canonical
    transform is the identity.

    Defined here rather than in :mod:`..droplet_metrics._scales` so that this
    module — which ``_scales`` already imports from lazily — does not have to
    import back into ``droplet_metrics``.
    """
    bc = config.bc_config or {}
    for edge in _EDGE_SCAN_ORDER:
        if bc.get(edge) == "wetting":
            return edge
    return "bottom"


def _contact_line_length_from_rho(rho: np.ndarray, rho_mean: float, wall_edge: str = "bottom") -> float | None:
    """Return setup contact-line spacing from a rho field using wall-row transitions.

    *wall_edge* names the wetting wall. The row is taken from the wall-aligned
    canonical view for that edge, so a droplet or bubble on ``"top"``,
    ``"left"`` or ``"right"`` is measured at its own wall. Slicing ``y = 0``
    unconditionally sampled the far wall for those runs, found no crossings,
    and silently pushed ``resolve_r_zero`` onto its nominal-radius fallback —
    which then propagated into ``L_eff`` and every dimensionless number.
    """
    # Lazily, and from droplet_metrics rather than the JAX original in
    # `operators.wetting._canonical_view`: this is a numpy field, and the twin
    # there is already annotated and tested for numpy. The import is deferred
    # because `droplet_metrics._scales` imports back into this module the same
    # way, so neither may reach the other at module load.
    from src.simulation_io.analysis.droplet_metrics._snapshot import to_canonical_2d

    try:
        plane = np.asarray(rho[:, :, 0, 0, 0], dtype=float)
        row = np.asarray(to_canonical_2d(plane, wall_edge)[:, 0], dtype=float)

        mask = (row < float(rho_mean)).astype(np.int32)
        diff = np.diff(mask)
        # Positional outermost crossings, so a bubble (whose density steps the
        # other way) yields a positive spacing rather than being discarded.
        hits = np.nonzero(np.abs(diff) == 1)[0]
        if hits.size < _MIN_CROSSINGS:
            return None

        idx_left = int(hits[0])
        idx_right = int(hits[-1])
        if idx_left + 1 >= row.size or idx_right + 1 >= row.size:
            return None

        denom_left = row[idx_left + 1] - row[idx_left]
        denom_right = row[idx_right + 1] - row[idx_right]
        if abs(denom_left) <= _MIN_CROSSING_SLOPE or abs(denom_right) <= _MIN_CROSSING_SLOPE:
            return None

        x_left = idx_left + ((rho_mean - row[idx_left]) / denom_left)
        x_right = idx_right + ((rho_mean - row[idx_right]) / denom_right)
        length = float(x_right - x_left)
        if length > 0.0:
            return length
        return None  # noqa: TRY300
    except (IndexError, TypeError, ValueError):
        return None


class _InitField(NamedTuple):
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


def _load_init_rho(config: SimulationConfig) -> _InitField | None:
    """Load the init rho field from NPZ and measure its densities, for init_from_file.

    Returns ``None`` when no file resolves, it holds no ``rho``, or the field is
    empty or non-finite.
    """
    npz_path = _resolve_npz_path(config.init_dir or config.initialisation.get("npz_path"))
    if not npz_path:
        return None

    try:
        stat = Path(npz_path).stat()
    except OSError:
        return None
    return _load_field_cached(npz_path, (stat.st_mtime_ns, stat.st_size))


@lru_cache(maxsize=4)
def _load_field_cached(npz_path: str, _stat_key: tuple[int, int]) -> _InitField | None:
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

    return _measure_field(rho)


def _measure_field(rho: np.ndarray) -> _InitField | None:
    """Return *rho* with its extrema, midpoint and contrast, or None when unusable."""
    try:
        values = np.asarray(rho, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            return None
        rho_min = float(np.min(values))
        rho_max = float(np.max(values))
    except (TypeError, ValueError):
        return None
    return _InitField(
        rho=rho,
        rho_min=rho_min,
        rho_max=rho_max,
        rho_mean=0.5 * (rho_max + rho_min),
        drho=rho_max - rho_min,
    )


def measure_init_phase_densities(config: SimulationConfig) -> tuple[float, float] | None:
    """Return ``(rho_min, rho_max)`` measured off the run's init NPZ, or None.

    The public face of :func:`_load_init_rho` for callers outside this module —
    :mod:`src.operators.force._gravity_masked` bands its phase indicator on the
    same measured densities this file reports the buoyancy contrast with, so the
    contrast a run injects and the one its Bond number quotes cannot diverge.
    Returns None when the run has no init file, or it holds no usable ``rho``.
    """
    field = _load_init_rho(config)
    if field is None:
        return None
    return field.rho_min, field.rho_max


def _get_contact_line_length_from_file(config: SimulationConfig) -> float | None:
    """Load rho from NPZ and estimate setup contact-line spacing for init_from_file."""
    field = _load_init_rho(config)
    if field is None:
        return None
    return _contact_line_length_from_rho(field.rho, field.rho_mean, resolve_wall_edge(config))


def _get_setup_droplet_area(config: SimulationConfig) -> float | None:
    """Analytic droplet area from init geometry: circle clipped by the nearest wall."""
    init = config.initialisation
    if not init or not isinstance(init, dict):
        return None
    try:
        centres = init.get("centres", [])
        radii = init.get("radii", [])
        if not centres or not radii:
            return None

        nx = float(config.grid_shape[0])
        ny = float(config.grid_shape[1])

        fx, fy = float(centres[0][0]), float(centres[0][1])
        r = float(radii[0]) * min(nx, ny)

        dist_x = min(fx * nx, (1.0 - fx) * nx)
        dist_y = min(fy * ny, (1.0 - fy) * ny)
        wall_dist = min(dist_x, dist_y)

        area = math.pi * r**2
        if wall_dist < r:
            # Subtract the circular segment cut off by the nearest wall.
            area -= r**2 * math.acos(wall_dist / r) - wall_dist * math.sqrt(r**2 - wall_dist**2)
        if area > 0.0:
            return area
    except (IndexError, ValueError, TypeError):
        pass
    return None


def inclusion_mask_from_rho(rho: np.ndarray, rho_mean: float) -> np.ndarray | None:
    """Boolean ``(nx, ny)`` mask of the inclusion in the z=0 plane.

    Both topologies put the inclusion in the minority phase, so thresholding at
    ``rho_mean`` and keeping the smaller side selects a droplet or a bubble
    without being told which it is. Taking ``rho > rho_mean`` unconditionally
    selected the *continuous* phase of a bubble run — nearly the whole domain —
    which inflated ``L_eff`` and every dimensionless number built on it.

    Cells exactly at ``rho_mean`` fall in neither phase. Returns ``None`` when
    the field cannot be sliced or neither phase is present.
    """
    try:
        plane = np.asarray(rho[:, :, 0, 0, 0], dtype=float)
        liquid = plane > rho_mean
        vapour = plane < rho_mean
    except (IndexError, TypeError, ValueError):
        return None
    mask = liquid if np.count_nonzero(liquid) <= np.count_nonzero(vapour) else vapour
    return mask if np.any(mask) else None


def _inclusion_area_from_rho(rho: np.ndarray, rho_mean: float) -> float | None:
    """Inclusion area in the z=0 plane, as the cell count of :func:`inclusion_mask_from_rho`."""
    mask = inclusion_mask_from_rho(rho, rho_mean)
    return None if mask is None else float(np.count_nonzero(mask))


def _get_droplet_area(config: SimulationConfig) -> tuple[float, str] | None:
    """Return ``(area, source)`` for the setup droplet, or None when unavailable."""
    if config.init_type == "init_from_file":
        field = _load_init_rho(config)
        area = _inclusion_area_from_rho(field.rho, field.rho_mean) if field is not None else None
        return (area, "init_from_file") if area is not None else None

    area = _get_setup_droplet_area(config)
    return (area, "init geometry") if area is not None else None


def _ensure_single_gravity_force_source(config: SimulationConfig) -> None:
    """Reject configs that define both gravity force variants simultaneously."""
    if config.gravity_force is not None and config.gravity_masked_force is not None:
        msg = "Only one gravity force can be applied: set either gravity_force or gravity_masked_force, not both."
        raise ValueError(msg)


def _resolve_gravity_value(config: SimulationConfig) -> float | None:
    """Resolve gravity from config.g or known force dictionaries."""
    _ensure_single_gravity_force_source(config)

    if config.g is not None:
        return float(config.g)

    for force_name in ("gravity_force", "gravity_masked_force"):
        force_dict = getattr(config, force_name, None)
        if force_dict and isinstance(force_dict, dict) and "force_g" in force_dict:
            return float(force_dict["force_g"])
    return None


def _resolve_gravity_inclination(config: SimulationConfig) -> float:
    """Return inclination_angle_deg from the active force config, or 0.0."""
    _ensure_single_gravity_force_source(config)

    for force_name in ("gravity_force", "gravity_masked_force"):
        force_dict = getattr(config, force_name, None)
        if force_dict and isinstance(force_dict, dict):
            return float(force_dict.get("inclination_angle_deg", 0.0))
    return 0.0


def _derive_multiphase_parameters(config: SimulationConfig) -> tuple[float, float] | None:
    """Return (delta_rho_phases, gamma) when multiphase parameters are available and valid."""
    if config.kappa is None or config.interface_width is None or config.rho_l is None or config.rho_v is None:
        return None
    if config.interface_width == 0:
        return None

    drho = float(config.rho_l) - float(config.rho_v)
    gamma = (2.0 / 3.0) * (float(config.kappa) / float(config.interface_width)) * (drho**2)
    return drho, gamma


# EOS without a closed-form surface tension; sigma is measured at run time and
# stored in config.extra by src.simulation_io.analysis.surface_tension.
_EOS_REQUIRING_CALIBRATION = frozenset({"carnahan-starling"})


def _resolve_surface_tension(config: SimulationConfig) -> tuple[float, float, str] | None:
    """Return ``(drho, gamma, source)`` preferring a measured value.

    ``source`` is "measured" or "analytical". Returns ``None`` when no value is
    available (e.g. a calibration-only EOS that has not been measured yet).

    The ``drho`` returned here is the *prescribed* contrast, and belongs only to
    the closed form ``gamma = 2/3 (kappa/W) drho^2``. The buoyancy contrast in
    Bo/Ar/Re comes from :func:`_resolve_buoyancy_delta_rho` instead.
    """
    if config.rho_l is None or config.rho_v is None:
        return None
    drho = float(config.rho_l) - float(config.rho_v)

    measured = config.extra.get("surface_tension")
    if measured is not None:
        return drho, float(measured), "measured"
    if config.eos in _EOS_REQUIRING_CALIBRATION:
        return None
    derived = _derive_multiphase_parameters(config)
    if derived is None:
        return None
    return derived[0], derived[1], "analytical"


def _resolve_buoyancy_delta_rho(config: SimulationConfig) -> tuple[float, str] | None:
    """Return ``(drho, source)`` for buoyancy, measured off the init field when there is one.

    Deliberately separate from the ``drho`` :func:`_resolve_surface_tension`
    returns. That one feeds the closed form ``gamma = 2/3 (kappa/W) drho^2``,
    which is derived for the *prescribed* double-well and is computed from the
    same prescribed densities in ``droplet_metrics/_scales.py``; measuring it
    there would diverge the two. The buoyancy contrast in Bo/Ar/Re is a property
    of the field, so it is measured wherever a field exists.
    """
    if config.init_type == "init_from_file":
        field = _load_init_rho(config)
        if field is not None:
            return field.drho, "measured"
    if config.rho_l is not None and config.rho_v is not None:
        return float(config.rho_l) - float(config.rho_v), "config"
    return None


def _resolve_length_for_dimensionless_numbers(config: SimulationConfig) -> tuple[float, str]:
    """Resolve shared length scale and annotation for Oh/Bo rows.

    Uses the effective dispersed phase radius L_eff = sqrt(Area/pi) from the setup
    dispersed phase area, falling back to grid_x when no dispersed area can be resolved.
    """
    dispersed_phase_resolved = _get_droplet_area(config)
    if dispersed_phase_resolved is not None:
        area, source = dispersed_phase_resolved
        l_eff = math.sqrt(area / math.pi)
        return l_eff, f"L_eff={l_eff:.4g} (sqrt(A/pi), {source})"

    length = float(config.grid_shape[0])
    return length, f"L={length} (grid_x)"


# ---------------------------------------------------------------------------
# Dimensionless numbers
# ---------------------------------------------------------------------------
#
# The numbers themselves live one-per-file under ``numbers/`` and self-register
# under registry kind ``dimensionless``. Nothing below names a specific number:
# this section resolves the inputs they share, evaluates whatever is registered,
# and renders the results. Adding a number is adding a file.


def resolve_dimensionless_inputs(config: SimulationConfig) -> DimensionlessInputs | None:
    """Resolve everything the registered dimensionless numbers are built from.

    Mirrors the resolution sequence in :func:`_add_multiphase_section`: surface
    tension via :func:`_resolve_surface_tension`, buoyancy contrast via
    :func:`_resolve_buoyancy_delta_rho`, length via
    :func:`_resolve_length_for_dimensionless_numbers`, gravity via
    :func:`_resolve_gravity_value` and :func:`_resolve_gravity_inclination`.

    Returns ``None`` only when *nothing* could be computed -- no surface tension
    (a calibration-only EOS not yet measured), no density contrast, or no
    ``rho_l``. Missing *gravity* is deliberately not such a case: it is carried
    through as ``g=None`` so ``Oh`` and ``La``, which need none, still resolve
    while the buoyancy-driven numbers report themselves unresolved.
    """
    resolved = _resolve_surface_tension(config)
    if resolved is None:
        return None
    _drho_config, gamma, gamma_source = resolved

    buoyancy = _resolve_buoyancy_delta_rho(config)
    if buoyancy is None or config.rho_l is None:
        return None
    drho, drho_source = buoyancy

    length, length_label = _resolve_length_for_dimensionless_numbers(config)
    g_val = _resolve_gravity_value(config)
    return DimensionlessInputs(
        gamma=gamma,
        gamma_source=gamma_source,
        drho=drho,
        drho_source=drho_source,
        length=length,
        length_label=length_label,
        nu=_nu(float(config.tau)),
        rho_l=float(config.rho_l),
        g=g_val,
        angle_deg=None if g_val is None else _resolve_gravity_inclination(config),
    )


def _dimensionless_entries() -> list[OperatorEntry]:
    """Registered dimensionless numbers, in display order.

    Sorted by the ``order`` metadata, not by registration order: the latter is
    module import order, which would silently reshuffle ``physical_parameters.txt``
    the moment a file is added to ``numbers/``.
    """
    entries = get_operators("dimensionless").values()
    return sorted(entries, key=lambda entry: (_meta_of(entry).get("order", 0), entry.name))


def _meta_of(entry: OperatorEntry) -> dict[str, object]:
    """The entry's metadata, never ``None``."""
    return entry.metadata or {}


def dimensionless_keys() -> tuple[str, ...]:
    """Every registered dimensionless number's key, in display order."""
    return tuple(entry.name for entry in _dimensionless_entries())


def dimensionless_label(key: str) -> str:
    r"""The mathtext label for *key*, e.g. ``r"$\mathrm{Oh}$"``.

    Falls back to the key itself, so a number registered without a label still
    plots rather than crashing an axis.
    """
    entry = get_operators("dimensionless").get(key)
    if entry is None:
        msg = f"unknown dimensionless number: {key!r}"
        raise KeyError(msg)
    return str(_meta_of(entry).get("label", key))


def _evaluate_dimensionless(inputs: DimensionlessInputs) -> dict[str, float | None]:
    """Call every registered number, recording a failure as an unresolved value.

    A number that raises must not take down the overview file that is written at
    the start of every run, so the whole set is best-effort.
    """
    values: dict[str, float | None] = {}
    for entry in _dimensionless_entries():
        operator = cast("DimensionlessNumberOperator", entry.target)
        try:
            values[entry.name] = operator(inputs)
        except (ArithmeticError, TypeError, ValueError):
            values[entry.name] = None
    return values


@dataclass(frozen=True)
class DimensionlessNumbers:
    """Every registered dimensionless number for one config, keyed by name.

    Registry-shaped rather than one field per number: adding a file under
    ``numbers/`` adds a key here, and every consumer -- legend labels, regime-map
    axes, the overview file -- reads by key. An unresolvable number is present
    with value ``None`` rather than absent.

    ``inclination_deg`` is not itself dimensionless; it rides along because it is
    what decides whether ``bo`` or ``bo_parallel`` is the Bond number worth
    reporting for a run, and re-deriving it from the config would duplicate the
    force-dict precedence in :func:`_resolve_gravity_inclination`.
    """

    values: Mapping[str, float | None] = field(default_factory=dict)
    inclination_deg: float | None = None

    def get(self, key: str) -> float | None:
        """The value of *key*, or ``None`` when unregistered or unresolvable."""
        return self.values.get(key)


def compute_dimensionless_numbers(config: SimulationConfig) -> DimensionlessNumbers:
    """Every registered dimensionless number for one config; never raises.

    Returns an empty set of values when the shared inputs cannot be resolved at
    all (see :func:`resolve_dimensionless_inputs`).
    """
    inputs = resolve_dimensionless_inputs(config)
    if inputs is None:
        return DimensionlessNumbers()
    return DimensionlessNumbers(values=_evaluate_dimensionless(inputs), inclination_deg=inputs.angle_deg)


def _dimensionless_rows(config: SimulationConfig) -> list[str]:
    """One overview row per resolvable dimensionless number, in display order."""
    inputs = resolve_dimensionless_inputs(config)
    if inputs is None:
        return []
    values = _evaluate_dimensionless(inputs)
    rows: list[str] = []
    for entry in _dimensionless_entries():
        value = values.get(entry.name)
        if value is None:
            continue
        meta = _meta_of(entry)
        # Only the gravity-driven numbers use the buoyancy contrast, so only
        # they annotate its provenance.
        scale_label = (
            f"{inputs.length_label}, Δρ {inputs.drho_source}" if meta.get("needs_gravity") else inputs.length_label
        )
        rows.append(
            _row(str(meta.get("row_label", entry.name)), f"{value:.6g}  [{meta.get('formula', '')}, {scale_label}]")
        )
    return rows


def _format_critical_inclination_angle_row(config: SimulationConfig, gamma: float) -> str:
    g_val = _resolve_gravity_value(config)
    if config.chemical_step_config is None or g_val is None or config.rho_l is None or config.rho_v is None:
        msg = "chemical_step_config, a gravity force, rho_l, and rho_v must be set"
        raise RuntimeError(msg)
    ca_adv = math.radians(float(config.chemical_step_config["ca_advancing_pre_step"]))
    ca_rec = math.radians(float(config.chemical_step_config["ca_receding_pre_step"]))
    g = g_val
    radius = float(config.initialisation["radii"][0])
    nx = int(config.grid_shape[0])
    # The drive per unit area is the *net* buoyancy of the inclusion, so it
    # scales with the density contrast, not with rho_l — identical for a bubble
    # and a droplet. The body force itself now sits on the liquid rather than
    # being injected into the inclusion, but the reaction it produces is the
    # same drho*g*a, so this balance is unchanged.
    drho = abs(float(config.rho_l) - float(config.rho_v))

    a = (np.pi * (radius * nx) ** 2) / 2  # Assuming perfectly spherical cap
    hysteresis_force = (np.cos(ca_rec) - np.cos(ca_adv)) * gamma
    sina = hysteresis_force / (g * a * drho)
    a_rad = np.arcsin(sina)
    a_deg = math.degrees(a_rad)

    if -1 <= sina <= 1:
        return _row(
            "Critical Inclination Angle",
            f"{a_deg:.6g}  [arcsin((cos(ca_rec)-cos(ca_adv))*gamma / (g*a*drho))]",
        )

    return _row("Critical Inclination Angle", "This droplet will remain pinned")


def _add_measured_density_rows(lines: list[str], config: SimulationConfig) -> None:
    """Report the densities measured off the init field, when there is one.

    Makes the thresholds the length scale and the buoyancy contrast are built
    from auditable in the file itself, rather than inferred from the config.
    """
    if config.init_type != "init_from_file":
        return
    field = _load_init_rho(config)
    if field is None:
        return
    lines.append(_row("rho_min / rho_max:", f"{field.rho_min:.6g} / {field.rho_max:.6g}  [measured, init NPZ]"))
    lines.append(_row("rho_mean (threshold):", f"{field.rho_mean:.6g}  [(rho_max+rho_min)/2]"))


def _add_buoyancy_delta_rho_row(lines: list[str], config: SimulationConfig) -> None:
    """Append the Δρ row, making the contrast the buoyancy numbers use auditable.

    Printed whenever it resolves, gravity or not: it is a property of the field,
    and reporting it is what lets a reader check the Bond number against the
    density the run actually has.
    """
    buoyancy = _resolve_buoyancy_delta_rho(config)
    if buoyancy is None:
        return
    drho, drho_source = buoyancy
    note = "measured, rho_max-rho_min" if drho_source == "measured" else "config, rho_l-rho_v"
    lines.append(_row("Δρ (buoyancy contrast):", f"{drho:.6g}  [{note}]"))


def _add_multiphase_section(lines: list[str], config: SimulationConfig) -> None:
    if "multiphase" not in config.sim_type:
        return
    lines.append(_section("Multiphase"))
    lines.append(_row("EOS:", config.eos or "-"))
    lines.append(_row("kappa:", config.kappa))
    lines.append(_row("rho_liquid:", config.rho_l))
    lines.append(_row("rho_vapour:", config.rho_v))
    lines.append(_row("Interface width:", config.interface_width))
    _add_measured_density_rows(lines, config)
    if config.g is not None:
        lines.append(_row("g (gravity):", config.g))

    resolved = _resolve_surface_tension(config)
    if resolved is None:
        if config.eos in _EOS_REQUIRING_CALIBRATION:
            lines.append(_row("gamma (surface tension):", "requires Young–Laplace calibration"))
        return
    _drho_config, gamma, source = resolved
    note = "measured, Young–Laplace" if source == "measured" else "2/3(κ/W)(Δρ)²"
    lines.append(_row("gamma (surface tension):", f"{gamma:.6g}  [{note}]"))

    _add_buoyancy_delta_rho_row(lines, config)
    lines.extend(_dimensionless_rows(config))
    if config.chemical_step_config is not None and _resolve_gravity_value(config) is not None:
        lines.append(_format_critical_inclination_angle_row(config, gamma))


def _add_key_value_section(lines: list[str], title: str, values: dict | None) -> None:
    if not values:
        return
    lines.append(_section(title))
    for k, v in values.items():
        lines.append(_row(f"{k}:", v))


def _add_forces_section(lines: list[str], config: SimulationConfig) -> None:
    force_fields = [
        f for f in ("gravity_force", "electric_force", "gravity_masked_force") if getattr(config, f, None) is not None
    ]
    if not force_fields:
        return
    lines.append(_section("Forces"))
    for fname in force_fields:
        lines.append(f"  {fname}:")
        for k, v in getattr(config, fname).items():
            lines.append(_row(f"{k}:", v, indent=4))


def build_overview(config: SimulationConfig) -> str:
    """Return the full physical parameter overview as a string."""
    lines: list[str] = []
    sep = "=" * 72

    lines += [
        sep,
        "PHYSICAL PARAMETER OVERVIEW",
        f"Generated : {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        sep,
    ]

    _add_simulation_section(lines, config)
    _add_grid_section(lines, config)
    _add_collision_section(lines, config)
    _add_multiphase_section(lines, config)
    _add_key_value_section(lines, "Boundary Conditions", config.bc_config)
    _add_key_value_section(lines, "Wetting", config.wetting_config)
    _add_key_value_section(lines, "Hysteresis", config.hysteresis_config)
    _add_key_value_section(lines, "Chemical Step", config.chemical_step_config)
    _add_forces_section(lines, config)

    lines.append("\n" + sep)
    return "\n".join(lines) + "\n"


def write_physical_parameters(config: SimulationConfig, path: str | Path) -> None:
    """Write ``physical_parameters.txt`` to *path*.

    Args:
        config: Validated :class:`~src.config.simulation_config.SimulationConfig`.
        path:   Destination file path (typically ``<run_dir>/physical_parameters.txt``).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(build_overview(config), encoding="utf-8")
