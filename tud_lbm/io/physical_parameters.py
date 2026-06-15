"""Write a human-readable physical parameter overview to disk.

Generates ``physical_parameters.txt`` in the run directory whenever a
:class:`~tud_lbm.io.SimulationIO` is created with a config.  Unlike the
saved TOML (machine-readable, round-trippable), this file is intended to
be read directly by a human and includes derived quantities such as
kinematic viscosity.

Public API::

    from tud_lbm.io.physical_parameters import write_physical_parameters
    write_physical_parameters(config, "/path/to/run/physical_parameters.txt")
"""

from __future__ import annotations
import math
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from tud_lbm.config.simulation_config import SimulationConfig

_CS2 = 1.0 / 3.0  # Speed of sound squared for D2Q9/D3Q19

# Prefix remaps for analysing runs downloaded off DelftBlue: data stored under
# /scratch/user_cluster/LBM/ on the cluster lives under /Users/local_user/ locally.
_INIT_PATH_REMAPS: tuple[tuple[str, str], ...] = (
    ("/scratch/sbszkudlarek/LBM/26_TUD_LBM/TUD_LBM_data/", "/Users/sbszkudlarek/TUD_LBM_data/DB/"),
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


def _contact_line_length_from_rho(rho: np.ndarray, rho_mean: float) -> float | None:
    """Return setup contact-line spacing from a rho field using wall-row transitions."""
    try:
        row = np.asarray(rho[:, 0, 0, 0, 0], dtype=float)

        mask = (row < float(rho_mean)).astype(np.int32)
        diff = np.diff(mask)
        left_hits = np.nonzero(diff == -1)[0]
        right_hits = np.nonzero(diff == 1)[0]
        if left_hits.size == 0 or right_hits.size == 0:
            return None

        idx_left = int(left_hits[0])
        idx_right = int(right_hits[-1] + 1)
        if idx_left + 1 >= row.size or idx_right - 1 < 0 or idx_right >= row.size:
            return None

        denom_left = row[idx_left + 1] - row[idx_left]
        denom_right = row[idx_right - 1] - row[idx_right]
        if math.isclose(denom_left, 0.0) or math.isclose(denom_right, 0.0):
            return None

        x_left = idx_left + ((rho_mean - row[idx_left]) / denom_left)
        x_right = idx_right - ((rho_mean - row[idx_right]) / denom_right)
        length = float(x_right - x_left)
        if length > 0.0:
            return length
        return None  # noqa: TRY300
    except (IndexError, TypeError, ValueError):
        return None


def _get_contact_line_length_from_file(config: SimulationConfig) -> float | None:
    """Load rho from NPZ and estimate setup contact-line spacing for init_from_file."""
    init = config.initialisation if isinstance(config.initialisation, dict) else {}
    npz_path = _resolve_npz_path(config.init_dir or init.get("npz_path"))
    if not npz_path:
        return None

    try:
        with np.load(npz_path) as data:
            if "rho" not in data:
                return None
            rho = np.asarray(data["rho"])

        if config.rho_l is not None and config.rho_v is not None:
            rho_mean = 0.5 * (float(config.rho_l) + float(config.rho_v))
        else:
            rho_mean = float(np.mean(rho))
        return _contact_line_length_from_rho(rho, rho_mean)
    except (KeyError, OSError, TypeError, ValueError):
        return None


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


def _resolve_length_for_dimensionless_numbers(config: SimulationConfig) -> tuple[float, str]:
    """Resolve shared length scale and annotation for Oh/Bo rows."""
    cl_length = _get_setup_contact_line_length(config)
    if cl_length is not None:
        if config.init_type == "init_from_file":
            return cl_length, f"L={cl_length:.4g} (init_from_file)"
        return cl_length, f"L={cl_length:.4g} (contact line)"

    length = float(config.grid_shape[0])
    return length, f"L={length} (grid_x)"


def _format_ohnesorge_number_row(config: SimulationConfig, gamma: float, length: float, length_label: str) -> str:
    """Build Ohnesorge-number row from lattice kinematic viscosity."""
    nu = _nu(float(config.tau))
    if config.rho_l is None:
        msg = "rho_l is required for Ohnesorge number"
        raise ValueError(msg)
    oh = nu / (gamma * length * config.rho_l) ** 0.5
    return _row("Oh (Ohnesorge number):", f"{oh:.6g}  [ν√(ρ_l/(γL)), {length_label}]")


def _format_bond_number_row(
    delta_rho_phases: float,
    gamma: float,
    g_val: float,
    length: float,
    length_label: str,
    angle_deg: float = 0.0,
) -> list[str]:
    """Build Bond-number row(s) from shared length scale."""
    bo = (delta_rho_phases * (length**2) * g_val) / gamma
    angle_rad = math.radians(angle_deg)
    bo_normal = bo * math.cos(angle_rad)
    bo_tangential = bo * math.sin(angle_rad)
    return [
        _row("Bo (Bond number):", f"{bo:.6g}  [(ΔρgL²)/γ, {length_label}]"),
        _row("Bo_perp (Bond normal):", f"{bo_normal:.6g}  [(Δρ*g*cos({angle_deg:.4g}deg)*L^2)/gamma, {length_label}]"),
        _row(
            "Bo_parallel (Bond tangential):",
            f"{bo_tangential:.6g}  [(Δρ*g*sin({angle_deg:.4g}deg)*L^2)/gamma, {length_label}]",
        ),
    ]


def _format_archimedes_number_row(
    drho: float, g_val: float, length: float, length_label: str, nu: float, rho_l: float
) -> str:
    """Build Archimedes-number row: Ar = gL^3Δρ / (ν^2ρ_l)."""
    ar = (g_val * (length**3) * drho) / ((nu**2) * rho_l)
    return _row("Ar (Archimedes number):", f"{ar:.6g}  [gL³Δρ/(ν²ρ_l), {length_label}]")


def _add_multiphase_section(lines: list[str], config: SimulationConfig) -> None:
    if "multiphase" not in config.sim_type:
        return
    lines.append(_section("Multiphase"))
    lines.append(_row("EOS:", config.eos or "-"))
    lines.append(_row("kappa:", config.kappa))
    lines.append(_row("rho_liquid:", config.rho_l))
    lines.append(_row("rho_vapour:", config.rho_v))
    lines.append(_row("Interface width:", config.interface_width))
    if config.g is not None:
        lines.append(_row("g (gravity):", config.g))

    derived = _derive_multiphase_parameters(config)
    if derived is None:
        return
    drho, gamma = derived
    lines.append(_row("gamma (surface tension):", f"{gamma:.6g}  [2/3(κ/W)(Δρ)²]"))

    length, length_label = _resolve_length_for_dimensionless_numbers(config)
    lines.append(_format_ohnesorge_number_row(config, gamma, length, length_label))

    g_val = _resolve_gravity_value(config)
    if g_val is None:
        return

    angle_deg = _resolve_gravity_inclination(config)
    lines.extend(_format_bond_number_row(drho, gamma, g_val, length, length_label, angle_deg))
    nu = _nu(float(config.tau))
    if config.rho_l is None:
        msg = "rho_l is required for Archimedes number"
        raise ValueError(msg)
    lines.append(_format_archimedes_number_row(drho, g_val, length, length_label, nu, float(config.rho_l)))


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
        config: Validated :class:`~tud_lbm.config.simulation_config.SimulationConfig`.
        path:   Destination file path (typically ``<run_dir>/physical_parameters.txt``).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(build_overview(config), encoding="utf-8")
