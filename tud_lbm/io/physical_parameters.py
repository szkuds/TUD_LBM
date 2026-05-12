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
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tud_lbm.config.simulation_config import SimulationConfig

_CS2 = 1.0 / 3.0  # Speed of sound squared for D2Q9/D3Q19


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


def _resolve_gravity_value(config: SimulationConfig) -> float | None:
    """Resolve gravity from config.g or known force dictionaries."""
    if config.g is not None:
        return float(config.g)

    for force_name in ("gravity_force", "gravity_masked_force"):
        force_dict = getattr(config, force_name, None)
        if force_dict and isinstance(force_dict, dict) and "force_g" in force_dict:
            return float(force_dict["force_g"])
    return None


def _derive_multiphase_parameters(config: SimulationConfig) -> tuple[float, float] | None:
    """Return (drho, gamma) when multiphase parameters are available and valid."""
    has_params = all(x is not None for x in (config.kappa, config.interface_width, config.rho_l, config.rho_v))
    if not has_params or config.interface_width == 0:
        return None

    drho = float(config.rho_l) - float(config.rho_v)
    gamma = (2.0 / 3.0) * (float(config.kappa) / float(config.interface_width)) * (drho**2)
    return drho, gamma


def _format_bond_number_row(config: SimulationConfig, drho: float, gamma: float, g_val: float) -> str:
    """Build Bond-number row with contact-line length or grid-x fallback."""
    cl_length = _get_setup_contact_line_length(config)
    if cl_length is not None:
        length = cl_length
        bo = (drho * g_val * (length**2)) / gamma
        return _row("Bo (Bond number):", f"{bo:.6g}  [ΔρgL²/gamma, L={length:.4g} (contact line)]")

    length = float(config.grid_shape[0])
    bo = (drho * g_val * (length**2)) / gamma
    return _row("Bo (Bond number):", f"{bo:.6g}  [ΔρgL²/gamma, L={length} (grid_x)]")


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

    g_val = _resolve_gravity_value(config)
    if g_val is None:
        return

    lines.append(_format_bond_number_row(config, drho, gamma, g_val))


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
        f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
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
