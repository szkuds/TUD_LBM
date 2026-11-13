"""``[multiphase]``: the section's presence is what makes a run multiphase.

Holds the section's schema, the per-EOS parameter table, the JIT-static
:class:`MultiphaseParams` and the densities the run's phases actually sit at.
Adding an EOS with its own parameters is one row of :data:`EOS_PARAMETERS`.

``rho_l``/``rho_v`` may be omitted for an EOS that registers a coexistence
solver (kind ``"coexistence"``, the Maxwell construction):
:func:`complete_multiphase` fills them from the EOS's own parameters. A
configured density always wins. To sweep ``t_eos``, omit them, since a
configured density does not follow the temperature.

``equilibrium_reference_density`` (``rho_0``) belongs to an equilibrium that
carries a pressure (``wb_hydro_inc``): the pressure is ``rho_0 * (mu - mu_b)`` and
the interparticle force ``-(rho - rho_0) * grad(mu)``. It defaults to the vapour
density at coexistence, and is rejected for any other equilibrium, where it would
change the force with nothing to carry the pressure.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from src.config._checks import registered_names
from src.config._checks import validate_positive
from src.config.equilibrium import uses_reference_density
from src.config.init_field import measure_init_phase_densities

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig

#: Every EOS needs these. ``rho_l``/``rho_v`` are filled by :func:`complete_multiphase`
#: when omitted for an EOS with a registered coexistence solver.
REQUIRED: tuple[str, ...] = ("kappa", "rho_l", "rho_v", "interface_width", "eos")
#: Keys any EOS may carry.
OPTIONAL: tuple[str, ...] = ("g",)
#: ``rho_0`` of an equilibrium that carries a pressure; see the module docstring.
REFERENCE_DENSITY: str = "equilibrium_reference_density"
#: The further parameters each EOS needs, all of them required for it and invalid for any other.
EOS_PARAMETERS: dict[str, tuple[str, ...]] = {
    "double-well": (),
    "carnahan-starling": ("a_eos", "b_eos", "r_eos", "t_eos"),
}


class MultiphaseParams(NamedTuple):
    """Equation-of-state and surface-tension parameters.

    All fields are Python scalars (compile-time constants).
    """

    eos: str
    kappa: float
    rho_l: float
    rho_v: float
    interface_width: int
    g: float | None = None
    a_eos: float | None = None
    b_eos: float | None = None
    r_eos: float | None = None
    t_eos: float | None = None
    equilibrium_reference_density: float | None = None
    #: ``mu_b``: derived by :func:`multiphase_params`, never a section key.
    equilibrium_reference_potential: float | None = None


def is_multiphase(config: SimulationConfig) -> bool:
    """A run is multiphase when it has a ``[multiphase]`` section."""
    return config.multiphase is not None


def _maxwell_pair(completed: dict[str, Any]) -> tuple[float, float] | None:
    """The Maxwell ``(rho_v, rho_l)`` of *completed* to six figures, or ``None`` when it has none.

    ``None`` without an EOS that registers a coexistence solver, or without every
    parameter of its :data:`EOS_PARAMETERS` row. Six significant figures keep the
    saved ``config.toml`` readable, and the surface-tension cache key (which holds
    ``rho_l``/``rho_v``) cannot differ in the last bits between machines.
    """
    eos = completed.get("eos")
    if not isinstance(eos, str):
        return None
    row = EOS_PARAMETERS.get(eos)
    if row is None or not set(row) <= completed.keys():
        return None

    # Imported here: the EOS package imports this module.
    from src.operators.macroscopic.eos import coexistence_densities

    pair = coexistence_densities(eos, completed)
    return None if pair is None else (float(f"{pair[0]:.6g}"), float(f"{pair[1]:.6g}"))


def complete_multiphase(section: dict[str, Any] | None, equilibrium: str | None = None) -> dict[str, Any] | None:
    """*section* without its null keys, and with the values the Maxwell construction supplies.

    A null means the key is not set, and TOML cannot write one. When ``rho_l``
    or ``rho_v`` is missing and the EOS registers a coexistence solver with
    every parameter of its :data:`EOS_PARAMETERS` row present, the missing
    density is the Maxwell value. For an *equilibrium* that carries a pressure,
    an omitted :data:`REFERENCE_DENSITY` is the coexistence vapour density: the
    Maxwell one, else the section's ``rho_v``. A configured value is never
    overwritten; anything else is left to :func:`validate_multiphase`.

    Raises:
        ValueError: If the EOS parameters admit no liquid-vapour coexistence.
    """
    if section is None:
        return None
    completed = {name: value for name, value in section.items() if value is not None}
    needs_reference = uses_reference_density(equilibrium) and REFERENCE_DENSITY not in completed
    if {"rho_l", "rho_v"} <= completed.keys() and not needs_reference:
        return completed
    pair = _maxwell_pair(completed)
    if pair is not None:
        completed.setdefault("rho_v", pair[0])
        completed.setdefault("rho_l", pair[1])
    if needs_reference and "rho_v" in completed:
        completed[REFERENCE_DENSITY] = completed["rho_v"] if pair is None else pair[0]
    return completed


def validate_multiphase(config: SimulationConfig) -> None:
    """Check the section against the common schema and its EOS's row of :data:`EOS_PARAMETERS`."""
    section = config.multiphase
    if section is None:
        return
    for name in REQUIRED:
        if section.get(name) is None:
            msg = f"'{name}' is required for multiphase simulations"
            raise ValueError(msg)

    validate_positive(section["rho_l"], "rho_l")
    validate_positive(section["rho_v"], "rho_v")
    if section["rho_l"] <= section["rho_v"]:
        msg = f"rho_l ({section['rho_l']}) must be greater than rho_v ({section['rho_v']})"
        raise ValueError(msg)
    validate_positive(section["kappa"], "kappa")
    validate_positive(section["interface_width"], "interface_width")

    eos = section["eos"]
    valid_eos = registered_names("src.operators.macroscopic.eos", "eos")
    if eos not in valid_eos:
        msg = f"eos must be one of {sorted(valid_eos)}, got '{eos}'"
        raise ValueError(msg)

    eos_parameters = EOS_PARAMETERS.get(eos, ())
    for name in eos_parameters:
        if section.get(name) is None:
            msg = f"'{name}' is required when eos = '{eos}'"
            raise ValueError(msg)
    unknown = sorted(set(section) - {*REQUIRED, *OPTIONAL, *eos_parameters, REFERENCE_DENSITY})
    if unknown:
        msg = f"[multiphase] has unknown key(s) for eos = '{eos}': {', '.join(unknown)}"
        raise ValueError(msg)
    _validate_reference_density(section, config.equilibrium)


def _validate_reference_density(section: dict[str, Any], equilibrium: str | None) -> None:
    """:data:`REFERENCE_DENSITY` belongs to an equilibrium that carries a pressure, between 0 and ``rho_l``."""
    if not uses_reference_density(equilibrium):
        if REFERENCE_DENSITY in section:
            msg = (
                f"[multiphase] {REFERENCE_DENSITY} needs an equilibrium that carries a pressure "
                f"(e.g. 'wb_hydro_inc'), got equilibrium = '{equilibrium}'"
            )
            raise ValueError(msg)
        return
    rho_0 = section.get(REFERENCE_DENSITY)
    if rho_0 is None:
        msg = f"'{REFERENCE_DENSITY}' is required for equilibrium = '{equilibrium}'"
        raise ValueError(msg)
    validate_positive(rho_0, REFERENCE_DENSITY)
    if rho_0 >= section["rho_l"]:
        msg = f"{REFERENCE_DENSITY} ({rho_0}) must be below rho_l ({section['rho_l']})"
        raise ValueError(msg)


def multiphase_params(config: SimulationConfig) -> MultiphaseParams | None:
    """The multiphase parameters, or ``None`` for a single-phase run.

    :func:`validate_multiphase` has already guaranteed every key, so no consumer
    guards them again. With a reference density the parameters also carry
    ``mu_b``, the chemical potential at coexistence, which the equilibrium's
    pressure is measured from.
    """
    if config.multiphase is None:
        return None
    params = MultiphaseParams(**config.multiphase)
    if params.equilibrium_reference_density is None:
        return params

    # Imported here: the EOS package imports this module. The potential is solved
    # for here because the step is compiled and cannot run the Maxwell solver.
    from src.operators.macroscopic.eos import coexistence_potential

    return params._replace(equilibrium_reference_potential=coexistence_potential(params))


def phase_references(config: SimulationConfig) -> tuple[float, float] | None:
    """``(rho_lo, rho_hi)`` the run's phases actually sit at, or ``None`` without two phases.

    Measured off the init field for ``init_from_file`` — an equilibrated field
    relaxes away from the prescribed coexistence densities — else the configured
    ``(rho_v, rho_l)``. ``physical_parameters`` measures the same field through
    the same reader, so the buoyancy contrast a run injects and the one its Bond
    number reports cannot diverge.
    """
    params = multiphase_params(config)
    if params is None:
        return None
    measured = measure_init_phase_densities(config) if config.init_type == "init_from_file" else None
    return measured or (float(params.rho_v), float(params.rho_l))
