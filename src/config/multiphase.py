"""``[multiphase]``: the section's presence is what makes a run multiphase.

Holds the section's schema, the per-EOS parameter table, the JIT-static
:class:`MultiphaseParams` and the densities the run's phases actually sit at.
Adding an EOS with its own parameters is one row of :data:`EOS_PARAMETERS`.

``rho_l``/``rho_v`` may be omitted for an EOS that registers a coexistence
solver (kind ``"coexistence"``, the Maxwell construction):
:func:`complete_multiphase` fills them from the EOS's own parameters. A
configured density always wins. To sweep ``t_eos``, omit them, since a
configured density does not follow the temperature.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from src.config._checks import registered_names
from src.config._checks import validate_positive
from src.config.init_field import measure_init_phase_densities

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig

#: Every EOS needs these. ``rho_l``/``rho_v`` are filled by :func:`complete_multiphase`
#: when omitted for an EOS with a registered coexistence solver.
REQUIRED: tuple[str, ...] = ("kappa", "rho_l", "rho_v", "interface_width", "eos")
#: Keys any EOS may carry.
OPTIONAL: tuple[str, ...] = ("g",)
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


def is_multiphase(config: SimulationConfig) -> bool:
    """A run is multiphase when it has a ``[multiphase]`` section."""
    return config.multiphase is not None


def complete_multiphase(section: dict[str, Any] | None) -> dict[str, Any] | None:
    """*section* without its null keys, and with omitted densities filled from the Maxwell construction.

    A null means the key is not set, and TOML cannot write one. When ``rho_l``
    or ``rho_v`` is missing and the EOS registers a coexistence solver with
    every parameter of its :data:`EOS_PARAMETERS` row present, the missing
    density is the Maxwell value. A configured density is never overwritten;
    anything else is left to :func:`validate_multiphase`.

    Raises:
        ValueError: If the EOS parameters admit no liquid-vapour coexistence.
    """
    if section is None:
        return None
    completed = {name: value for name, value in section.items() if value is not None}
    eos = completed.get("eos")
    if not isinstance(eos, str) or {"rho_l", "rho_v"} <= completed.keys():
        return completed
    row = EOS_PARAMETERS.get(eos)
    if row is None or not set(row) <= completed.keys():
        return completed

    # Imported here: the EOS package imports this module.
    from src.operators.macroscopic.eos import coexistence_densities

    pair = coexistence_densities(eos, completed)
    if pair is not None:
        rho_v, rho_l = pair
        # Six significant figures: the saved config.toml stays readable, and the
        # surface-tension cache key (which holds rho_l/rho_v) cannot differ in the
        # last bits between machines.
        completed.setdefault("rho_v", float(f"{rho_v:.6g}"))
        completed.setdefault("rho_l", float(f"{rho_l:.6g}"))
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
    unknown = sorted(set(section) - {*REQUIRED, *OPTIONAL, *eos_parameters})
    if unknown:
        msg = f"[multiphase] has unknown key(s) for eos = '{eos}': {', '.join(unknown)}"
        raise ValueError(msg)


def multiphase_params(config: SimulationConfig) -> MultiphaseParams | None:
    """The multiphase parameters, or ``None`` for a single-phase run.

    :func:`validate_multiphase` has already guaranteed every key, so no consumer
    guards them again.
    """
    return None if config.multiphase is None else MultiphaseParams(**config.multiphase)


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
