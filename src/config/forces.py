"""``[*_force]``: one schema per force, held by the ``forces`` field under the section's name.

After :func:`validate_forces`, a force's ``build`` reads its section with
``params[key]`` and never checks it again. Adding a force is its operator file,
a row of :data:`FORCE_SCHEMAS` and, if it has defaults, a row of
:data:`src.config.defaults.FORCES`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from src.config._checks import validate_positive
from src.config.boundary import Edge
from src.config.defaults import FORCES

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig


#: The edges whose solid surfaces carry the electric plate voltages.
ELECTRODE_EDGES: tuple[Edge, Edge] = (Edge.BOTTOM, Edge.TOP)

#: The boundary conditions that make an edge a solid wall (half-way bounce-back).
SOLID_BOUNDARY_CONDITIONS: frozenset[str] = frozenset({"bounce-back", "wetting"})


class ForceSchema(NamedTuple):
    """What a force needs from its section; its defaults are in :data:`src.config.defaults.FORCES`."""

    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    positive: tuple[str, ...] = ()


FORCE_SCHEMAS: dict[str, ForceSchema] = {
    "gravity_force": ForceSchema(required=("force_g",)),
    "gravity_masked_force": ForceSchema(required=("force_g",), optional=("ramp_steps",), positive=("ramp_steps",)),
    "electric_force": ForceSchema(
        required=("permittivity_liquid", "permittivity_vapour", "conductivity_liquid", "conductivity_vapour"),
    ),
}


def apply_force_defaults(forces: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """*forces* with each section's defaults filled in."""
    return {name: {**FORCES.get(name, {}), **params} for name, params in forces.items()}


def validate_forces(config: SimulationConfig) -> None:
    """Check every configured force section against its schema."""
    gravities = sorted(name for name in config.forces if name.startswith("gravity"))
    if len(gravities) > 1:
        msg = f"Only one gravity force can be applied, got {', '.join(gravities)}."
        raise ValueError(msg)

    for name, params in config.forces.items():
        schema = FORCE_SCHEMAS.get(name)
        if schema is None:
            msg = f"Unknown force '{name}'. Expected one of: {', '.join(sorted(FORCE_SCHEMAS))}."
            raise ValueError(msg)
        missing = [key for key in schema.required if key not in params]
        if missing:
            msg = f"[{name}] is missing required key(s): {', '.join(missing)}"
            raise ValueError(msg)
        allowed = {*schema.required, *FORCES.get(name, {}), *schema.optional}
        unknown = sorted(set(params) - allowed)
        if unknown:
            msg = f"[{name}] has unknown key(s): {', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}"
            raise ValueError(msg)
        for key in schema.positive:
            validate_positive(params.get(key), f"[{name}] {key}")
    if "electric_force" in config.forces:
        _validate_electrodes(config)


def _validate_electrodes(config: SimulationConfig) -> None:
    """The electric plates are the solid surfaces of the bottom and top edges: both must be walls."""
    edges = config.boundary_conditions or {}
    not_solid = [
        f"{edge}={edges.get(edge)!r}" for edge in ELECTRODE_EDGES if edges.get(edge) not in SOLID_BOUNDARY_CONDITIONS
    ]
    if not_solid:
        solid = ", ".join(sorted(SOLID_BOUNDARY_CONDITIONS))
        msg = (
            "[electric_force] needs solid bottom and top edges: their surfaces carry voltage_bottom and "
            f"voltage_top. Set each to one of {solid}; got {', '.join(not_solid)}."
        )
        raise ValueError(msg)
