"""``[wetting]``, ``[hysteresis]``, ``[chemical_step]``: schemas, cross-section rules and the wall.

The wall of a chemical-step run is static for the whole run, so it is built here
and baked into the wetting applicator, not re-derived by the operators.

The wetting modification acts on a band of ghost cells around each contact line.
Near the step that band straddles two surfaces, and the cells on the far side of
the step from the line are *not* the line's to tune: they are fixed at their own
surface's wettability. Pre-step cells take the configured ``[wetting]`` values —
the hydrophobic surface pushing the liquid away — and post-step cells take the
clamp limit toward the post surface — pulling it in. Without that split one
parameter covered both surfaces, the rear line's hydrophilic ``phi`` inflated the
pre-step cells too, and the step was no energy barrier: the line could be pulled
back onto a surface that should repel it.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from src.config.defaults import CHEMICAL_STEP
from src.config.defaults import HYSTERESIS
from src.config.defaults import NEUTRAL_WETTING
from src.config.init_field import load_init_wetting
from src.config.multiphase import multiphase_params

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig

#: Full-scale ``phi - 1`` and ``d_rho`` times the interface width — the clamp
#: limits of the hysteresis optimiser, ``phi <= 1 + 5/W`` and ``d_rho <= 1.5/W``.
PHI_SPAN_W: float = 5.0
D_RHO_SPAN_W: float = 1.5

#: Every key the hysteresis operators read.
_HYSTERESIS_KEYS: frozenset[str] = frozenset({"ca_advancing", "ca_receding", "max_iterations_above", *HYSTERESIS})


class ChemicalStepWall(NamedTuple):
    """The stepped wall, as the wetting applicator needs it.

    Attributes:
        edge: The wall carrying the step (``chemical_step_edge``).
        step_x: Tangential position of the step, in cells.
        pre_phi_left: ``phi`` of pre-step cells in the left line's band.
        pre_phi_right: ``phi`` of pre-step cells in the right line's band.
        pre_d_rho_left: ``d_rho`` of pre-step cells in the left line's band.
        pre_d_rho_right: ``d_rho`` of pre-step cells in the right line's band.
        post_phi: ``phi`` of post-step cells across the step from a line.
        post_d_rho: ``d_rho`` of post-step cells across the step from a line.
    """

    edge: str
    step_x: float
    pre_phi_left: float
    pre_phi_right: float
    pre_d_rho_left: float
    pre_d_rho_right: float
    post_phi: float
    post_d_rho: float


def complete_wetting(section: dict[str, Any] | None, hysteresis: dict[str, Any] | None) -> dict[str, Any] | None:
    """*section*, or a neutral wall for a run that asks for hysteresis without declaring one."""
    return dict(NEUTRAL_WETTING) if section is None and hysteresis is not None else section


def complete_hysteresis(section: dict[str, Any] | None) -> dict[str, Any] | None:
    """*section* with the optimiser defaults filled in."""
    if section is None:
        return None
    completed = {**HYSTERESIS, **section}
    completed.setdefault("max_iterations_above", completed["max_iterations"])
    return completed


def complete_chemical_step(section: dict[str, Any] | None, edges: dict[str, Any]) -> dict[str, Any] | None:
    """*section* with its defaults, the step on the measurement wall unless it names an edge."""
    if section is None:
        return None
    from src.operators.wetting._edge_config import first_wetting_edge

    completed = {**CHEMICAL_STEP, **section}
    completed.setdefault("chemical_step_edge", first_wetting_edge(edges) or "bottom")
    return completed


def validate_wetting(config: SimulationConfig) -> None:
    """Reject ``[wetting]`` keys no wall parameter reads: the section is free-form TOML."""
    if config.wetting is None:
        return
    unknown = sorted(set(config.wetting) - set(NEUTRAL_WETTING))
    if unknown:
        msg = f"Unknown [wetting] keys {unknown}; allowed: {sorted(NEUTRAL_WETTING)}"
        raise ValueError(msg)


def validate_hysteresis(config: SimulationConfig) -> None:
    """Reject ``[hysteresis]`` keys the optimiser does not read.

    The section is free-form TOML, so a misspelled key would otherwise be
    ignored without a word.
    """
    if config.hysteresis is None:
        return
    unknown = set(config.hysteresis) - _HYSTERESIS_KEYS
    if unknown:
        msg = f"Unknown [hysteresis] keys {sorted(unknown)}; allowed: {sorted(_HYSTERESIS_KEYS)}"
        raise ValueError(msg)


def validate_chemical_step(config: SimulationConfig) -> None:
    """Reject a chemical step that is not on the wall the contact angles are measured at.

    The wetting applicator splits a wall by surface only on ``chemical_step_edge``,
    while the hysteresis reads its contact lines off the first ``"wetting"`` edge.
    A step configured on any other edge would be ignored without a word.
    """
    if config.chemical_step is None:
        return
    from src.operators.wetting._edge_config import first_wetting_edge

    wall = first_wetting_edge(config.boundary_conditions)
    edge = config.chemical_step["chemical_step_edge"]
    if wall is not None and edge != wall:
        msg = f"chemical_step_edge must be the wetting wall '{wall}', got '{edge}'"
        raise ValueError(msg)


def has_wetting_wall(config: SimulationConfig) -> bool:
    """A multiphase run with a ``[wetting]`` section (every hysteresis run has one)."""
    return config.multiphase is not None and config.wetting is not None


def wetting_scalars(config: SimulationConfig) -> dict[str, float] | None:
    """The four wall parameters by name, a null one taking its default; ``None`` without wetting."""
    if config.wetting is None:
        return None
    section = config.wetting
    return {
        name: float(value) if (value := section.get(name)) is not None else default
        for name, default in NEUTRAL_WETTING.items()
    }


def chemical_step_wall(config: SimulationConfig) -> ChemicalStepWall | None:
    """The stepped wall, or ``None`` without a chemical step or wetting.

    The post surface is pulled toward its own wettability: to the ``phi`` clamp
    limit when it wets more than the pre surface (a lower advancing angle), else
    to the ``d_rho`` limit.
    """
    step = config.chemical_step
    wall = wetting_scalars(config)
    params = multiphase_params(config)
    if step is None or wall is None or params is None:
        return None
    width = float(params.interface_width)
    post_wets_more = float(step["ca_advancing_post_step"]) < float(step["ca_advancing_pre_step"])
    return ChemicalStepWall(
        edge=str(step["chemical_step_edge"]),
        step_x=float(step["chemical_step_location"]) * config.grid_shape[0],
        pre_phi_left=wall["phi_left"],
        pre_phi_right=wall["phi_right"],
        pre_d_rho_left=wall["d_rho_left"],
        pre_d_rho_right=wall["d_rho_right"],
        post_phi=1.0 + PHI_SPAN_W / width if post_wets_more else 1.0,
        post_d_rho=0.0 if post_wets_more else D_RHO_SPAN_W / width,
    )


def restored_wetting(config: SimulationConfig) -> dict[str, float] | None:
    """Wetting state a hysteresis restart resumes from, or ``None`` for a fresh start.

    The hysteresis optimiser accumulates ``phi``/``d_rho`` and moves the
    contact-line anchors over a run; restarting ``init_from_file`` from one of
    its snapshots must continue from those values. Seeding them from ``[wetting]``
    instead dropped a wall at ``phi`` ≈ 1.7 back to 1.0 on restart, and the contact
    angle ran up past its advancing bound. Only for hysteresis runs: a
    fixed-wetting run's parameters are its configuration.
    """
    if config.hysteresis is None or config.init_type != "init_from_file":
        return None
    return load_init_wetting(config)
