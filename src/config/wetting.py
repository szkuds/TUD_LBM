"""``[wetting]``, ``[hysteresis]``, ``[chemical_step]``: schemas, cross-section rules and the wall.

The wall of a chemical-step run is static for the whole run, so it is built here
and baked into the wetting applicator, not re-derived by the operators.

The wetting modification acts on a band of ghost cells around each contact line.
Near the step that band encounters two surfaces, and the cells on the far side of
the step from the line are *not* the line's to tune: they are fixed at their own
surface's wettability. Pre-step cells take the configured ``[wetting]`` values —
the hydrophobic surface pushing the liquid away — and post-step cells take the
clamp limit toward the post surface — pulling it in. Without that split one
parameter covered both surfaces, the rear line's hydrophilic ``phi`` inflated the
pre-step cells too, and the step was no energy barrier: the line could be pulled
back onto a surface that should repel it.

"Pre" and "post" are the inclusion's *start* and *destination* surfaces, for either
direction of travel: ``chemical_step_pre_side`` names the tangential side the
inclusion starts on, derived from the ``[initialisation]`` centres when the section
does not set it. Every "is this cell past the step" test goes through
:meth:`ChemicalStepWall.on_post`, so a bubble rising toward ``-x`` gets the same
step rules as a droplet sliding toward ``+x``.
"""

from __future__ import annotations
import math
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import overload
from src.config.boundary import Edge
from src.config.defaults import CHEMICAL_STEP
from src.config.defaults import HYSTERESIS
from src.config.defaults import NEUTRAL_WETTING
from src.config.init_field import load_init_wetting
from src.config.initialisation import DispersedPhase
from src.config.initialisation import dispersed_phase
from src.config.multiphase import multiphase_params
from src.config.multiphase import phase_references

if TYPE_CHECKING:
    import jax
    import numpy as np
    from src.config.simulation_config import SimulationConfig

#: Full-scale ``phi - 1`` and ``d_rho`` times the interface width — the clamp
#: limits of the hysteresis optimiser, ``phi <= 1 + 5/W`` and ``d_rho <= 1.5/W``.
PHI_SPAN_W: float = 5.0
D_RHO_SPAN_W: float = 1.5


class ContactLine(StrEnum):
    """One of the two contact lines on a wetting wall, in ascending tangential order.

    ``LEFT`` has the lower tangential coordinate: on a bottom or top wall the
    smaller ``x``, on a left or right wall the smaller ``y``. The values are the
    suffixes of the saved per-line keys (``ca_left``, ``phi_right``, ``cll_left``),
    so they must not change.
    """

    LEFT = "left"
    RIGHT = "right"

    @property
    def other(self) -> ContactLine:
        """The contact line at the other end of the inclusion."""
        return ContactLine.RIGHT if self == ContactLine.LEFT else ContactLine.LEFT


class TangentialSide(StrEnum):
    """The side of a chemical step an inclusion starts on: ``chemical_step_pre_side``.

    ``LOW`` is the lower tangential coordinate (smaller ``x``), so a ``LOW`` start
    travels toward ``+x``. The values are written to ``config.toml`` and must not
    change.
    """

    LOW = "low"
    HIGH = "high"


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
        pre_side: The side of the step the inclusion starts on.
    """

    edge: Edge
    step_x: float
    pre_phi_left: float
    pre_phi_right: float
    pre_d_rho_left: float
    pre_d_rho_right: float
    post_phi: float
    post_d_rho: float
    pre_side: TangentialSide

    @property
    def travel(self) -> float:
        """``+1`` when the destination surface is at higher ``x``, else ``-1``."""
        return 1.0 if self.pre_side == TangentialSide.LOW else -1.0

    @property
    def hold_x(self) -> float:
        """The first position on the post surface: the threshold and a held line's pin.

        A ``LOW`` start uses ``step_x`` itself. A ``HIGH`` start uses its mirror
        about the cell face between the two surfaces (``ceil(step_x) - 0.5``), so a
        run and its x-mirror classify every position alike, not just whole cells.
        """
        if self.pre_side == TangentialSide.LOW:
            return self.step_x
        return 2.0 * (math.ceil(self.step_x) - 0.5) - self.step_x

    @overload
    def on_post(self, x: np.ndarray) -> np.ndarray: ...
    @overload
    def on_post(self, x: jax.Array) -> jax.Array: ...
    def on_post(self, x: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
        """Whether each tangential position *x* lies on the destination (post) surface."""
        return x >= self.hold_x if self.pre_side == TangentialSide.LOW else x <= self.hold_x


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


def start_side(location: float, centres: object, nx: int) -> TangentialSide | None:
    """The side of the step every initial centre lies on, or ``None`` if that is not one side.

    Centres are fractions of ``nx - 1`` (the initialiser's cell rule) and the step
    sits at ``location * nx`` (``chemical_step_wall``). No centres, a centre on the
    step, or centres on both sides leave the side to the configuration.
    """
    if not isinstance(centres, list | tuple) or not centres:
        return None
    xs = [centre[0] for centre in centres if isinstance(centre, list | tuple) and centre]
    if len(xs) != len(centres) or not all(isinstance(x, int | float) for x in xs):
        return None
    step_x = float(location) * nx
    cells = [round(x * (nx - 1)) for x in xs if isinstance(x, int | float)]
    if all(cell < step_x for cell in cells):
        return TangentialSide.LOW
    if all(cell > step_x for cell in cells):
        return TangentialSide.HIGH
    return None


def complete_chemical_step(
    section: dict[str, Any] | None,
    edges: dict[str, Any],
    initialisation: dict[str, Any],
    nx: int,
) -> dict[str, Any] | None:
    """*section* with its defaults: the step on the measurement wall, and the inclusion's start side.

    ``chemical_step_edge`` defaults to the first ``"wetting"`` edge, and
    ``chemical_step_pre_side`` to the side of the step the ``[initialisation]``
    centres lie on. Neither overrides a configured value.
    """
    if section is None:
        return None
    from src.operators.wetting._edge_config import first_wetting_edge

    completed = {**CHEMICAL_STEP, **section}
    completed.setdefault("chemical_step_edge", (first_wetting_edge(edges) or Edge.BOTTOM).value)
    if "chemical_step_pre_side" not in completed and "chemical_step_location" in completed:
        side = start_side(completed["chemical_step_location"], initialisation.get("centres"), nx)
        if side is not None:
            completed["chemical_step_pre_side"] = side.value
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
    side = config.chemical_step.get("chemical_step_pre_side")
    if side is None:
        msg = (
            "[chemical_step] chemical_step_pre_side ('low' or 'high': the side of the step the inclusion "
            "starts on) is required when the [initialisation] centres do not all lie on one side of it"
        )
        raise ValueError(msg)
    if side not in set(TangentialSide):
        msg = f"chemical_step_pre_side must be one of {[s.value for s in TangentialSide]}, got {side!r}"
        raise ValueError(msg)
    _validate_dispersed_phase(config)


def _validate_dispersed_phase(config: SimulationConfig) -> None:
    """A chemical step needs the dispersed phase: it picks the destination surface's knob."""
    value = config.initialisation.get("dispersed")
    if value is not None and value not in set(DispersedPhase):
        msg = f"[initialisation] dispersed must be one of {[p.value for p in DispersedPhase]}, got {value!r}"
        raise ValueError(msg)
    if dispersed_phase(config) is None:
        msg = (
            f"[initialisation] dispersed ('vapour' or 'liquid') is required for a chemical-step run "
            f"with init_type '{config.init_type}'"
        )
        raise ValueError(msg)


def has_wetting_wall(config: SimulationConfig) -> bool:
    """A multiphase run with a ``[wetting]`` section (every hysteresis run has one)."""
    return config.multiphase is not None and config.wetting is not None


def wall_rho_mean(config: SimulationConfig) -> float | None:
    """The density the wetting wall separates the phases at, or ``None`` without two phases.

    The midpoint of :func:`~src.config.multiphase.phase_references`: on an
    ``init_from_file`` run, the densities the init field actually sits at, not
    the prescribed ``rho_l``/``rho_v``. The applicator's anchors and contrast
    gate, the measured angles and contact lines, the hysteresis optimiser and the
    contact-angle overlay all read this one value.
    """
    references = phase_references(config)
    return None if references is None else 0.5 * (references[0] + references[1])


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

    The post surface is pulled toward its own wettability. The windows are in the
    dispersed frame, so a lower post advancing angle means the post surface wets
    the *dispersed* phase more. ``phi`` and ``d_rho`` are liquid-frame knobs: that
    is ``phi`` (more liquid-wetting) for a droplet and ``d_rho`` (less) for a
    bubble, each at its clamp limit.
    """
    step = config.chemical_step
    wall = wetting_scalars(config)
    params = multiphase_params(config)
    phase = dispersed_phase(config)
    if step is None or wall is None or params is None:
        return None
    assert phase is not None  # noqa: S101 - validate_chemical_step guarantees it
    width = float(params.interface_width)
    post_wets_dispersed_more = float(step["ca_advancing_post_step"]) < float(step["ca_advancing_pre_step"])
    post_wets_more = post_wets_dispersed_more == (phase == DispersedPhase.LIQUID)
    return ChemicalStepWall(
        edge=Edge(step["chemical_step_edge"]),
        step_x=float(step["chemical_step_location"]) * config.grid_shape[0],
        pre_phi_left=wall["phi_left"],
        pre_phi_right=wall["phi_right"],
        pre_d_rho_left=wall["d_rho_left"],
        pre_d_rho_right=wall["d_rho_right"],
        post_phi=1.0 + PHI_SPAN_W / width if post_wets_more else 1.0,
        post_d_rho=0.0 if post_wets_more else D_RHO_SPAN_W / width,
        pre_side=TangentialSide(step["chemical_step_pre_side"]),
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
