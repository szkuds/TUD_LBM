"""``[wetting]``, ``[hysteresis]``, ``[chemical_step]``: schemas, cross-section rules and the wall.

A chemical step splits the wall into two regions. Each contact line's wetting band can
cross the step, and the cells across it keep their own region's wettability: pre-step
cells the ``[wetting]`` values, post-step cells the clamp limit toward the post-step
angles. With one parameter per band the step would be no energy barrier.

"Pre" and "post" are the inclusion's start and destination regions, for either direction
of travel (``chemical_step_pre_side``). Every "past the step" test goes through
:meth:`ChemicalStepRegions.on_post`.
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
from src.config.init_field import measure_init_interface_width
from src.config.initialisation import DispersedPhase
from src.config.initialisation import dispersed_phase
from src.config.initialisation import validate_dispersed_phase
from src.config.multiphase import multiphase_params
from src.config.multiphase import phase_references

if TYPE_CHECKING:
    import jax
    import numpy as np
    from src.config.simulation_config import SimulationConfig

#: Full-scale ``phi - 1`` times the interface width, and full-scale ``d_rho`` times the interface
#: width over the density contrast: the clamp limits of the hysteresis optimiser,
#: ``phi <= 1 + 5/W`` and ``d_rho <= 1.5 * (rho_l - rho_v)/W`` (:func:`wetting_knob_limits`).
PHI_SPAN_W: float = 5.0
D_RHO_SPAN_W: float = 1.5

#: Every key the hysteresis operators read.
_HYSTERESIS_KEYS: frozenset[str] = frozenset({"ca_advancing", "ca_receding", "max_iterations_above", *HYSTERESIS})


class ContactLine(StrEnum):
    """One of the two contact lines on a wetting edge, in ascending tangential order.

    ``LEFT`` has the lower tangential coordinate on every wall. The values are the
    suffixes of the saved per-line keys (``ca_left``, ``phi_right``), so they must
    not change.
    """

    LEFT = "left"
    RIGHT = "right"

    @property
    def other(self) -> ContactLine:
        """The contact line at the other end of the inclusion."""
        return ContactLine.RIGHT if self == ContactLine.LEFT else ContactLine.LEFT


class KnobLimits(NamedTuple):
    """The upper clamp limits of the two wetting knobs (both lower limits are neutral).

    Attributes:
        phi_max: Largest ``phi``, ``1 + PHI_SPAN_W / interface_width``.
        d_rho_max: Largest ``d_rho``, ``D_RHO_SPAN_W * (rho_l - rho_v) / W``.
    """

    phi_max: float
    d_rho_max: float


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


def validate_wetting(config: SimulationConfig) -> None:
    """Reject ``[wetting]`` keys no wall parameter reads: the section is free-form TOML."""
    if config.wetting is None:
        return
    unknown = sorted(set(config.wetting) - set(NEUTRAL_WETTING))
    if unknown:
        msg = f"Unknown [wetting] keys {unknown}; allowed: {sorted(NEUTRAL_WETTING)}"
        raise ValueError(msg)


def validate_hysteresis(config: SimulationConfig) -> None:
    """Reject ``[hysteresis]`` keys the optimiser does not read, and bad latch settings."""
    if config.hysteresis is None:
        return
    unknown = set(config.hysteresis) - _HYSTERESIS_KEYS
    if unknown:
        msg = f"Unknown [hysteresis] keys {sorted(unknown)}; allowed: {sorted(_HYSTERESIS_KEYS)}"
        raise ValueError(msg)
    gap = config.hysteresis["latch_gap"]
    if isinstance(gap, bool) or not isinstance(gap, int | float) or gap < 0:
        msg = f"[hysteresis] latch_gap must be a number of degrees >= 0, got {gap!r}"
        raise ValueError(msg)
    steps = config.hysteresis["stall_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        msg = f"[hysteresis] stall_steps must be an integer >= 1, got {steps!r}"
        raise ValueError(msg)


def has_wetting_edge(config: SimulationConfig) -> bool:
    """A multiphase run with a ``[wetting]`` section (every hysteresis run has one)."""
    return config.multiphase is not None and config.wetting is not None


def wetting_rho_mean(config: SimulationConfig) -> float | None:
    """The density the wetting edge separates the phases at, or ``None`` without two phases.

    The midpoint of :func:`~src.config.multiphase.phase_references`, so on an
    ``init_from_file`` run the init field's own densities. Every wetting consumer
    (applicator, measurement, optimiser, overlay) reads this one value.
    """
    references = phase_references(config)
    return None if references is None else 0.5 * (references[0] + references[1])


def wetting_interface_width(config: SimulationConfig) -> float | None:
    """The interface width the wetting bands are bounded by, in cells, or ``None``.

    Measured off the init field of an ``init_from_file`` run. ``None`` for a run that
    starts from an initialiser: it has no equilibrated interface to measure.
    """
    if config.multiphase is None or config.init_type != "init_from_file":
        return None
    return measure_init_interface_width(config)


def wetting_scalars(config: SimulationConfig) -> dict[str, float] | None:
    """The four wall parameters by name, a null one taking its default; ``None`` without wetting."""
    if config.wetting is None:
        return None
    section = config.wetting
    return {
        name: float(value) if (value := section.get(name)) is not None else default
        for name, default in NEUTRAL_WETTING.items()
    }


def _d_rho_scale(config: SimulationConfig) -> float | None:
    """``(rho_l - rho_v)/W``, the density scale of ``d_rho``, or ``None`` without two phases.

    The contrast is the run's phase references, ``W`` the measured interface width on
    a restart, else ``interface_width`` (only an initialisation hint for Carnahan-Starling).
    """
    params = multiphase_params(config)
    references = phase_references(config)
    if params is None or references is None:
        return None
    width = wetting_interface_width(config) or float(params.interface_width)
    return (references[1] - references[0]) / width


def wetting_knob_limits(config: SimulationConfig) -> KnobLimits | None:
    """The clamp limits of ``phi`` and ``d_rho``, or ``None`` without two phases.

    ``phi`` multiplies the ghost density, so its limit is dimensionless, ``1 + 5/W``.
    ``d_rho`` is subtracted from it, so it is a density and its limit scales with the
    contrast, ``1.5 * (rho_l - rho_v)/W`` (:func:`_d_rho_scale`).
    """
    params = multiphase_params(config)
    scale = _d_rho_scale(config)
    if params is None or scale is None:
        return None
    return KnobLimits(phi_max=1.0 + PHI_SPAN_W / float(params.interface_width), d_rho_max=D_RHO_SPAN_W * scale)


def restored_wetting(config: SimulationConfig) -> dict[str, float] | None:
    """Wetting state a hysteresis restart resumes from, or ``None`` for a fresh start.

    A hysteresis restart continues from the snapshot's ``phi``/``d_rho`` and anchors,
    not from ``[wetting]``. A fixed-wetting run's parameters are its configuration.
    """
    if config.hysteresis is None or config.init_type != "init_from_file":
        return None
    return load_init_wetting(config)


#: Full-scale ``d_rho`` of a bubble's fixed post-step cells, in units of ``(rho_l - rho_v)/W``:
#: a third of the optimiser's limit (see :func:`chemical_step_regions`).
POST_D_RHO_SPAN_W: float = 0.5
#: The keys every ``[chemical_step]`` needs: the location and the contact angles of both regions.
CHEMICAL_STEP_REQUIRED_KEYS: tuple[str, ...] = (
    "chemical_step_location",
    "ca_advancing_pre_step",
    "ca_receding_pre_step",
    "ca_advancing_post_step",
    "ca_receding_post_step",
)


class TangentialSide(StrEnum):
    """The side of a chemical step an inclusion starts on: ``chemical_step_pre_side``.

    ``LOW`` is the lower tangential coordinate, so a ``LOW`` start travels toward
    ``+x``. The values are written to ``config.toml`` and must not change.
    """

    LOW = "low"
    HIGH = "high"


class ChemicalStepRegions(NamedTuple):
    """The stepped edge, as the wetting applicator needs it.

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
        """``+1`` when the post-step region is at higher ``x``, else ``-1``."""
        return 1.0 if self.pre_side == TangentialSide.LOW else -1.0

    @property
    def hold_x(self) -> float:
        """The first position on the post-step region: the threshold and a held line's pin.

        A ``HIGH`` start mirrors ``step_x`` about the cell face between the regions,
        so a run and its x-mirror classify every position alike.
        """
        if self.pre_side == TangentialSide.LOW:
            return self.step_x
        return 2.0 * (math.ceil(self.step_x) - 0.5) - self.step_x

    @overload
    def on_post(self, x: np.ndarray) -> np.ndarray: ...
    @overload
    def on_post(self, x: jax.Array) -> jax.Array: ...
    def on_post(self, x: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
        """Whether each tangential position *x* lies on the post-step (destination) region."""
        return x >= self.hold_x if self.pre_side == TangentialSide.LOW else x <= self.hold_x


def start_side(location: float, centres: object, nx: int) -> TangentialSide | None:
    """The side of the step every initial centre lies on, or ``None`` if that is not one side.

    Centres are fractions of ``nx - 1`` (the initialiser's cell rule) and the step
    sits at ``location * nx``. No centres, a centre on the step, or centres on both
    sides leave the side to the configuration.
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
    """*section* with its defaults: the step on the measurement edge, and the inclusion's start side.

    ``chemical_step_edge`` defaults to the first ``"wetting"`` edge, and
    ``chemical_step_pre_side`` to the side the ``[initialisation]`` centres lie on.
    Neither overrides a configured value.
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


def validate_chemical_step(config: SimulationConfig) -> None:
    """Reject an incomplete chemical step, or one off the edge the contact angles are measured at.

    The applicator splits only ``chemical_step_edge``, while the hysteresis reads its
    contact lines off the first ``"wetting"`` edge, so a step anywhere else would be
    ignored without a word.
    """
    if config.chemical_step is None:
        return
    from src.operators.wetting._edge_config import first_wetting_edge

    missing = [key for key in CHEMICAL_STEP_REQUIRED_KEYS if key not in config.chemical_step]
    if missing:
        msg = (
            "[chemical_step] needs the chemical step location and the contact angles "
            f"of both regions; missing {missing}"
        )
        raise ValueError(msg)
    if config.wetting is None:
        # complete_wetting fills a neutral [wetting] for a [hysteresis] run, so
        # only a step with neither section reaches this: no region would be split.
        msg = (
            "[chemical_step] needs a wetting boundary condition: add a [wetting] section "
            "(fixed contact angle) or a [hysteresis] section (contact angle hysteresis)"
        )
        raise ValueError(msg)
    wall = first_wetting_edge(config.boundary_conditions)
    edge = config.chemical_step["chemical_step_edge"]
    if wall is not None and edge != wall:
        msg = f"chemical_step_edge must be the wetting edge '{wall}', got '{edge}'"
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
    # The dispersed phase picks the post-step region's knob.
    validate_dispersed_phase(config, "a chemical-step run")


def chemical_step_regions(config: SimulationConfig) -> ChemicalStepRegions | None:
    """The stepped edge, or ``None`` without a chemical step or wetting.

    The windows are dispersed-frame and the knobs liquid-frame, so "the post-step
    region wets the dispersed phase more" is ``phi`` at its clamp limit for a droplet
    and ``d_rho`` for a bubble. A bubble's post-step cells take a third of the
    ``d_rho`` limit, not the limit: they sit in the leading line's band on liquid, and
    the limit drove that liquid to the spinodal before the line reached the step.
    """
    step = config.chemical_step
    wall = wetting_scalars(config)
    limits = wetting_knob_limits(config)
    phase = dispersed_phase(config)
    if step is None or wall is None or limits is None:
        return None
    assert phase is not None  # noqa: S101 - validate_chemical_step guarantees it
    post_wets_dispersed_more = float(step["ca_advancing_post_step"]) < float(step["ca_advancing_pre_step"])
    post_wets_more = post_wets_dispersed_more == (phase == DispersedPhase.LIQUID)
    scale = _d_rho_scale(config)
    # wetting_knob_limits returned limits, so the run has two phases.
    assert scale is not None  # noqa: S101
    return ChemicalStepRegions(
        edge=Edge(step["chemical_step_edge"]),
        step_x=float(step["chemical_step_location"]) * config.grid_shape[0],
        pre_phi_left=wall["phi_left"],
        pre_phi_right=wall["phi_right"],
        pre_d_rho_left=wall["d_rho_left"],
        pre_d_rho_right=wall["d_rho_right"],
        post_phi=limits.phi_max if post_wets_more else 1.0,
        post_d_rho=0.0 if post_wets_more else POST_D_RHO_SPAN_W * scale,
        pre_side=TangentialSide(step["chemical_step_pre_side"]),
    )
