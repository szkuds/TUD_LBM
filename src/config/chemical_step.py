"""The wall of a chemical-step run: where the step is, and what each surface is fixed at.

Resolved only by :attr:`SimulationConfig.chemical_step_wall
<src.config.simulation_config.SimulationConfig.chemical_step_wall>`. Static for the
whole run, so it is built by the configuration and baked into the wetting
applicator, not re-derived by the operators.

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
from typing import NamedTuple

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig

#: Full-scale ``phi - 1`` and ``d_rho`` times the interface width — the clamp
#: limits of the hysteresis optimiser, ``phi <= 1 + 5/W`` and ``d_rho <= 1.5/W``.
PHI_SPAN_W: float = 5.0
D_RHO_SPAN_W: float = 1.5


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


def build_chemical_step_wall(config: SimulationConfig) -> ChemicalStepWall | None:
    """The stepped wall of *config*, or ``None`` without a chemical step or wetting.

    The post surface is pulled toward its own wettability: to the ``phi`` clamp
    limit when it wets more than the pre surface (a lower advancing angle), else
    to the ``d_rho`` limit.
    """
    csc = config.chemical_step_config
    wetting = config.wetting_defaults
    if csc is None or wetting is None or config.interface_width is None:
        return None
    w = float(config.interface_width)
    post_wets_more = float(csc["ca_advancing_post_step"]) < float(csc["ca_advancing_pre_step"])
    return ChemicalStepWall(
        edge=str(csc["chemical_step_edge"]),
        step_x=float(csc["chemical_step_location"]) * config.grid_shape[0],
        pre_phi_left=wetting["phi_left"],
        pre_phi_right=wetting["phi_right"],
        pre_d_rho_left=wetting["d_rho_left"],
        pre_d_rho_right=wetting["d_rho_right"],
        post_phi=1.0 + PHI_SPAN_W / w if post_wets_more else 1.0,
        post_d_rho=0.0 if post_wets_more else D_RHO_SPAN_W / w,
    )
