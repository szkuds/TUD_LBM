"""Every configuration default, once.

The section modules read these; no operator or addon package declares a default
a configuration can reach.
"""

from __future__ import annotations
from typing import Any

#: The equilibrium a run gets when it names none, by phase.
DEFAULT_EQUILIBRIUM: dict[str, str] = {"single_phase": "standard_equilibrium", "multiphase": "wb"}

#: The boundary condition of an edge the configuration does not name.
DEFAULT_BOUNDARY_CONDITION: str = "periodic"

#: Each boundary condition's parameters, by BC name; filled into the edge's
#: ``{edge}_{name}`` key of ``[boundary_conditions]``, under what it sets itself.
BOUNDARY_PARAMETERS: dict[str, dict[str, Any]] = {"velocity-inlet": {"u0": 0.05}}

#: The ``[obstacle]`` shape when the section names none.
DEFAULT_OBSTACLE_SHAPE: str = "circle"

#: ``[initialisation] dispersed`` of a ``multiphase_bubbles`` run that names none
#: (the initialiser's own default; ``config.initialisation.dispersed_phase``).
DISPERSED_PHASE: str = "vapour"

#: The four wall parameters of ``[wetting]``; these values leave the wall unmodified.
NEUTRAL_WETTING: dict[str, float] = {
    "phi_left": 1.0,
    "phi_right": 1.0,
    "d_rho_left": 0.0,
    "d_rho_right": 0.0,
}

#: ``[hysteresis]`` optimiser settings. ``max_iterations_above`` defaults to the
#: run's own ``max_iterations``, so it is filled in by the wetting section.
HYSTERESIS: dict[str, Any] = {
    "learning_rate": 0.005,
    "learning_rate_above": 0.05,
    "max_iterations": 50,
    "loss_tol": 1e-4,
    "trial_steps": 2,
    # Chemical-step runs: degrees from its bound past which a line on (or held
    # at) the post surface skips the optimiser and takes its knob's clamp limit.
    "saturation_gap": 1.0,
    # A moving line stays moving while its angle is within ``latch_gap`` degrees of
    # its bound and it has passed its anchor within the last ``stall_steps`` steps.
    # The count has to outlast the release transient: a line held at its bound
    # jitters by ~1e-3 cells over ~10 steps and went up to 49 steps without passing
    # its anchor right after release (R070 replay, 2026-11-11). At 5 every release
    # ended in that transient.
    "latch_gap": 1.0,
    "stall_steps": 200,
}

#: ``[chemical_step]``: distance (lattice units) either side of a contact line at
#: which the surfaces are probed for its advancing and receding bounds.
#: ``chemical_step_edge`` defaults to the measurement wall, so the wetting section fills it in.
CHEMICAL_STEP: dict[str, Any] = {"edge_width": 1.0}

#: Each ``[*_force]`` section's defaults, by force name.
FORCES: dict[str, dict[str, Any]] = {
    "gravity_force": {"inclination_angle_deg": 0.0},
    "gravity_masked_force": {"inclination_angle_deg": 0.0, "ramp_start_t": 0.0},
    "electric_force": {"applied_voltage": 0.0, "voltage_top": 0.0, "voltage_bottom": 0.0},
}
