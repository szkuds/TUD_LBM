"""Multiphase LBM step with hysteresis and chemical-step wetting.

Registered as ``update_timestep:multiphase_hysteresis_chemical_step``.

Identical to step_multiphase_hysteresis except the wetting-state update
uses update_wetting_state_chemical_step, which selects (ca_advancing, ca_receding)
per contact-line side based on position relative to the chemical step.
"""

from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
from tud_lbm.operators.force import compute_total_force_ext
from tud_lbm.operators.step._common import _multiphase_pipeline
from tud_lbm.operators.step._multiphase_hysteresis import _make_wetting_ops
from tud_lbm.operators.step._multiphase_hysteresis import _trial_step
from tud_lbm.operators.wetting.hysteresis import update_wetting_state_chemical_step
from tud_lbm.pipeline.state import update_extra_state
from tud_lbm.registry import update_timestep_operator

if TYPE_CHECKING:
    from tud_lbm.pipeline.setup import SimulationSetup
    from tud_lbm.pipeline.state.state import State


@update_timestep_operator(name="multiphase_hysteresis_chemical_step")
def step_multiphase_hysteresis_chemical_step(setup: SimulationSetup, state: State) -> State:
    """Hysteresis-enabled multiphase step with chemical-step wetting.

    Used when wetting_config, hysteresis_config, and chemical_step_config are all present.
    Per-side CA targets are determined at each timestep by comparing each contact-line
    location against the chemical step position.

    Args:
        setup: SimulationSetup carrying gradient_density_wetting,
               laplacian_density_wetting, and chemical_step_config.
        state: Current State. state.wetting is a WettingState.

    Returns:
        Updated State with optimised wetting parameters.
    """
    # 1. Compute external forces
    force_ext, state = compute_total_force_ext(setup, state, setup.forces)

    # 2. Build operators from live wetting parameters
    grad, lap = _make_wetting_ops(setup, state.wetting)

    # 3. Run multiphase physics kernel
    f_out, rho, u, force_tot = _multiphase_pipeline(setup, state.f, force_ext, grad, lap)

    # 4. Optimise wetting parameters via chemical-step hysteresis
    new_wetting = update_wetting_state_chemical_step(
        state.wetting,
        rho,
        setup,
        trial_step_fn=partial(_trial_step, setup, f_out, force_ext),
    )

    # 5. Assemble new state
    new_state = state._replace(
        f=f_out,
        rho=rho,
        u=u,
        force=force_tot,
        wetting=new_wetting,
        t=state.t + 1,
    )

    # 6. Update extra state (plugins)
    return update_extra_state(setup, state, new_state, force_ext=force_ext)
