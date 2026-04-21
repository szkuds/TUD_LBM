"""Multiphase LBM step operator.

Registered as ``update_timestep:multiphase`` via the operator registry.
Supports both wetting and non-wetting multiphase simulations.
"""

from __future__ import annotations
from operators.force import compute_total_force_ext
from operators.step._common import _apply_common_step
from operators.step._wetting_differential_operators import _make_wetting_differential_ops
from registry import update_timestep_operator
from setup import SimulationSetup
from state.state import State


@update_timestep_operator(name="multiphase")
def step_multiphase(setup: SimulationSetup, state: State) -> State:
    """Multiphase LBM step for both wetting and non-wetting simulations.

    When wetting is not active (state.wetting is None), uses the prebuilt
    gradient_density and laplacian_density closures from setup.

    When wetting is active (state.wetting is not None), builds wetting shims
    that inject live wetting parameters into the density operators at each step.

    Args:
        setup: Closed-over :class:`~setup.simulation_setup.SimulationSetup`.
        state: Current :class:`~state.state.State`.

    Returns:
        Updated :class:`~state.state.State` after one time step.
    """
    # 1. External forces
    force_ext, state = compute_total_force_ext(setup, state, setup.forces, setup.streaming_fn)

    # 1.1. Resolve density operators (wetting shims if applicable)
    # Wetting shims should only be applied if BOTH the state has wetting AND setup was built with wetting config
    if state.wetting is not None and setup.config.wetting_config is not None:
        gradient_density, laplacian_density = _make_wetting_differential_ops(setup, state.wetting)
    else:
        gradient_density = setup.gradient_density
        laplacian_density = setup.laplacian_density

    # 2. Multiphase macroscopic (chemical potential force, wetting-corrected operators)
    rho, u, force_tot = setup.macroscopic_fn(
        state.f,
        setup.lattice,
        setup.multiphase_params,
        force_ext,
        gradient_standard=setup.gradient_standard,
        laplacian_density=laplacian_density,
    )

    # 3–6. Shared pipeline (equilibrium → collision (+source with grad_density) → streaming → BCs)
    new_state = _apply_common_step(setup, state, rho, u, force_tot, gradient_density=gradient_density)

    # 7. Hysteresis update (if applicable)
    new_wetting = new_state.wetting
    if new_state.wetting is not None and setup.wetting_fn is not None:
        new_wetting = setup.wetting_fn(
            state.wetting,
            rho,
            setup,
            state.f,
            force_ext=force_ext,
        )

    return new_state._replace(
        force=force_tot,
        wetting=new_wetting,
    )
