"""Extra-state plugins for force modules."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
import jax.numpy as jnp
from tud_lbm.operators.force._electric import ElectricParams
from tud_lbm.operators.force._electric import _equilibrium_h
from tud_lbm.operators.force._electric import _rho_to_phi
from tud_lbm.registry import extra_state_plugin

if TYPE_CHECKING:
    from tud_lbm.config import SimulationConfig
    from tud_lbm.pipeline.setup import SimulationSetup
    from tud_lbm.pipeline.state import State


def _get_electric_params(setup: SimulationSetup) -> ElectricParams | None:
    if setup.forces is None:
        return None
    for spec in setup.forces.specs:
        if spec.name == "electric_force":
            return spec.precomputed
    return None


@extra_state_plugin(name="electric")
class ElectricExtraStatePlugin:
    """Initialises and updates electric potential distributions."""

    @staticmethod
    def is_active(config: SimulationConfig) -> bool:
        return getattr(config, "electric_force", None) is not None

    @staticmethod
    def init_state(setup: SimulationSetup) -> dict[str, Any]:
        params = _get_electric_params(setup)
        if params is None:
            return {}

        nx, ny, nz = setup.grid_shape[0], setup.grid_shape[1], setup.grid_shape[2]
        y_vals = jnp.linspace(params.voltage_bottom, params.voltage_top, ny)
        potential = jnp.broadcast_to(y_vals[None, :], (nx, ny))[:, :, None, None, None]
        potential = jnp.broadcast_to(potential, (nx, ny, nz, 1, 1))
        return {"h": _equilibrium_h(potential, setup.lattice.w)}

    @staticmethod
    def update_state(setup: SimulationSetup, prev_state: State, new_state: State) -> State:
        params = _get_electric_params(setup)
        if params is None or prev_state.h is None:
            return new_state

        rho_4d = jnp.sum(prev_state.f, axis=3, keepdims=True)  # (nx, ny, nz, 1, 1)
        rho_2d = rho_4d[:, :, 0, :, 0]  # Extract spatial slice to (nx, ny, 1)
        conductivity = _rho_to_phi(
            rho_2d,
            params.conductivity_liquid,
            params.conductivity_vapour,
        )

        potential = jnp.sum(prev_state.h, axis=3, keepdims=True)  # (nx, ny, nz, 1, 1)
        h_eq = _equilibrium_h(potential, setup.lattice.w)

        tau_e = 3.0 * conductivity[:, :, :, None, None] + 0.5
        omega_e = 1.0 / tau_e
        h_col = (1.0 - omega_e) * prev_state.h + omega_e * h_eq

        top_potential = jnp.full((prev_state.h.shape[0], 1, prev_state.h.shape[2], 1, 1), params.voltage_top)
        h_col = h_col.at[:, -1:, :, :, :].set(_equilibrium_h(top_potential, setup.lattice.w))
        bottom_potential = jnp.full((prev_state.h.shape[0], 1, prev_state.h.shape[2], 1, 1), params.voltage_bottom)
        h_col = h_col.at[:, :1, :, :, :].set(_equilibrium_h(bottom_potential, setup.lattice.w))

        return new_state._replace(h=setup.streaming_fn(h_col, setup.lattice))
