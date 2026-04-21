"""Extra-state plugins for force modules."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
import jax.numpy as jnp
from operators.force._electric import ElectricParams
from operators.force._electric import _equilibrium_h
from operators.force._electric import _rho_to_phi
from registry import extra_state_plugin

if TYPE_CHECKING:
    from config import SimulationConfig
    from setup import SimulationSetup
    from state import State


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

        nx, ny = setup.grid_shape[:2]
        y_vals = jnp.linspace(params.voltage_bottom, params.voltage_top, ny)
        potential = jnp.broadcast_to(y_vals[None, :], (nx, ny))[:, :, None, None]
        return {"h": _equilibrium_h(potential, setup.lattice.w)}

    @staticmethod
    def update_state(setup: SimulationSetup, prev_state: State, new_state: State) -> State:
        params = _get_electric_params(setup)
        if params is None or prev_state.h is None:
            return new_state

        rho_2d = jnp.sum(prev_state.f, axis=2)[:, :, 0]
        conductivity = _rho_to_phi(
            rho_2d,
            params.conductivity_liquid,
            params.conductivity_vapour,
        )

        potential = jnp.sum(prev_state.h, axis=2, keepdims=True)
        h_eq = _equilibrium_h(potential, setup.lattice.w)

        tau_e = 3.0 * conductivity[:, :, None, None] + 0.5
        omega_e = 1.0 / tau_e
        h_col = (1.0 - omega_e) * prev_state.h + omega_e * h_eq

        top_potential = jnp.full((prev_state.h.shape[0], 1, 1, 1), params.voltage_top)
        h_col = h_col.at[:, -1:, :, :].set(_equilibrium_h(top_potential, setup.lattice.w))
        bottom_potential = jnp.full((prev_state.h.shape[0], 1, 1, 1), params.voltage_bottom)
        h_col = h_col.at[:, :1, :, :].set(_equilibrium_h(bottom_potential, setup.lattice.w))

        return new_state._replace(h=setup.streaming_fn(h_col, setup.lattice))
