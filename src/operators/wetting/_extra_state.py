"""Wetting extra-state plugin."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
import jax.numpy as jnp
from src.operators.wetting._contact_angle import compute_contact_angle
from src.operators.wetting._contact_line import compute_contact_line_location
from src.operators.wetting._params import WettingParams
from src.pipeline.state.state import State
from src.pipeline.state.state import WettingState
from src.registry import extra_state_plugin

if TYPE_CHECKING:
    from src.pipeline.setup import SimulationSetup


@extra_state_plugin(name="wetting")
class WettingExtraStatePlugin:
    """Initialises and updates wetting extra state."""

    @staticmethod
    def is_active(config: SimulationSetup) -> bool:
        return (
            getattr(config, "wetting_config", None) is not None
            or getattr(config, "hysteresis_config", None) is not None
        )

    @staticmethod
    def init_state(setup: SimulationSetup) -> dict[str, Any]:
        defaults = setup.config.wetting_defaults
        assert defaults is not None  # noqa: S101 - active only with a wetting_config (hysteresis gets a neutral one)
        params = WettingParams.from_defaults(defaults)

        if setup.initial_f_fn is None:
            msg = "initial_f_fn is required for wetting initial state"
            raise TypeError(msg)
        f_init = setup.initial_f_fn()
        rho_init = jnp.sum(f_init, axis=-2, keepdims=True)

        mp = setup.multiphase_params
        rho_mean = 0.5 * (mp.rho_l + mp.rho_v) if mp is not None else 1.0
        if setup.wetting_edge is None:
            msg = "wetting_edge is required for wetting initial state"
            raise TypeError(msg)
        edge = setup.wetting_edge

        ca_left, ca_right = compute_contact_angle(rho_init, jnp.array(rho_mean), edge=edge)
        cll_left, cll_right = compute_contact_line_location(
            rho_init,
            ca_left,
            ca_right,
            jnp.array(rho_mean),
            edge=edge,
        )

        wetting = WettingState(
            phi_left=params.phi_left,
            phi_right=params.phi_right,
            d_rho_left=params.d_rho_left,
            d_rho_right=params.d_rho_right,
            ca_left=ca_left,
            ca_right=ca_right,
            cll_left=cll_left,
            cll_right=cll_right,
        )
        # A hysteresis restart resumes the wall parameters and anchors it saved;
        # the angles above are always measured off the field.
        restored = setup.config.restored_wetting
        if restored is not None:
            wetting = wetting._replace(**{key: jnp.array(value) for key, value in restored.items()})
        return {"wetting": wetting}

    @staticmethod
    def update_state(
        setup: SimulationSetup,
        prev_state: State,
        new_state: State,
        **context: Any,  # noqa: ANN401
    ) -> State:
        if prev_state.wetting is None or setup.wetting_fn is None:
            return new_state

        updated_wetting = setup.wetting_fn(
            prev_state.wetting,
            new_state.rho,
            setup,
            trial_step_fn=context.get("trial_step_fn"),
            t=new_state.t,
        )
        return new_state._replace(wetting=updated_wetting)
