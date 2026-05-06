"""Wetting extra-state plugin."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
import jax.numpy as jnp
from tud_lbm.operators.wetting._contact_angle import compute_contact_angle
from tud_lbm.operators.wetting._contact_line import compute_contact_line_location
from tud_lbm.pipeline.state.state import State
from tud_lbm.pipeline.state.state import WettingState
from tud_lbm.registry import extra_state_plugin

if TYPE_CHECKING:
    from tud_lbm.pipeline.setup import SimulationSetup


def _cfg_value(cfg: dict[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        if key in cfg:
            return float(cfg[key])
    return default


@extra_state_plugin(name="wetting")
class WettingExtraStatePlugin:
    """Initialises and updates wetting extra state."""

    @staticmethod
    def is_active(config: SimulationSetup) -> bool:
        return getattr(config, "wetting_config", None) is not None

    @staticmethod
    def init_state(setup: SimulationSetup) -> dict[str, Any]:
        wetting_cfg = setup.config.wetting_config
        if wetting_cfg is None:
            return {}

        f_init = setup.initial_f_fn()
        rho_init = jnp.sum(f_init, axis=-2, keepdims=True)

        mp = setup.multiphase_params
        rho_mean = 0.5 * (mp.rho_l + mp.rho_v) if mp is not None else 1.0

        ca_left, ca_right = compute_contact_angle(rho_init, jnp.array(rho_mean))
        cll_left, cll_right = compute_contact_line_location(
            rho_init,
            ca_left,
            ca_right,
            jnp.array(rho_mean),
        )

        return {
            "wetting": WettingState(
                phi_left=wetting_cfg["phi_left"],
                phi_right=wetting_cfg["phi_right"],
                d_rho_left=wetting_cfg["d_rho_left"],
                d_rho_right=wetting_cfg["d_rho_right"],
                ca_left=ca_left,
                ca_right=ca_right,
                cll_left=cll_left,
                cll_right=cll_right,
            ),
        }

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
        )
        return new_state._replace(wetting=updated_wetting)
