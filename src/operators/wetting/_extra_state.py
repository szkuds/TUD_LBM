"""Wetting extra-state plugin."""

from __future__ import annotations
from typing import Any
import jax.numpy as jnp
from operators.wetting._contact_angle import compute_contact_angle
from operators.wetting._contact_line import compute_contact_line_location
from registry import extra_state_plugin
from state.state import WettingState


def _cfg_value(cfg: dict[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        if key in cfg:
            return float(cfg[key])
    return default


@extra_state_plugin(name="wetting")
class WettingExtraStatePlugin:
    """Initialises and updates wetting extra state."""

    @staticmethod
    def is_active(config: Any) -> bool:
        return getattr(config, "wetting_config", None) is not None

    @staticmethod
    def init_state(setup: Any) -> dict[str, Any]:
        wetting_cfg = setup.config.wetting_config
        if wetting_cfg is None:
            return {}

        f_init = setup.initial_f_fn()
        rho_init = jnp.sum(f_init, axis=2, keepdims=True)

        mp = setup.multiphase_params
        rho_mean = 0.5 * (mp.rho_l + mp.rho_v) if mp is not None else 1.0

        ca_left, ca_right = compute_contact_angle(rho_init, rho_mean)
        cll_left, cll_right = compute_contact_line_location(
            rho_init,
            ca_left,
            ca_right,
            rho_mean,
        )

        return {
            "wetting": WettingState(
                d_rho_left=jnp.array(_cfg_value(wetting_cfg, "d_rho_left", "d_rho_l", default=0.05)),
                d_rho_right=jnp.array(_cfg_value(wetting_cfg, "d_rho_right", "d_rho_r", default=0.05)),
                phi_left=jnp.array(_cfg_value(wetting_cfg, "phi_left", "phi_l", default=1.2)),
                phi_right=jnp.array(_cfg_value(wetting_cfg, "phi_right", "phi_r", default=1.2)),
                ca_left=ca_left,
                ca_right=ca_right,
                cll_left=cll_left,
                cll_right=cll_right,
                opt_state_left=None,
                opt_state_right=None,
            )
        }

    @staticmethod
    def update_state(setup: Any, prev_state: Any, new_state: Any, **context: Any) -> Any:
        if prev_state.wetting is None or setup.wetting_fn is None:
            return new_state

        updated_wetting = setup.wetting_fn(
            prev_state.wetting,
            new_state.rho,
            setup,
            prev_state.f,
            force_ext=context.get("force_ext"),
        )
        return new_state._replace(wetting=updated_wetting)
