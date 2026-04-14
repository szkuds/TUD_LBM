"""Private helper: extra pytree leaves from registered force specs and wetting."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
import jax.numpy as jnp

if TYPE_CHECKING:
    from setup.simulation_setup import SimulationSetup


def _build_extra_state(setup: SimulationSetup) -> dict[str, Any]:
    """Collect extra State fields initialised by registered force specs and wetting.

    Some force implementations define additional fields that must be
    stored in the State pytree (e.g. electric potential ``h`` for
    electrokinetic flows). For wetting simulations, a WettingState is
    also initialised from the config.

    Returns an empty dict when no forces or wetting are registered,
    keeping the call site unconditional and simplifying the
    orchestrator logic.

    Args:
        setup: :class:`~setup.simulation_setup.SimulationSetup`.

    Returns:
        A dictionary mapping field names to initialised values.
        Empty when no forces/wetting are active.
    """
    extra: dict[str, Any] = {}

    # Initialize wetting state if applicable
    if setup.config.wetting_config is not None:
        from state.state import WettingState

        wetting_cfg = setup.config.wetting_config
        extra["wetting"] = WettingState(
            d_rho_left=jnp.array(wetting_cfg.get("d_rho_left", 0.05)),
            d_rho_right=jnp.array(wetting_cfg.get("d_rho_right", 0.05)),
            phi_left=jnp.array(wetting_cfg.get("phi_left", 1.2)),
            phi_right=jnp.array(wetting_cfg.get("phi_right", 1.2)),
            ca_left=jnp.array(0.0),  # Will be computed by hysteresis
            ca_right=jnp.array(0.0),
            cll_left=jnp.array(0.0),  # Will be computed by hysteresis
            cll_right=jnp.array(0.0),
            opt_state_left=None,
            opt_state_right=None,
        )

    if setup.forces is None:
        return extra

    # Collect force spec fields
    for spec in setup.forces.specs:
        extra.update(spec.init_fn(setup.grid_shape, setup.lattice, spec.precomputed))

    return extra
