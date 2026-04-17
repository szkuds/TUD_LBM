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
        from operators.initialise import build_initialise_fn
        from operators.wetting._contact_angle import compute_contact_angle
        from operators.wetting._contact_line import compute_contact_line_location

        wetting_cfg = setup.config.wetting_config

        # Compute initial rho from f to seed contact angles and contact-line locations
        init_type = setup.config.init_type
        kw: dict = {}
        mp = setup.multiphase_params
        if mp is not None:
            kw.update(rho_l=mp.rho_l, rho_v=mp.rho_v, interface_width=mp.interface_width)
        if init_type == "init_from_file" and "npz_path" not in kw and setup.config.init_dir is not None:
            kw["npz_path"] = setup.config.init_dir

        nx, ny = setup.grid_shape[0], setup.grid_shape[1]
        f_init = build_initialise_fn(init_type)(nx, ny, setup.lattice, **kw)
        rho_init = jnp.sum(f_init, axis=2, keepdims=True)
        rho_mean = 0.5 * (mp.rho_l + mp.rho_v)

        # Measure initial contact angles and contact-line locations
        ca_left, ca_right = compute_contact_angle(rho_init, rho_mean)
        cll_left, cll_right = compute_contact_line_location(
            rho_init,
            ca_left,
            ca_right,
            rho_mean,
        )

        extra["wetting"] = WettingState(
            d_rho_left=jnp.array(wetting_cfg.get("d_rho_left", 0.05)),
            d_rho_right=jnp.array(wetting_cfg.get("d_rho_right", 0.05)),
            phi_left=jnp.array(wetting_cfg.get("phi_left", 1.2)),
            phi_right=jnp.array(wetting_cfg.get("phi_right", 1.2)),
            ca_left=ca_left,
            ca_right=ca_right,
            cll_left=cll_left,
            cll_right=cll_right,
            opt_state_left=None,
            opt_state_right=None,
        )

    if setup.forces is None:
        return extra

    # Collect force spec fields
    for spec in setup.forces.specs:
        extra.update(spec.init_fn(setup.grid_shape, setup.lattice, spec.precomputed))

    return extra
