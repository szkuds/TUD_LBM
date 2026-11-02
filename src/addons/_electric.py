"""Electric addon — the electric-potential distributions ``h``.

The leaky-dielectric potential is a second lattice Boltzmann equation
(Zhang & Kwok 2005, Eqs. 15-18) carried in ``State.ext["electric"] = ElectricState(h)``.
The electric *force* is a plain force (``electric_force`` under the ``force``
kind) that reads ``h``; this addon only owns ``h``:

* ``init_state``: the equilibrium of the linear potential between the two plates.
* ``post_step``: collide ``h(t)`` with ``tau_h = 3*sigma + 0.5``, where sigma comes
  from the density at the same time level, and stream it to ``h(t+1)``. Both are
  read from the state *before* the step: pairing ``h(t)`` with ``sigma(t+1)`` would
  change the scheme.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple
import jax.numpy as jnp
from src.addons.hooks import Hooks
from src.operators.force._electric import ElectricParams
from src.operators.force._electric import equilibrium_h
from src.operators.force._electric import rho_to_phi
from src.registry import addon_operator

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State
    from src.protocols import InitStateHook
    from src.protocols import PostStepHook

#: This addon's ``State.ext`` key, and its registered name.
NAME = "electric"


class ElectricState(NamedTuple):
    """The electric addon's substate, ``State.ext["electric"]``.

    Attributes:
        h: Electric-potential distributions, shape ``(nx, ny, nz, q, 1)``.
    """

    h: jnp.ndarray


def _init_state(params: ElectricParams) -> InitStateHook[ElectricState]:
    def init_state(setup: SimulationSetup) -> ElectricState:
        nx, ny, nz = setup.grid_shape[0], setup.grid_shape[1], setup.grid_shape[2]
        y_vals = jnp.linspace(params.voltage_bottom, params.voltage_top, ny)
        potential = jnp.broadcast_to(y_vals[None, :], (nx, ny))[:, :, None, None, None]
        potential = jnp.broadcast_to(potential, (nx, ny, nz, 1, 1))
        return ElectricState(h=equilibrium_h(potential, setup.lattice.w))

    return init_state


def _potential_update(params: ElectricParams) -> PostStepHook:
    def post_step(setup: SimulationSetup, prev: State, new: State) -> State:
        h = prev.ext[NAME].h
        rho_4d = jnp.sum(prev.f, axis=3, keepdims=True)  # (nx, ny, nz, 1, 1)
        rho_2d = rho_4d[:, :, 0, :, 0]  # Extract spatial slice to (nx, ny, 1)
        conductivity = rho_to_phi(rho_2d, params.conductivity_liquid, params.conductivity_vapour)

        potential = jnp.sum(h, axis=3, keepdims=True)  # (nx, ny, nz, 1, 1)
        h_eq = equilibrium_h(potential, setup.lattice.w)

        tau_e = 3.0 * conductivity[:, :, :, None, None] + 0.5
        omega_e = 1.0 / tau_e
        h_col = (1.0 - omega_e) * h + omega_e * h_eq

        top_potential = jnp.full((h.shape[0], 1, h.shape[2], 1, 1), params.voltage_top)
        h_col = h_col.at[:, -1:, :, :, :].set(equilibrium_h(top_potential, setup.lattice.w))
        bottom_potential = jnp.full((h.shape[0], 1, h.shape[2], 1, 1), params.voltage_bottom)
        h_col = h_col.at[:, :1, :, :, :].set(equilibrium_h(bottom_potential, setup.lattice.w))

        return new._replace(ext=new.ext | {NAME: ElectricState(h=setup.streaming_fn(h_col, setup.lattice))})

    return post_step


@addon_operator(name=NAME)
def build_electric_addon(config: SimulationConfig) -> Hooks[ElectricState] | None:
    """The electric-potential hooks, or ``None`` without an ``electric_force`` section."""
    section = config.active_forces.get("electric_force")
    if section is None:
        return None
    params = ElectricParams(**section)
    return Hooks(post_step=_potential_update(params), init_state=_init_state(params))
