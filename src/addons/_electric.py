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

**The plates are the solid surfaces of the bottom and top edges** (``PLATES``),
which the config requires to be walls. As for the fluid's half-way bounce-back,
the surface lies half a cell outside the edge row, so the plate distance is
``ny``. The potential is imposed there with the registered ``"bounce-back"``
reflection applied to ``2*w*U_plate - h_col``: since ``w_i = w_opp(i)``, each
incoming ``h_i`` becomes ``-h_col[opp(i)] + 2*w_i*U_plate``. The state therefore
carries no population from outside the domain, and the effective plate does not
move with ``tau_h``. The earlier scheme reset the edge rows to ``w*U_plate``
before streaming, which put the plate at ``1 - tau_h`` cells and left the
streamed edge rows in the state.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple
import jax.numpy as jnp
from src.addons.hooks import Hooks
from src.operators.force._electric import PLATES
from src.operators.force._electric import ElectricParams
from src.operators.force._electric import equilibrium_h
from src.operators.force._electric import plate_voltage
from src.operators.force._electric import rho_to_phi
from src.registry import addon_operator
from src.registry import get_operators

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State
    from src.protocols import BoundaryCondition
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
        # Linear between the plate surfaces at y = -1/2 and y = ny - 1/2.
        fraction = (jnp.arange(ny) + 0.5) / ny
        y_vals = params.voltage_bottom + (params.voltage_top - params.voltage_bottom) * fraction
        potential = jnp.broadcast_to(y_vals[None, :, None, None, None], (nx, ny, nz, 1, 1))
        return ElectricState(h=equilibrium_h(potential, setup.lattice.w))

    return init_state


def _potential_update(params: ElectricParams, reflect: BoundaryCondition) -> PostStepHook:
    def post_step(setup: SimulationSetup, prev: State, new: State) -> State:
        h = prev.ext[NAME].h
        lattice = setup.lattice
        rho_4d = jnp.sum(prev.f, axis=3, keepdims=True)  # (nx, ny, nz, 1, 1)
        rho_2d = rho_4d[:, :, 0, :, 0]  # Extract spatial slice to (nx, ny, 1)
        conductivity = rho_to_phi(rho_2d, params.conductivity_liquid, params.conductivity_vapour)

        potential = jnp.sum(h, axis=3, keepdims=True)  # (nx, ny, nz, 1, 1)
        h_eq = equilibrium_h(potential, lattice.w)

        tau_e = 3.0 * conductivity[:, :, :, None, None] + 0.5
        omega_e = 1.0 / tau_e
        h_col = (1.0 - omega_e) * h + omega_e * h_eq

        h_next = setup.streaming_fn(h_col, lattice)
        for edge in PLATES:
            h_next = reflect(h_next, 2.0 * lattice.w * plate_voltage(params, edge) - h_col, lattice, edge)

        return new._replace(ext=new.ext | {NAME: ElectricState(h=h_next)})

    return post_step


@addon_operator(name=NAME)
def build_electric_addon(config: SimulationConfig) -> Hooks[ElectricState] | None:
    """The electric-potential hooks, or ``None`` without an ``electric_force`` section."""
    section = config.forces.get("electric_force")
    if section is None:
        return None
    # Imported for its registration side effect: the lookup below needs "bounce-back".
    import src.operators.boundary  # noqa: F401

    params = ElectricParams(**section)
    reflect = get_operators("boundary_condition")["bounce-back"].target
    return Hooks(post_step=_potential_update(params, reflect), init_state=_init_state(params))
