"""A periodic axis carries no preferred position.

Two states related by a pure lattice translation along a periodic axis must
evolve to the translated result. This held only after the stencil padding was
fixed: the predecessor left a zero-gradient seam at index 0 of every fully
periodic axis (see :mod:`src.operators.differential._pad_utils`), which is a
spurious wall pinned to the array origin.

A periodic domain has no restoring force for translation, so such a seam does
not merely perturb — it integrates. Measured on a 201-cell periodic wetting
wall, a bubble initialised at the domain centre held its centroid to
``100.00000`` over 2000 steps while *its own 40-cell roll* walked to 139.147,
and on the full 50 000-step run the inclusion drifted some 8 cells with no
force applied. This is the concrete form of the D4/lattice-symmetry invariant
the multiphase notes warn about.
"""

from __future__ import annotations
from typing import Any
import jax.numpy as jnp
import numpy as np
import pytest
from src.config.simulation_config import SimulationConfig
from src.operators.equilibrium._equilibrium import compute_equilibrium
from src.pipeline.runner import init_state
from src.pipeline.runner import run
from src.pipeline.setup import build_setup

_NX, _NY = 16, 12
_RHO_L, _RHO_V = 1.0, 0.33
_SHIFT = 5
_STEPS = 6


def _config(bc_config: dict[str, Any] | None) -> SimulationConfig:
    return SimulationConfig(
        grid_shape=(_NX, _NY),
        tau=0.8,
        nt=_STEPS,
        sim_type="multiphase",
        eos="double-well",
        kappa=0.017,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=4,
        bc_config=bc_config,
    )


def _seeded_state(setup):
    """A state with an off-centre inclusion, so a roll is not a no-op."""
    x, y = np.meshgrid(np.arange(_NX), np.arange(_NY), indexing="ij")
    distance = np.sqrt((x - 4) ** 2 + (y - _NY / 2) ** 2)
    rho_2d = 0.5 * (_RHO_L + _RHO_V) + 0.5 * (_RHO_L - _RHO_V) * np.tanh((distance - 3.0) / 2.0)
    rho = jnp.asarray(rho_2d.reshape(_NX, _NY, 1, 1, 1))
    u = jnp.zeros((_NX, _NY, 1, 1, setup.lattice.d))
    return init_state(setup)._replace(f=compute_equilibrium(rho, u, setup.lattice), rho=rho, u=u)


@pytest.mark.parametrize(
    "bc_config",
    [
        pytest.param(None, id="all-periodic"),
        pytest.param(
            {"left": "periodic", "right": "periodic", "top": "symmetry", "bottom": "bounce-back"},
            id="periodic-x-walled-y",
        ),
    ],
)
def test_a_roll_along_a_periodic_axis_commutes_with_the_step(bc_config):
    """Exact, not approximate: there is no tolerance to hide a seam behind.

    The wetting band is deliberately not exercised here. Its window, contiguity
    label and nearest-anchor split are flat-index, so a window clipped by the
    array end (or a contact line straddling the seam) still carries a small
    translation dependence — about 2.5e-05 in rho after one step on a 201-cell
    production config, three orders below the seam this padding fix removed.
    Making that band periodic-aware is separate work.
    """
    config = _config(bc_config)
    setup = build_setup(config)

    state = _seeded_state(setup)
    rolled = state._replace(
        f=jnp.roll(state.f, _SHIFT, axis=0),
        rho=jnp.roll(state.rho, _SHIFT, axis=0),
        u=jnp.roll(state.u, _SHIFT, axis=0),
    )

    evolved, _ = run(setup, state, nt=_STEPS)
    evolved_rolled, _ = run(setup, rolled, nt=_STEPS)

    np.testing.assert_allclose(
        np.asarray(jnp.roll(evolved.rho, _SHIFT, axis=0)),
        np.asarray(evolved_rolled.rho),
        rtol=0.0,
        atol=1e-12,
    )
