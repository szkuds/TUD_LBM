"""Mass conservation for a fully closed domain.

Buoyancy-driven runs close the tangential axis (see
:mod:`src.operators.force._gravity_masked`): a periodic axis cannot support the
linear pressure ramp that balances the tangential component of gravity, so the
liquid would shear into a channel flow instead of standing still. That makes
the closed-box boundary combination load-bearing, and its two halves have to
agree exactly — streaming zeroes the layers a roll wrapped across a
non-periodic axis, and bounce-back has to re-inject precisely those.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.lattice.lattice import build_lattice
from src.operators.boundary import build_bc
from src.operators.streaming import build_streaming_fn

NX, NY, NZ = 8, 6, 1

#: What a buoyancy-driven bubble run uses: closed in x, wetting wall on one
#: side in y. ``"wetting"`` dispatches to bounce-back populations.
CLOSED_BOX = {"left": "bounce-back", "right": "bounce-back", "top": "wetting", "bottom": "bounce-back"}


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


def _total_mass_after(bc_config, lattice, steps=25):
    stream = build_streaming_fn("standard", bc_config)
    bc_fn = build_bc(bc_config, lattice)

    f = jax.random.uniform(jax.random.PRNGKey(0), (NX, NY, NZ, lattice.q, 1))
    masses = [float(jnp.sum(f))]
    for _ in range(steps):
        f = bc_fn(stream(f, lattice), f, None)
        masses.append(float(jnp.sum(f)))
    return np.array(masses)


def test_closed_box_conserves_mass(lattice):
    """No population escapes and none is injected, over many steps."""
    masses = _total_mass_after(CLOSED_BOX, lattice)
    np.testing.assert_allclose(masses, masses[0], rtol=1e-12)


def test_closing_x_does_not_leak_relative_to_the_periodic_case(lattice):
    """Closing x must be as tight as the all-periodic baseline, not merely close.

    Streaming keys periodicity off *both* edges of an axis, so flipping
    ``left``/``right`` changes which layers get zeroed; the bounce-back pass has
    to refill exactly those and no others.
    """
    periodic_x = {**CLOSED_BOX, "left": "periodic", "right": "periodic"}

    closed = _total_mass_after(CLOSED_BOX, lattice)
    periodic = _total_mass_after(periodic_x, lattice)

    assert abs(closed[-1] / closed[0] - 1.0) <= abs(periodic[-1] / periodic[0] - 1.0) + 1e-12
