"""The ``"guo"`` source term and the pressure of the equilibrium it pairs with.

A single-phase run takes ``"standard_equilibrium"`` with ``"guo"``. The source
must add the force to the momentum and nothing to the mass, and the equilibrium
must carry the isotropic pressure ``cs^2 * rho`` that the ``"wb"`` one omits.
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
import pytest
from src.lattice import build_lattice
from src.operators.equilibrium import build_equilibrium_fn
from src.operators.source_term import build_source_fn

CS2 = 1.0 / 3.0
SHAPE = (6, 5, 1)


@pytest.fixture
def lattice():
    return build_lattice("D2Q9")


@pytest.fixture
def fields():
    rng = np.random.default_rng(0)
    rho = jnp.asarray(1.0 + 0.1 * rng.random((*SHAPE, 1, 1)))
    u = jnp.asarray(0.05 * (rng.random((*SHAPE, 1, 2)) - 0.5))
    force = jnp.asarray(1e-3 * (rng.random((*SHAPE, 1, 2)) - 0.5))
    return rho, u, force


def _zero_gradient(grid: jnp.ndarray, wetting: object = None) -> jnp.ndarray:
    return jnp.zeros((*grid.shape[:-1], 2))


def test_guo_source_adds_the_force_to_the_momentum_and_no_mass(lattice, fields):
    rho, u, force = fields

    source = build_source_fn("guo")(rho, u, force, lattice, gradient=_zero_gradient)

    np.testing.assert_allclose(np.asarray(jnp.sum(source, axis=-2)), 0.0, atol=1e-15)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(source * lattice.c, axis=-2, keepdims=True)), np.asarray(force), atol=1e-15
    )


def test_guo_source_is_the_wb_source_without_its_density_gradient(lattice, fields):
    rho, u, force = fields

    guo = build_source_fn("guo")(rho, u, force, lattice, gradient=_zero_gradient)
    wb = build_source_fn("wb")(rho, u, force, lattice, gradient=_zero_gradient)

    np.testing.assert_allclose(np.asarray(guo), np.asarray(wb), atol=1e-15)


@pytest.mark.parametrize(("scheme", "pressure"), [("standard_equilibrium", CS2), ("wb", 0.0)])
def test_only_the_standard_equilibrium_carries_pressure(lattice, fields, scheme, pressure):
    rho, _, _ = fields
    at_rest = jnp.zeros((*SHAPE, 1, 2))

    feq = build_equilibrium_fn(scheme)(rho, at_rest, lattice)
    second_moment_xx = jnp.sum(feq * lattice.c[..., 0:1] ** 2, axis=-2, keepdims=True)

    np.testing.assert_allclose(np.asarray(second_moment_xx), pressure * np.asarray(rho), atol=1e-15)
