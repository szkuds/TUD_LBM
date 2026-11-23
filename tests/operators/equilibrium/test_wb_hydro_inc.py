"""The well-balanced equilibrium with a pressure, and the force split that pays it back.

Zhang et al., Int. J. Multiphase Flow 198, 105668 (2026): the equilibrium carries
``p = rho_0 (mu - mu_b)`` in its second moment and the interparticle force becomes
``-(rho - rho_0) grad(mu)``, so the momentum equation still sees ``-rho grad(mu)``.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.config.multiphase import MultiphaseParams
from src.config.multiphase import multiphase_params
from src.lattice import build_lattice
from src.operators.equilibrium import build_equilibrium_fn
from src.operators.macroscopic._multiphase import compute_macroscopic_multiphase
from src.operators.macroscopic.eos import build_eos_fn
from src.operators.macroscopic.eos import coexistence_densities
from src.operators.macroscopic.eos import coexistence_potential
from src.simulation_io.readers import DictAdapter
from tests.support.differential import bound_operators

NX, NY = 24, 12
_CS = {"a_eos": 0.00031459670905604266, "b_eos": 0.1490857142857143, "r_eos": 1.0, "t_eos": 0.00039808421247983624}
_RHO_0 = 0.0168045


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


def _fields() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """A smooth, non-uniform ``(rho, u, pressure)``."""
    x = jnp.arange(NX)[:, None] * (2 * jnp.pi / NX)
    y = jnp.arange(NY)[None, :] * (2 * jnp.pi / NY)
    rho = (1.0 + 0.3 * jnp.sin(x) * jnp.cos(y))[:, :, None, None, None]
    u = jnp.stack([0.03 * jnp.cos(x) + 0.0 * y, 0.02 * jnp.sin(y) + 0.0 * x], axis=-1)[:, :, None, None, :]
    pressure = (0.02 * jnp.cos(x) * jnp.sin(y))[:, :, None, None, None]
    return rho, u, pressure


def _mp(eos: str, rho_0: float | None) -> MultiphaseParams:
    """The run's parameters as the config derives them, ``mu_b`` included."""
    parameters = _CS if eos == "carnahan-starling" else {}
    section: dict[str, object] = {
        "eos": eos,
        "kappa": 0.015,
        "rho_l": 12.18,
        "rho_v": 0.015,
        "interface_width": 5,
        **parameters,
    }
    if rho_0 is not None:
        section["equilibrium_reference_density"] = rho_0
    raw = {
        "grid_shape": (NX, NY),
        "nt": 10,
        "equilibrium": "wb" if rho_0 is None else "wb_hydro_inc",
        "multiphase": section,
    }
    params = multiphase_params(DictAdapter().load(raw))
    assert params is not None
    return params


def _interface_populations(lattice) -> jnp.ndarray:
    """Populations at rest with a liquid slab between two vapour regions."""
    x = jnp.arange(NX)
    profile = 0.015 + 0.5 * (12.18 - 0.015) * (jnp.tanh((x - 6) / 2.5) - jnp.tanh((x - 18) / 2.5))
    rho = jnp.broadcast_to(profile[:, None], (NX, NY))[:, :, None, None, None]
    return build_equilibrium_fn("wb")(rho, jnp.zeros((NX, NY, 1, 1, 2)), lattice)


def test_the_moments_are_density_momentum_and_pressure_plus_momentum_flux(lattice):
    rho, u, pressure = _fields()

    feq = build_equilibrium_fn("wb_hydro_inc")(rho, u, lattice, pressure)

    c = lattice.c
    np.testing.assert_allclose(jnp.sum(feq, axis=-2, keepdims=True), rho, atol=1e-14)
    np.testing.assert_allclose(jnp.sum(feq * c, axis=-2, keepdims=True), rho * u, atol=1e-14)
    second = jnp.einsum(
        "xyzqo,xyzqa,xyzqb->xyzab", feq, jnp.broadcast_to(c, (*feq.shape[:4], 2)), c * jnp.ones_like(feq)
    )
    expected = pressure[..., 0, :, None] * jnp.eye(2) + rho[..., 0, :, None] * jnp.einsum("xyzoa,xyzob->xyzab", u, u)
    np.testing.assert_allclose(second, expected, atol=1e-14)


def test_without_a_pressure_it_is_the_wb_equilibrium(lattice):
    rho, u, _ = _fields()

    feq = build_equilibrium_fn("wb_hydro_inc")(rho, u, lattice)

    np.testing.assert_array_equal(feq, build_equilibrium_fn("wb")(rho, u, lattice))


def test_the_pressure_only_moves_mass_between_the_rest_and_the_moving_directions(lattice):
    rho, u, pressure = _fields()
    equilibrium = build_equilibrium_fn("wb_hydro_inc")

    shift = equilibrium(rho, u, lattice, pressure) - equilibrium(rho, u, lattice)

    np.testing.assert_allclose(shift[..., 1:, :], 3.0 * lattice.w[..., 1:, :] * pressure, atol=1e-15)
    np.testing.assert_allclose(jnp.sum(shift, axis=-2), 0.0, atol=1e-15)


@pytest.mark.parametrize("eos", ["carnahan-starling", "double-well"])
def test_the_force_and_the_pressure_gradient_add_up_to_the_wb_force(lattice, eos):
    """``-(rho - rho_0) grad(mu) - grad(rho_0 (mu - mu_b))`` is ``-rho grad(mu)``."""
    gradient, laplacian = bound_operators(lattice)
    f = _interface_populations(lattice)

    _, _, force_wb, pressure_wb, none = compute_macroscopic_multiphase(
        f, lattice, mp=_mp(eos, None), gradient_standard=gradient, laplacian_density=laplacian
    )
    rho, _, force, pressure, lattice_pressure = compute_macroscopic_multiphase(
        f, lattice, mp=_mp(eos, _RHO_0), gradient_standard=gradient, laplacian_density=laplacian
    )

    assert none is None
    assert lattice_pressure is not None
    assert float(jnp.max(jnp.abs(force_wb))) > 1e-4  # an interface, not a uniform field
    np.testing.assert_allclose(force - gradient(lattice_pressure), force_wb, atol=1e-12)
    np.testing.assert_allclose(force, force_wb * (rho - _RHO_0) / rho, atol=1e-12)
    np.testing.assert_array_equal(pressure, pressure_wb)  # the reported EOS pressure is untouched


def test_the_lattice_pressure_vanishes_in_the_bulk_at_coexistence(lattice):
    """``mu_b`` is the Maxwell chemical potential, so a bulk phase at coexistence carries ``p = 0``."""
    gradient, laplacian = bound_operators(lattice)
    mp = _mp("carnahan-starling", _RHO_0)
    rho_v, rho_l = coexistence_densities(mp.eos, mp._asdict()) or (0.0, 0.0)

    for density in (rho_v, rho_l):
        f = jnp.full((NX, NY, 1, 9, 1), density) * lattice.w
        *_, lattice_pressure = compute_macroscopic_multiphase(
            f, lattice, mp=mp, gradient_standard=gradient, laplacian_density=laplacian
        )
        assert lattice_pressure is not None
        np.testing.assert_allclose(lattice_pressure, 0.0, atol=1e-9)


def test_the_coexistence_potential_is_the_same_in_both_phases():
    mp = _mp("carnahan-starling", None)
    rho_v, rho_l = coexistence_densities(mp.eos, mp._asdict()) or (0.0, 0.0)
    eos = build_eos_fn(mp.eos, mp)

    assert coexistence_potential(mp) == pytest.approx(float(eos(jnp.asarray(rho_l))), abs=1e-10)
    assert coexistence_potential(mp) == pytest.approx(float(eos(jnp.asarray(rho_v))), abs=1e-12)
    # The double-well's wells are its coexistence densities, where mu_0 = 0.
    assert coexistence_potential(_mp("double-well", None)) == pytest.approx(0.0, abs=1e-12)


def test_the_macroscopic_operator_with_a_reference_density_is_jittable(lattice):
    gradient, laplacian = bound_operators(lattice)
    mp = _mp("carnahan-starling", _RHO_0)

    @jax.jit
    def moments(f: jnp.ndarray) -> jnp.ndarray:
        *_, lattice_pressure = compute_macroscopic_multiphase(
            f, lattice, mp=mp, gradient_standard=gradient, laplacian_density=laplacian
        )
        assert lattice_pressure is not None
        return lattice_pressure

    assert bool(jnp.all(jnp.isfinite(moments(_interface_populations(lattice)))))
