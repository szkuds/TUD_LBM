"""The ``"wb_improved"`` equilibrium: the decoupled-viscosity term of Zhang, Guo & Wang (2022).

The improved equilibrium (registered separately, built on the untouched ``wb``)
carries ``cs2*rho*A*S`` in its second moment, so the shear moments can relax at
``1/lambda_v`` while ``nu = cs2*(lambda_v - 1/2 - A)``. Asserted through the
moments (Appendix C, Eq. C8/C9) rather than the PDF's explicit populations, whose
rest-population sign is a transcription error.
"""

from __future__ import annotations
import functools
import jax.numpy as jnp
import numpy as np
import pytest
from src.config.viscosity_params import build_viscosity_params
from src.lattice.lattice import build_lattice
from src.operators.equilibrium import build_equilibrium_fn
from src.operators.source_term import build_source_fn
from src.registry import get_operators

CS2 = 1.0 / 3.0
NX, NY = 8, 6
_PERIODIC = ("wrap", "wrap", "wrap", "wrap")
_EDGE = ("edge", "edge", "edge", "edge")
_PARAMS = build_viscosity_params(relaxation_time=1.0, tau_liquid=0.75, tau_gas=0.6, rho_l=2.0, rho_v=0.5)


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


def build_gradient(lattice, pad_modes):
    """The pipeline's gradient accessor.

    Entered through ``src.pipeline``: importing ``src.operators.differential``
    cold hits the pre-existing differential -> wetting -> pipeline import cycle.
    """
    import src.pipeline  # noqa: F401
    from src.operators.differential import build_gradient_fn

    return build_gradient_fn(lattice, pad_modes)


def _improved(lattice, params, pad_modes=_PERIODIC):
    """``"wb_improved"`` bound the way ``build_setup`` binds it."""
    return functools.partial(
        get_operators("equilibrium")["wb_improved"].target,
        viscosity=params,
        reference_pressure=None,
        gradient=build_gradient(lattice, pad_modes),
    )


def _fields(seed: int = 0):
    rng = np.random.default_rng(seed)
    rho = jnp.asarray(rng.uniform(0.5, 2.0, (NX, NY, 1, 1, 1)))
    u = jnp.asarray(rng.uniform(-0.05, 0.05, (NX, NY, 1, 1, 2)))
    return rho, u


def _moments(feq, lattice):
    f = np.asarray(feq)
    c = np.asarray(lattice.c)
    zeroth = f.sum(axis=-2)[..., 0]
    first = (f * c).sum(axis=-2)
    second = (f[..., None] * c[..., :, None] * c[..., None, :]).sum(axis=-3)
    return zeroth, first, second


def _linear_velocity(g: np.ndarray) -> jnp.ndarray:
    x, y = np.meshgrid(np.arange(NX, dtype=float), np.arange(NY, dtype=float), indexing="ij")
    u = np.stack([g[0, 0] * x + g[0, 1] * y, g[1, 0] * x + g[1, 1] * y], axis=-1)
    return jnp.asarray(u[:, :, None, None, :])


def _stress_moment(lattice, params, rho, u):
    """``(sum cc feq - sum cc feq_wb) / (cs2 rho)`` — the ``A S`` the equilibrium carries."""
    improved = _moments(_improved(lattice, params, _EDGE)(rho, u, lattice), lattice)[2]
    plain = _moments(build_equilibrium_fn("wb")(rho, u, lattice), lattice)[2]
    return (improved - plain) / (CS2 * np.asarray(rho)[..., 0, :, None])


def test_mass_and_momentum_are_those_of_wb(lattice):
    rho, u = _fields()

    plain = _moments(build_equilibrium_fn("wb")(rho, u, lattice), lattice)
    improved = _moments(_improved(lattice, _PARAMS)(rho, u, lattice), lattice)

    np.testing.assert_allclose(improved[0], plain[0], atol=1e-6)
    np.testing.assert_allclose(improved[1], plain[1], atol=1e-6)


def test_moments_match_eq_c8_on_a_linear_shear(lattice):
    """``sum feq = rho``, ``sum c feq = rho u``, ``sum cc feq = rho uu + cs2 rho A S`` (p_g = 0).

    A linear velocity field makes ``S = g + g^T`` exact in the interior; at uniform
    ``rho = rho_l`` the paper's ``A = lambda_v - tau = 0.25``.
    """
    g = np.array([[0.002, -0.003], [0.004, 0.001]])
    u = _linear_velocity(g)
    rho = jnp.full((NX, NY, 1, 1, 1), _PARAMS.rho_l)

    zeroth, first, second = _moments(_improved(lattice, _PARAMS, _EDGE)(rho, u, lattice), lattice)

    u_np = np.asarray(u)[..., 0, :]
    interior = (slice(1, -1), slice(1, -1), 0)
    np.testing.assert_allclose(zeroth, np.asarray(rho)[..., 0, 0], atol=1e-6)
    np.testing.assert_allclose(first, _PARAMS.rho_l * u_np, atol=1e-6)
    uu = u_np[..., :, None] * u_np[..., None, :]
    expected = _PARAMS.rho_l * (uu + CS2 * 0.25 * (g + g.T))
    np.testing.assert_allclose(second[interior], expected[interior], atol=1e-6)


def test_without_its_terms_the_improved_equilibrium_is_wb(lattice):
    rho, u = _fields(2)

    np.testing.assert_array_equal(
        np.asarray(_improved(lattice, None)(rho, u, lattice)),
        np.asarray(build_equilibrium_fn("wb")(rho, u, lattice)),
    )


def test_wb_source_moments_match_eq_c9(lattice):
    """The unchanged ``wb`` source carries no mass and exactly the force: ``sum F_i = 0``, ``sum c F_i = F``."""
    rho, u = _fields(3)
    force = jnp.asarray(np.random.default_rng(3).uniform(-1e-3, 1e-3, (NX, NY, 1, 1, 2)))
    gradient = build_gradient(lattice, _PERIODIC)

    source = np.asarray(build_source_fn()(rho, u, force, lattice, gradient=gradient))

    np.testing.assert_allclose(source.sum(axis=-2), 0.0, atol=1e-8)
    np.testing.assert_allclose((source * np.asarray(lattice.c)).sum(axis=-2), np.asarray(force)[..., 0, :], atol=1e-8)


@pytest.mark.parametrize(
    ("lambda_v", "tau", "a_expected", "nu_expected"),
    [(1.0, 1.0, 0.0, 1.0 / 6.0), (1.0, 0.75, 0.25, 0.0833333)],
)
def test_a_reproduces_the_papers_viscosities(lattice, lambda_v, tau, a_expected, nu_expected):
    """Paper Sec. III: ``lambda_v = 1`` with ``A = 0`` gives 1/6 and ``A = 0.25`` gives 0.0833."""
    params = build_viscosity_params(relaxation_time=lambda_v, tau_liquid=tau, tau_gas=tau, rho_l=2.0, rho_v=0.5)
    g = np.array([[0.0, 0.01], [0.0, 0.0]])
    rho = jnp.full((NX, NY, 1, 1, 1), 1.3)

    stress = _stress_moment(lattice, params, rho, _linear_velocity(g))

    assert params.nu_l == pytest.approx(nu_expected, rel=1e-5)
    assert CS2 * (lambda_v - 0.5 - a_expected) == pytest.approx(params.nu_l, rel=1e-5)
    expected = np.broadcast_to(a_expected * (g + g.T), (NX - 2, NY - 2, 2, 2))
    np.testing.assert_allclose(stress[1:-1, 1:-1, 0], expected, atol=1e-8)


def test_a_interpolates_the_viscosity_linearly_between_the_phases(lattice):
    """At ``rho_l`` / ``rho_v`` ``A`` is ``lambda_v - tau`` / ``lambda_v - tau_gas``; linear between."""
    params = build_viscosity_params(relaxation_time=1.0, tau_liquid=0.75, tau_gas=0.55, rho_l=6.0, rho_v=1.0)
    g = np.array([[0.0, 0.01], [0.0, 0.0]])  # S_xy = S_yx = 0.01

    for rho_value, a_expected in ((6.0, 0.25), (1.0, 0.45), (3.5, 0.35)):
        rho = jnp.full((NX, NY, 1, 1, 1), rho_value)
        stress = _stress_moment(lattice, params, rho, _linear_velocity(g))
        assert stress[3, 3, 0, 0, 1] == pytest.approx(0.01 * a_expected, rel=1e-5)
