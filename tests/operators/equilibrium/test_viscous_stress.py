"""The viscous-stress term of the decoupled-viscosity model (Zhang, Guo & Wang 2022).

The equilibrium carries ``cs2*rho*A*S`` in its second moment, so the shear
moments can relax at ``1/lambda_v`` while ``nu = cs2*(lambda_v - 1/2 - A)``.
Asserted through the moments (Appendix C, Eq. C8/C9) rather than the PDF's
explicit populations, whose rest-population sign is a transcription error.
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
import pytest
from src.config.viscosity_params import build_viscosity_params
from src.lattice.lattice import build_lattice
from src.operators.equilibrium import build_equilibrium_fn
from src.operators.equilibrium._viscous_stress import strain_rate
from src.operators.equilibrium._viscous_stress import viscous_stress
from src.operators.source_term import build_source_fn

CS2 = 1.0 / 3.0
NX, NY = 8, 6
_PERIODIC = ("wrap", "wrap", "wrap", "wrap")
_EDGE = ("edge", "edge", "edge", "edge")


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


def _fields(seed: int = 0):
    rng = np.random.default_rng(seed)
    rho = jnp.asarray(rng.uniform(0.5, 2.0, (NX, NY, 1, 1, 1)))
    u = jnp.asarray(rng.uniform(-0.05, 0.05, (NX, NY, 1, 1, 2)))
    raw = rng.uniform(-0.1, 0.1, (NX, NY, 1, 1, 2, 2))
    stress = jnp.asarray(raw + np.swapaxes(raw, -1, -2))
    return rho, u, stress


def _moments(feq, lattice):
    f = np.asarray(feq)
    c = np.asarray(lattice.c)
    zeroth = f.sum(axis=-2)[..., 0]
    first = (f * c).sum(axis=-2)
    second = (f[..., None] * c[..., :, None] * c[..., None, :]).sum(axis=-3)
    return zeroth, first, second


@pytest.mark.parametrize("scheme", ["wb", "standard_equilibrium"])
def test_stress_adds_cs2_rho_stress_to_the_second_moment_only(lattice, scheme):
    rho, u, stress = _fields()
    equilibrium = build_equilibrium_fn(scheme)

    plain = _moments(equilibrium(rho, u, lattice), lattice)
    stressed = _moments(equilibrium(rho, u, lattice, viscous_stress=stress), lattice)

    np.testing.assert_allclose(stressed[0], plain[0], atol=1e-6)
    np.testing.assert_allclose(stressed[1], plain[1], atol=1e-6)
    expected = np.asarray(CS2 * rho[..., None] * stress)[..., 0, :, :]
    np.testing.assert_allclose(stressed[2] - plain[2], expected, atol=1e-6)


def test_wb_equilibrium_moments_match_eq_c8(lattice):
    """``sum feq = rho``, ``sum c feq = rho u``, ``sum cc feq = rho uu + cs2 rho A S`` (p_g = 0)."""
    rho, u, stress = _fields(1)

    zeroth, first, second = _moments(build_equilibrium_fn("wb")(rho, u, lattice, viscous_stress=stress), lattice)

    rho_np, u_np, stress_np = np.asarray(rho), np.asarray(u), np.asarray(stress)
    np.testing.assert_allclose(zeroth, rho_np[..., 0, 0], atol=1e-6)
    np.testing.assert_allclose(first, (rho_np * u_np)[..., 0, :], atol=1e-6)
    uu = u_np[..., 0, :, None] * u_np[..., 0, None, :]
    expected = rho_np[..., 0, :, None] * (uu + CS2 * stress_np[..., 0, :, :])
    np.testing.assert_allclose(second, expected, atol=1e-6)


def test_no_stress_is_the_plain_equilibrium(lattice):
    rho, u, _ = _fields(2)
    equilibrium = build_equilibrium_fn("wb")

    np.testing.assert_array_equal(
        np.asarray(equilibrium(rho, u, lattice, viscous_stress=None)),
        np.asarray(equilibrium(rho, u, lattice)),
    )


def test_wb_source_moments_match_eq_c9(lattice):
    """The unchanged ``wb`` source carries no mass and exactly the force: ``sum F_i = 0``, ``sum c F_i = F``."""
    rho, u, _ = _fields(3)
    force = jnp.asarray(np.random.default_rng(3).uniform(-1e-3, 1e-3, (NX, NY, 1, 1, 2)))
    gradient = build_gradient(lattice, _PERIODIC)

    source = np.asarray(build_source_fn()(rho, u, force, lattice, gradient=gradient))

    np.testing.assert_allclose(source.sum(axis=-2), 0.0, atol=1e-8)
    np.testing.assert_allclose((source * np.asarray(lattice.c)).sum(axis=-2), np.asarray(force)[..., 0, :], atol=1e-8)


def _linear_velocity(g: np.ndarray) -> jnp.ndarray:
    x, y = np.meshgrid(np.arange(NX, dtype=float), np.arange(NY, dtype=float), indexing="ij")
    u = np.stack([g[0, 0] * x + g[0, 1] * y, g[1, 0] * x + g[1, 1] * y], axis=-1)
    return jnp.asarray(u[:, :, None, None, :])


def test_strain_rate_is_exact_on_a_linear_velocity_field(lattice):
    """``S = grad u + grad u^T`` with ``grad_u[a, b] = d u_a / d x_b``, away from the padded edges."""
    g = np.array([[0.010, -0.020], [0.030, 0.005]])
    gradient = build_gradient(lattice, _EDGE)

    strain = np.asarray(strain_rate(_linear_velocity(g), gradient))

    np.testing.assert_allclose(strain[1:-1, 1:-1, 0, 0], np.broadcast_to(g + g.T, (NX - 2, NY - 2, 2, 2)), atol=1e-6)


@pytest.mark.parametrize(
    ("lambda_v", "tau", "a_expected", "nu_expected"),
    [(1.0, 1.0, 0.0, 1.0 / 6.0), (1.0, 0.75, 0.25, 0.0833333)],
)
def test_a_reproduces_the_papers_viscosities(lattice, lambda_v, tau, a_expected, nu_expected):
    """Paper Sec. III: ``lambda_v = 1`` with ``A = 0`` gives 1/6 and ``A = 0.25`` gives 0.0833."""
    params = build_viscosity_params(relaxation_time=lambda_v, tau_liquid=tau, tau_gas=tau, rho_l=2.0, rho_v=0.5)
    g = np.array([[0.0, 0.01], [0.0, 0.0]])
    gradient = build_gradient(lattice, _EDGE)
    u = _linear_velocity(g)
    rho = jnp.full((NX, NY, 1, 1, 1), 1.3)

    stress = np.asarray(viscous_stress(rho, u, gradient, params))

    assert params.nu_l == pytest.approx(nu_expected, rel=1e-5)
    assert CS2 * (lambda_v - 0.5 - a_expected) == pytest.approx(params.nu_l, rel=1e-5)
    expected = np.broadcast_to(a_expected * (g + g.T), (NX - 2, NY - 2, 2, 2))
    np.testing.assert_allclose(stress[1:-1, 1:-1, 0, 0], expected, atol=1e-8)


def test_a_interpolates_the_viscosity_linearly_between_the_phases(lattice):
    """At ``rho_l`` / ``rho_v`` ``A`` is ``lambda_v - tau`` / ``lambda_v - tau_gas``; linear between."""
    params = build_viscosity_params(relaxation_time=1.0, tau_liquid=0.75, tau_gas=0.55, rho_l=6.0, rho_v=1.0)
    g = np.array([[0.0, 1.0], [0.0, 0.0]])  # S_xy = S_yx = 1
    gradient = build_gradient(lattice, _EDGE)
    u = _linear_velocity(g)

    for rho_value, a_expected in ((6.0, 0.25), (1.0, 0.45), (3.5, 0.35)):
        rho = jnp.full((NX, NY, 1, 1, 1), rho_value)
        stress = np.asarray(viscous_stress(rho, u, gradient, params))
        assert stress[3, 3, 0, 0, 0, 1] == pytest.approx(a_expected, rel=1e-5)
