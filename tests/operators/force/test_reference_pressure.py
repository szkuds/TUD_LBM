"""Buoyancy-referenced gravity: ``F_g = -(rho - rho_0) g`` plus ``p_g = rho_0 g.x`` (Zhang, Guo & Wang 2022).

The reference density moves the weight of ``rho_0`` out of the force and into
the equilibrium (``p_g I`` in the second moment) and the source term
(``-grad p_g`` in the velocity-force product). The total momentum source stays
``-rho g``, so a run with and without it must reach the same hydrostatic state.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.config import DictAdapter
from src.lattice.lattice import build_lattice
from src.operators.equilibrium import build_equilibrium_fn
from src.operators.source_term import build_source_fn

CS2 = 1.0 / 3.0
NX, NY = 6, 8
_G = 1e-4
_RHO_0 = 2.0


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


def _moments(f, lattice):
    f = np.asarray(f)
    c = np.asarray(lattice.c)
    return (
        f.sum(axis=-2)[..., 0],
        (f * c).sum(axis=-2),
        (f[..., None] * c[..., :, None] * c[..., None, :]).sum(axis=-3),
    )


def _config(gravity: dict[str, object], **overrides: object):
    base: dict[str, object] = {
        "sim_type": "multiphase",
        "grid_shape": (NX, NY),
        "eos": "double-well",
        "kappa": 0.01,
        "rho_l": _RHO_0,
        "rho_v": 0.5,
        "interface_width": 4,
        "gravity_force": gravity,
    }
    base.update(overrides)
    return DictAdapter().load(base)


def test_no_reference_density_means_no_reference_pressure():
    assert _config({"force_g": _G}).reference_pressure is None


@pytest.mark.parametrize("angle", [0.0, 30.0])
def test_reference_pressure_is_rho0_times_the_gravity_potential(angle):
    reference = _config({"force_g": _G, "reference_density": _RHO_0, "inclination_angle_deg": angle}).reference_pressure

    assert reference is not None
    g_vec = np.array([-_G * np.sin(np.radians(angle)), _G * np.cos(np.radians(angle))])
    gradient = np.asarray(reference.gradient)
    assert gradient.shape == (NX, NY, 1, 1, 2)
    np.testing.assert_allclose(gradient, np.broadcast_to(_RHO_0 * g_vec, gradient.shape), rtol=1e-12)
    field = np.asarray(reference.field)[:, :, 0, 0, 0]
    np.testing.assert_allclose(np.diff(field, axis=1), _RHO_0 * g_vec[1], rtol=1e-9, atol=1e-18)
    np.testing.assert_allclose(np.diff(field, axis=0), _RHO_0 * g_vec[0], rtol=1e-9, atol=1e-18)
    assert abs(field.mean()) < 1e-12


def test_gravity_force_is_only_the_excess_over_the_reference():
    from src.operators.force import build_forces
    from src.pipeline.state.state import State

    lattice = build_lattice("D2Q9")
    config = _config({"force_g": _G, "reference_density": _RHO_0})
    (gravity,) = build_forces(config, tuple(config.grid_shape), lattice)
    rho = jnp.linspace(0.5, 3.0, NX * NY).reshape(NX, NY, 1, 1, 1)
    f = lattice.w * rho
    state = State(f=f, rho=rho, u=jnp.zeros((NX, NY, 1, 1, 2)), t=jnp.array(0))

    force = np.asarray(gravity.compute(state))

    np.testing.assert_allclose(force[..., 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(force[..., 1], -(np.asarray(rho)[..., 0] - _RHO_0) * _G, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("scheme", ["wb", "standard_equilibrium"])
def test_equilibrium_adds_p_g_to_the_second_moment_only(lattice, scheme):
    rng = np.random.default_rng(0)
    rho = jnp.asarray(rng.uniform(0.5, 2.0, (NX, NY, 1, 1, 1)))
    u = jnp.asarray(rng.uniform(-0.05, 0.05, (NX, NY, 1, 1, 2)))
    p_g = jnp.asarray(rng.uniform(-0.01, 0.01, (NX, NY, 1, 1, 1)))
    equilibrium = build_equilibrium_fn(scheme)

    plain = _moments(equilibrium(rho, u, lattice), lattice)
    shifted = _moments(equilibrium(rho, u, lattice, reference_pressure=p_g), lattice)

    np.testing.assert_allclose(shifted[0], plain[0], atol=1e-7)
    np.testing.assert_allclose(shifted[1], plain[1], atol=1e-7)
    expected = np.asarray(p_g)[..., 0, 0, None, None] * np.eye(2)
    np.testing.assert_allclose(shifted[2] - plain[2], expected, atol=1e-7)


def test_source_subtracts_grad_p_g_from_the_velocity_force_product(lattice):
    """``sum F_i`` and ``sum c F_i`` are unchanged; ``sum cc F_i`` changes by ``-(u grad p_g + grad p_g u)``."""
    import src.pipeline  # noqa: F401 - enter through the pipeline (differential import cycle)
    from src.operators.differential import build_gradient_fn

    rng = np.random.default_rng(1)
    rho = jnp.asarray(rng.uniform(0.5, 2.0, (NX, NY, 1, 1, 1)))
    u = jnp.asarray(rng.uniform(-0.05, 0.05, (NX, NY, 1, 1, 2)))
    force = jnp.asarray(rng.uniform(-1e-3, 1e-3, (NX, NY, 1, 1, 2)))
    grad_p = jnp.broadcast_to(jnp.asarray([3e-4, -7e-4]), (NX, NY, 1, 1, 2))
    gradient = build_gradient_fn(lattice, ("wrap", "wrap", "wrap", "wrap"))
    source = build_source_fn()

    plain = _moments(source(rho, u, force, lattice, gradient=gradient), lattice)
    shifted = _moments(source(rho, u, force, lattice, gradient=gradient, reference_gradient=grad_p), lattice)

    np.testing.assert_allclose(shifted[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(shifted[1], plain[1], atol=1e-10)
    u_np, g_np = np.asarray(u)[..., 0, :], np.asarray(grad_p)[..., 0, :]
    expected = -(u_np[..., :, None] * g_np[..., None, :] + g_np[..., :, None] * u_np[..., None, :])
    np.testing.assert_allclose(shifted[2] - plain[2], expected, atol=1e-10)


def test_hydrostatic_state_does_not_depend_on_the_reference_density():
    """Walled liquid column: with or without ``rho_0`` the fluid settles to the same stratification.

    ``rho_0`` only moves ``rho_0 g`` between the force and the equilibrium, so
    the momentum source ``-rho g`` — and hence the rest state — is the same.
    """
    from src.pipeline.runner import init_state
    from src.pipeline.setup import build_setup

    a, b, t = 9.0 / 392.0, 2.0 / 21.0, 0.8 / 14.0
    rho_l, rho_v = 6.7645, 0.8388
    finals = []
    for reference in (None, rho_l):
        gravity: dict[str, object] = {"force_g": 1e-4}
        if reference is not None:
            gravity["reference_density"] = reference
        config = DictAdapter().load(
            {
                "sim_type": "multiphase",
                "grid_shape": (4, 40),
                "tau": 0.8,
                # MRT with unit free rates: under BGK the even-width column's
                # 2-cell density mode grows to NaN whatever the gravity.
                "collision_scheme": "mrt",
                "k_diag": (0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0),
                "eos": "van-der-waals",
                "a_eos": a,
                "b_eos": b,
                "r_eos": 1.0,
                "t_eos": t,
                "kappa": 0.01,
                "rho_l": rho_l,
                "rho_v": rho_v,
                "interface_width": 4,
                "init_type": "multiphase_bubbles",
                "initialisation": {"centres": [[0.5, 0.5]], "radii": [20.0], "dispersed": "liquid"},
                "bc_config": {"top": "bounce-back", "bottom": "bounce-back", "left": "periodic", "right": "periodic"},
                "gravity_force": gravity,
            }
        )
        setup = build_setup(config)
        state = init_state(setup)
        step_fn = setup.step_fn
        assert step_fn is not None
        body = jax.jit(
            lambda s, setup=setup, step_fn=step_fn: jax.lax.fori_loop(0, 20000, lambda _, s: step_fn(setup, s), s)
        )
        finals.append(body(state))

    plain, referenced = finals
    for final in finals:
        assert float(np.abs(np.asarray(final.u)).max()) < 1e-5
    rho_plain = np.asarray(plain.rho)[..., 0, 0, 0]
    rho_ref = np.asarray(referenced.rho)[..., 0, 0, 0]
    # Stratified (denser at the bottom) and the same in both runs.
    assert rho_plain[:, 5].mean() > rho_plain[:, -6].mean()
    np.testing.assert_allclose(rho_ref[:, 5:-5], rho_plain[:, 5:-5], rtol=1e-4)
