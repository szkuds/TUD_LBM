"""Tests for force operators — gravity and electric."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from setup.lattice import build_lattice
from state.state import State

NX, NY = 8, 8


@pytest.fixture
def lattice():
    return build_lattice("D2Q9")


def make_state(lattice, rho_value=1.0, h=None):
    f = jnp.ones((NX, NY, lattice.q, 1)) * (rho_value / lattice.q)
    rho = jnp.sum(f, axis=2, keepdims=True)
    return State(
        f=f,
        rho=rho,
        u=jnp.zeros((NX, NY, 1, lattice.d)),
        t=jnp.array(0),
        h=h,
    )


# =====================================================================
# Gravity force
# =====================================================================


class TestGravityForce:
    """GravityForceModule build/compute behaviour."""

    def test_template_shape(self, lattice):
        from operators.force._gravity import GravityForceModule

        template = GravityForceModule.build({"force_g": 0.001}, (NX, NY))
        assert template.shape == (NX, NY, 1, 2)

    def test_vertical_gravity(self, lattice):
        from operators.force._gravity import GravityForceModule

        template = GravityForceModule.build(
            {"force_g": 0.001, "inclination_angle_deg": 0.0},
            (NX, NY),
        )
        np.testing.assert_allclose(
            float(template[0, 0, 0, 0]),
            0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            float(template[0, 0, 0, 1]),
            0.001,
            atol=1e-10,
        )

    def test_inclined_gravity(self, lattice):
        from operators.force._gravity import GravityForceModule

        template = GravityForceModule.build(
            {"force_g": 0.001, "inclination_angle_deg": 90.0},
            (NX, NY),
        )
        np.testing.assert_allclose(
            float(template[0, 0, 0, 0]),
            -0.001,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            float(template[0, 0, 0, 1]),
            0.0,
            atol=1e-10,
        )

    def test_compute_gravity_force_shape(self, lattice):
        from operators.force._gravity import GravityForceModule

        template = GravityForceModule.build({"force_g": 0.001}, (NX, NY))
        state = make_state(lattice, rho_value=1.0)
        force = GravityForceModule.compute(state, template)
        assert force.shape == (NX, NY, 1, 2)

    def test_compute_gravity_force_value(self, lattice):
        from operators.force._gravity import GravityForceModule

        template = GravityForceModule.build({"force_g": 0.001}, (NX, NY))
        state = make_state(lattice, rho_value=2.0)
        force = GravityForceModule.compute(state, template)
        expected = -template * 2.0
        np.testing.assert_allclose(
            np.array(force),
            np.array(expected),
            atol=1e-12,
        )

    def test_jittable(self, lattice):
        from operators.force._gravity import GravityForceModule

        template = GravityForceModule.build({"force_g": 0.001}, (NX, NY))
        state = make_state(lattice, rho_value=1.0)
        force = jax.jit(lambda s: GravityForceModule.compute(s, template))(state)
        assert force.shape == (NX, NY, 1, 2)


# =====================================================================
# Electric params
# =====================================================================


class TestElectricParams:
    """ElectricForceModule.build creates a valid NamedTuple pytree."""

    def test_creation(self, lattice):
        from operators.force._electric import ElectricForceModule

        ep = ElectricForceModule.build({
            "permittivity_liquid": 80.0,
            "permittivity_vapour": 1.0,
            "conductivity_liquid": 0.01,
            "conductivity_vapour": 0.001,
        }, (NX, NY))
        assert ep.permittivity_liquid == 80.0
        assert ep.permittivity_vapour == 1.0

    def test_is_pytree(self, lattice):
        from operators.force._electric import ElectricForceModule

        ep = ElectricForceModule.build({
            "permittivity_liquid": 80.0,
            "permittivity_vapour": 1.0,
            "conductivity_liquid": 0.01,
            "conductivity_vapour": 0.001,
        }, (NX, NY))
        leaves, treedef = jax.tree_util.tree_flatten(ep)
        ep2 = treedef.unflatten(leaves)
        assert ep2.permittivity_liquid == ep.permittivity_liquid


# =====================================================================
# Electric init_hi
# =====================================================================


class TestInitState:
    """ElectricForceModule.init_state produces a valid initial distribution."""

    @staticmethod
    def _params(lattice):
        from operators.force._electric import ElectricForceModule

        return ElectricForceModule.build({
            "permittivity_liquid": 80.0,
            "permittivity_vapour": 1.0,
            "conductivity_liquid": 0.01,
            "conductivity_vapour": 0.001,
        }, (NX, NY))

    def test_shape(self, lattice):
        from operators.force._electric import ElectricForceModule

        params = self._params(lattice)
        hi = ElectricForceModule.init_state((NX, NY), lattice, params)["h"]
        assert hi.shape == (NX, NY, 9, 1)

    def test_linear_profile(self, lattice):
        from operators.force._electric import ElectricForceModule

        params = self._params(lattice)._replace(voltage_top=1.0, voltage_bottom=0.0)
        hi = ElectricForceModule.init_state((NX, NY), lattice, params)["h"]
        potential = jnp.sum(hi, axis=2, keepdims=True)
        np.testing.assert_allclose(
            float(potential[0, 0, 0, 0]),
            0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            float(potential[0, -1, 0, 0]),
            1.0,
            atol=1e-10,
        )


# =====================================================================
# Electric force
# =====================================================================


class TestComputeElectricForce:
    """ElectricForceModule.compute returns correct shape and is jittable."""

    def test_shape(self, lattice):
        from operators.force._electric import ElectricForceModule

        params = ElectricForceModule.build({
            "permittivity_liquid": 80.0,
            "permittivity_vapour": 1.0,
            "conductivity_liquid": 0.01,
            "conductivity_vapour": 0.001,
            "voltage_top": 1.0,
        }, (NX, NY))
        hi = ElectricForceModule.init_state((NX, NY), lattice, params)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        force = ElectricForceModule.compute(state, params)
        assert force.shape == (NX, NY, 1, 2)

    def test_zero_voltage_zero_force(self, lattice):
        from operators.force._electric import ElectricForceModule

        params = ElectricForceModule.build({
            "permittivity_liquid": 1.0,
            "permittivity_vapour": 1.0,
            "conductivity_liquid": 0.0,
            "conductivity_vapour": 0.0,
            "voltage_top": 0.0,
            "voltage_bottom": 0.0,
        }, (NX, NY))
        hi = ElectricForceModule.init_state((NX, NY), lattice, params)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        force = ElectricForceModule.compute(state, params)
        np.testing.assert_allclose(np.array(force), 0.0, atol=1e-10)

    def test_jittable(self, lattice):
        from operators.force._electric import ElectricForceModule

        params = ElectricForceModule.build({
            "permittivity_liquid": 80.0,
            "permittivity_vapour": 1.0,
            "conductivity_liquid": 0.01,
            "conductivity_vapour": 0.001,
            "voltage_top": 1.0,
        }, (NX, NY))
        hi = ElectricForceModule.init_state((NX, NY), lattice, params)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        jitted = jax.jit(lambda s: ElectricForceModule.compute(s, params))
        force = jitted(state)
        assert force.shape == (NX, NY, 1, 2)


# =====================================================================
# Update hi
# =====================================================================


class TestUpdateState:
    """ElectricForceModule.update_state advances the electric distribution."""

    def test_shape(self, lattice):
        from operators.force._electric import ElectricForceModule
        from operators.streaming._streaming import stream

        params = ElectricForceModule.build({
            "permittivity_liquid": 80.0,
            "permittivity_vapour": 1.0,
            "conductivity_liquid": 0.01,
            "conductivity_vapour": 0.001,
            "voltage_top": 1.0,
            "voltage_bottom": 0.0,
        }, (NX, NY))
        hi = ElectricForceModule.init_state((NX, NY), lattice, params)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        state_new = ElectricForceModule.update_state(state, params, lattice, stream)
        assert state_new.h.shape == hi.shape
