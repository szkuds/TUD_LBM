"""Tests for force operators — gravity and electric."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from config.simulation_config import SimulationConfig
from operators.force import ForceParams, ForceSetup
from setup.lattice import build_lattice
from state.state import State

NX, NY = 8, 8


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


@pytest.fixture(scope="module")
def sim_config():
    """Minimal all-periodic config for electric force tests."""
    return SimulationConfig(grid_shape=(NX, NY))


@pytest.fixture(scope="module")
def electric_params(lattice, sim_config):
    """Pre-built ElectricParams with a real gradient closure."""
    from operators.force._electric import ElectricForceModule

    return ElectricForceModule.build(
        {
            "permittivity_liquid": 80.0,
            "permittivity_vapour": 1.0,
            "conductivity_liquid": 0.01,
            "conductivity_vapour": 0.001,
            "voltage_top": 1.0,
            "voltage_bottom": 0.0,
        },
        (NX, NY),
        config=sim_config,
        lattice=lattice,
    )


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


def make_electric_setup(lattice, electric_params):
    from operators.streaming._streaming import stream

    specs = (
        ForceParams(
            name="electric_force",
            compute_fn=None,
            precomputed=electric_params,
        ),
    )
    return SimpleNamespace(
        grid_shape=(NX, NY),
        lattice=lattice,
        streaming_fn=stream,
        forces=ForceSetup(specs=specs, source_term=lambda *args, **kwargs: None),
    )


# =====================================================================
# Gravity force
# =====================================================================


class TestGravityForce:
    """GravityForceModule build/compute behaviour."""

    def test_template_shape(self, lattice, sim_config):
        from operators.force._gravity import GravityForceModule

        template = GravityForceModule.build(
            {"force_g": 0.001}, (NX, NY), config=sim_config, lattice=lattice
        )
        assert template.shape == (NX, NY, 1, 2)

    def test_vertical_gravity(self, lattice, sim_config):
        from operators.force._gravity import GravityForceModule

        template = GravityForceModule.build(
            {"force_g": 0.001, "inclination_angle_deg": 0.0},
            (NX, NY),
            config=sim_config,
            lattice=lattice,
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

    def test_inclined_gravity(self, lattice, sim_config):
        from operators.force._gravity import GravityForceModule

        template = GravityForceModule.build(
            {"force_g": 0.001, "inclination_angle_deg": 90.0},
            (NX, NY),
            config=sim_config,
            lattice=lattice,
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

    def test_compute_gravity_force_shape(self, lattice, sim_config):
        from operators.force._gravity import GravityForceModule

        template = GravityForceModule.build(
            {"force_g": 0.001}, (NX, NY), config=sim_config, lattice=lattice
        )
        state = make_state(lattice, rho_value=1.0)
        force = GravityForceModule.compute(state, template)
        assert force.shape == (NX, NY, 1, 2)

    def test_compute_gravity_force_value(self, lattice, sim_config):
        from operators.force._gravity import GravityForceModule

        template = GravityForceModule.build(
            {"force_g": 0.001}, (NX, NY), config=sim_config, lattice=lattice
        )
        state = make_state(lattice, rho_value=2.0)
        force = GravityForceModule.compute(state, template)
        expected = -template * 2.0
        np.testing.assert_allclose(
            np.array(force),
            np.array(expected),
            atol=1e-12,
        )

    def test_jittable(self, lattice, sim_config):
        from operators.force._gravity import GravityForceModule

        template = GravityForceModule.build(
            {"force_g": 0.001}, (NX, NY), config=sim_config, lattice=lattice
        )
        state = make_state(lattice, rho_value=1.0)
        force = jax.jit(lambda s: GravityForceModule.compute(s, template))(state)
        assert force.shape == (NX, NY, 1, 2)


# =====================================================================
# Electric params
# =====================================================================


class TestElectricParams:
    """ElectricForceModule.build creates a valid NamedTuple pytree."""

    def test_creation(self, lattice, sim_config):
        from operators.force._electric import ElectricForceModule

        ep = ElectricForceModule.build(
            {
                "permittivity_liquid": 80.0,
                "permittivity_vapour": 1.0,
                "conductivity_liquid": 0.01,
                "conductivity_vapour": 0.001,
            },
            (NX, NY),
            config=sim_config,
            lattice=lattice,
        )
        assert ep.permittivity_liquid == 80.0
        assert ep.permittivity_vapour == 1.0

    def test_is_pytree(self, lattice, sim_config):
        from operators.force._electric import ElectricForceModule

        ep = ElectricForceModule.build(
            {
                "permittivity_liquid": 80.0,
                "permittivity_vapour": 1.0,
                "conductivity_liquid": 0.01,
                "conductivity_vapour": 0.001,
            },
            (NX, NY),
            config=sim_config,
            lattice=lattice,
        )
        leaves, treedef = jax.tree_util.tree_flatten(ep)
        ep2 = treedef.unflatten(leaves)
        assert ep2.permittivity_liquid == ep.permittivity_liquid

    def test_legacy_state_hooks_removed(self):
        from operators.force._electric import ElectricForceModule

        assert not hasattr(ElectricForceModule, "init_state")
        assert not hasattr(ElectricForceModule, "update_state")


# =====================================================================
# Electric init_hi
# =====================================================================


class TestElectricExtraStateInit:
    """ElectricExtraStatePlugin.init_state produces a valid initial distribution."""

    def test_shape(self, lattice, electric_params):
        from operators.force._extra_state import ElectricExtraStatePlugin

        setup = make_electric_setup(lattice, electric_params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        assert hi.shape == (NX, NY, 9, 1)

    def test_linear_profile(self, lattice, sim_config):
        from operators.force._electric import ElectricForceModule
        from operators.force._extra_state import ElectricExtraStatePlugin

        params = ElectricForceModule.build(
            {
                "permittivity_liquid": 80.0,
                "permittivity_vapour": 1.0,
                "conductivity_liquid": 0.01,
                "conductivity_vapour": 0.001,
                "voltage_top": 1.0,
                "voltage_bottom": 0.0,
            },
            (NX, NY),
            config=sim_config,
            lattice=lattice,
        )
        setup = make_electric_setup(lattice, params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
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

    def test_shape(self, lattice, electric_params):
        from operators.force._electric import ElectricForceModule
        from operators.force._extra_state import ElectricExtraStatePlugin

        setup = make_electric_setup(lattice, electric_params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        force = ElectricForceModule.compute(state, electric_params)
        assert force.shape == (NX, NY, 1, 2)

    def test_zero_voltage_zero_force(self, lattice, sim_config):
        from operators.force._electric import ElectricForceModule
        from operators.force._extra_state import ElectricExtraStatePlugin

        params = ElectricForceModule.build(
            {
                "permittivity_liquid": 1.0,
                "permittivity_vapour": 1.0,
                "conductivity_liquid": 0.0,
                "conductivity_vapour": 0.0,
                "voltage_top": 0.0,
                "voltage_bottom": 0.0,
            },
            (NX, NY),
            config=sim_config,
            lattice=lattice,
        )
        setup = make_electric_setup(lattice, params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        force = ElectricForceModule.compute(state, params)
        np.testing.assert_allclose(np.array(force), 0.0, atol=1e-10)

    def test_jittable(self, lattice, electric_params):
        from operators.force._electric import ElectricForceModule
        from operators.force._extra_state import ElectricExtraStatePlugin

        setup = make_electric_setup(lattice, electric_params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        jitted = jax.jit(lambda s: ElectricForceModule.compute(s, electric_params))
        force = jitted(state)
        assert force.shape == (NX, NY, 1, 2)


# =====================================================================
# Update hi
# =====================================================================


class TestElectricExtraStateUpdate:
    """ElectricExtraStatePlugin.update_state advances the electric distribution."""

    def test_shape(self, lattice, electric_params):
        from operators.force._extra_state import ElectricExtraStatePlugin

        setup = make_electric_setup(lattice, electric_params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        state_new = ElectricExtraStatePlugin.update_state(setup, state, state)
        assert state_new.h.shape == hi.shape
