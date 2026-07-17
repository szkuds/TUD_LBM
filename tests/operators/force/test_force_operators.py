"""Tests for force operators — gravity and electric."""

from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tud_lbm.config.simulation_config import SimulationConfig
from tud_lbm.lattice.lattice import build_lattice
from tud_lbm.operators.force import ForceParams
from tud_lbm.operators.force import ForceSetup
from tud_lbm.pipeline.state.state import State

NX, NY, NZ = 8, 8, 1


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


@pytest.fixture(scope="module")
def sim_config():
    """Minimal all-periodic config for electric force tests."""
    return SimulationConfig(grid_shape=(NX, NY, NZ))


@pytest.fixture(scope="module")
def electric_params(lattice, sim_config):
    """Pre-built ElectricParams with a real gradient closure."""
    from tud_lbm.operators.force._electric import ElectricForceModule

    return ElectricForceModule.build(
        {
            "permittivity_liquid": 80.0,
            "permittivity_vapour": 1.0,
            "conductivity_liquid": 0.01,
            "conductivity_vapour": 0.001,
            "voltage_top": 1.0,
            "voltage_bottom": 0.0,
        },
        (NX, NY, NZ),
        config=sim_config,
        lattice=lattice,
    )


def make_state(lattice, rho_value=1.0, h=None):
    f = jnp.ones((NX, NY, NZ, lattice.q, 1)) * (rho_value / lattice.q)
    rho = jnp.sum(f, axis=3, keepdims=True)
    return State(
        f=f,
        rho=rho,
        u=jnp.zeros((NX, NY, NZ, 1, lattice.d)),
        t=jnp.array(0),
        h=h,
    )


def make_electric_setup(lattice, electric_params):
    from tud_lbm.operators.streaming._streaming import stream

    specs = (
        ForceParams(
            name="electric_force",
            compute_fn=None,
            precomputed=electric_params,
        ),
    )
    return SimpleNamespace(
        grid_shape=(NX, NY, NZ),
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
        from tud_lbm.operators.force._gravity import GravityForceModule

        template = GravityForceModule.build({"force_g": 0.001}, (NX, NY, NZ), config=sim_config, lattice=lattice)
        assert template.shape == (NX, NY, NZ, 1, 2)

    def test_vertical_gravity(self, lattice, sim_config):
        from tud_lbm.operators.force._gravity import GravityForceModule

        template = GravityForceModule.build(
            {"force_g": 0.001, "inclination_angle_deg": 0.0},
            (NX, NY, NZ),
            config=sim_config,
            lattice=lattice,
        )
        np.testing.assert_allclose(
            float(template[0, 0, 0, 0, 0]),
            0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            float(template[0, 0, 0, 0, 1]),
            0.001,
            atol=1e-10,
        )

    def test_inclined_gravity(self, lattice, sim_config):
        from tud_lbm.operators.force._gravity import GravityForceModule

        template = GravityForceModule.build(
            {"force_g": 0.001, "inclination_angle_deg": 90.0},
            (NX, NY, NZ),
            config=sim_config,
            lattice=lattice,
        )
        np.testing.assert_allclose(
            float(template[0, 0, 0, 0, 0]),
            -0.001,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            float(template[0, 0, 0, 0, 1]),
            0.0,
            atol=1e-10,
        )

    def test_compute_gravity_force_shape(self, lattice, sim_config):
        from tud_lbm.operators.force._gravity import GravityForceModule

        template = GravityForceModule.build({"force_g": 0.001}, (NX, NY, NZ), config=sim_config, lattice=lattice)
        state = make_state(lattice, rho_value=1.0)
        force = GravityForceModule.compute(state, template)
        assert force.shape == (NX, NY, NZ, 1, 2)

    def test_compute_gravity_force_value(self, lattice, sim_config):
        from tud_lbm.operators.force._gravity import GravityForceModule

        template = GravityForceModule.build({"force_g": 0.001}, (NX, NY, NZ), config=sim_config, lattice=lattice)
        state = make_state(lattice, rho_value=2.0)
        force = GravityForceModule.compute(state, template)
        expected = -template * 2.0
        np.testing.assert_allclose(
            np.array(force),
            np.array(expected),
            atol=1e-12,
        )

    def test_jittable(self, lattice, sim_config):
        from tud_lbm.operators.force._gravity import GravityForceModule

        template = GravityForceModule.build({"force_g": 0.001}, (NX, NY, NZ), config=sim_config, lattice=lattice)
        state = make_state(lattice, rho_value=1.0)
        force = jax.jit(lambda s: GravityForceModule.compute(s, template))(state)
        assert force.shape == (NX, NY, NZ, 1, 2)


class TestGravityMaskedForce:
    """GravityForceModule with phase mask based on rho_v/rho_l references."""

    def test_compute_matches_masked_formula(self, lattice):
        from tud_lbm.operators.force._gravity_masked import GravityForceModule

        cfg = SimpleNamespace(rho_l=1.0, rho_v=0.5)
        precomputed = GravityForceModule.build({"force_g": 0.001}, (NX, NY, NZ), config=cfg, lattice=lattice)
        state = make_state(lattice, rho_value=0.75)
        force = GravityForceModule.compute(state, precomputed)

        rho = jnp.sum(state.f, axis=-2, keepdims=True)
        mask = jnp.clip((rho - cfg.rho_v) / (cfg.rho_l - cfg.rho_v), 0.0, 1.0)
        expected = -precomputed.template * rho * mask
        np.testing.assert_allclose(np.array(force), np.array(expected), atol=1e-12)

    def test_compute_without_phase_refs_matches_single_phase(self, lattice):
        from tud_lbm.operators.force._gravity_masked import GravityForceModule

        precomputed = GravityForceModule.build({"force_g": 0.001}, (NX, NY, NZ), config=None, lattice=lattice)
        state = make_state(lattice, rho_value=1.2)
        force = GravityForceModule.compute(state, precomputed)
        expected = -precomputed.template * 1.2
        np.testing.assert_allclose(np.array(force), np.array(expected), atol=1e-12)


# =====================================================================
# Electric params
# =====================================================================


class TestElectricParams:
    """ElectricForceModule.build creates a valid NamedTuple pytree."""

    def test_creation(self, lattice, sim_config):
        from tud_lbm.operators.force._electric import ElectricForceModule

        ep = ElectricForceModule.build(
            {
                "permittivity_liquid": 80.0,
                "permittivity_vapour": 1.0,
                "conductivity_liquid": 0.01,
                "conductivity_vapour": 0.001,
            },
            (NX, NY, NZ),
            config=sim_config,
            lattice=lattice,
        )
        assert ep.permittivity_liquid == 80.0
        assert ep.permittivity_vapour == 1.0

    def test_is_pytree(self, lattice, sim_config):
        from tud_lbm.operators.force._electric import ElectricForceModule

        ep = ElectricForceModule.build(
            {
                "permittivity_liquid": 80.0,
                "permittivity_vapour": 1.0,
                "conductivity_liquid": 0.01,
                "conductivity_vapour": 0.001,
            },
            (NX, NY, NZ),
            config=sim_config,
            lattice=lattice,
        )
        leaves, treedef = jax.tree_util.tree_flatten(ep)
        ep2 = treedef.unflatten(leaves)
        assert ep2.permittivity_liquid == ep.permittivity_liquid

    def test_legacy_state_hooks_removed(self):
        from tud_lbm.operators.force._electric import ElectricForceModule

        assert not hasattr(ElectricForceModule, "init_state")
        assert not hasattr(ElectricForceModule, "update_state")


# =====================================================================
# Electric init_hi
# =====================================================================


class TestElectricExtraStateInit:
    """ElectricExtraStatePlugin.init_state produces a valid initial distribution."""

    def test_shape(self, lattice, electric_params):
        from tud_lbm.operators.force._extra_state import ElectricExtraStatePlugin

        setup = make_electric_setup(lattice, electric_params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        assert hi.shape == (NX, NY, NZ, 9, 1)

    def test_linear_profile(self, lattice, sim_config):
        from tud_lbm.operators.force._electric import ElectricForceModule
        from tud_lbm.operators.force._extra_state import ElectricExtraStatePlugin

        params = ElectricForceModule.build(
            {
                "permittivity_liquid": 80.0,
                "permittivity_vapour": 1.0,
                "conductivity_liquid": 0.01,
                "conductivity_vapour": 0.001,
                "voltage_top": 1.0,
                "voltage_bottom": 0.0,
            },
            (NX, NY, NZ),
            config=sim_config,
            lattice=lattice,
        )
        setup = make_electric_setup(lattice, params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        potential = jnp.sum(hi, axis=3, keepdims=True)
        np.testing.assert_allclose(
            float(potential[0, 0, 0, 0, 0]),
            0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            float(potential[0, -1, 0, 0, 0]),
            1.0,
            atol=1e-10,
        )


# =====================================================================
# Electric force
# =====================================================================


class TestComputeElectricForce:
    """ElectricForceModule.compute returns correct shape and is jittable."""

    def test_shape(self, lattice, sim_config, electric_params):
        from tud_lbm.operators.differential import build_diff_ops
        from tud_lbm.operators.force._electric import ElectricForceModule
        from tud_lbm.operators.force._extra_state import ElectricExtraStatePlugin

        gradient_standard, *_ = build_diff_ops(sim_config, mp_params=None, lattice=lattice)
        setup = make_electric_setup(lattice, electric_params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        force = ElectricForceModule.compute(state, electric_params, gradient_standard=gradient_standard)
        assert force.shape == (NX, NY, NZ, 1, 2)

    def test_zero_voltage_zero_force(self, lattice, sim_config):
        from tud_lbm.operators.differential import build_diff_ops
        from tud_lbm.operators.force._electric import ElectricForceModule
        from tud_lbm.operators.force._extra_state import ElectricExtraStatePlugin

        params = ElectricForceModule.build(
            {
                "permittivity_liquid": 1.0,
                "permittivity_vapour": 1.0,
                "conductivity_liquid": 0.0,
                "conductivity_vapour": 0.0,
                "voltage_top": 0.0,
                "voltage_bottom": 0.0,
            },
            (NX, NY, NZ),
            config=sim_config,
            lattice=lattice,
        )
        gradient_standard, *_ = build_diff_ops(sim_config, mp_params=None, lattice=lattice)
        setup = make_electric_setup(lattice, params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        force = ElectricForceModule.compute(state, params, gradient_standard=gradient_standard)
        np.testing.assert_allclose(np.array(force), 0.0, atol=1e-10)

    def test_jittable(self, lattice, sim_config, electric_params):
        from tud_lbm.operators.differential import build_diff_ops
        from tud_lbm.operators.force._electric import ElectricForceModule
        from tud_lbm.operators.force._extra_state import ElectricExtraStatePlugin

        gradient_standard, *_ = build_diff_ops(sim_config, mp_params=None, lattice=lattice)
        setup = make_electric_setup(lattice, electric_params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        jitted = jax.jit(lambda s: ElectricForceModule.compute(s, electric_params, gradient_standard=gradient_standard))
        force = jitted(state)
        assert force.shape == (NX, NY, NZ, 1, 2)


# =====================================================================
# Update hi
# =====================================================================


class TestElectricExtraStateUpdate:
    """ElectricExtraStatePlugin.update_state advances the electric distribution."""

    def test_shape(self, lattice, electric_params):
        from tud_lbm.operators.force._extra_state import ElectricExtraStatePlugin

        setup = make_electric_setup(lattice, electric_params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        state_new = ElectricExtraStatePlugin.update_state(setup, state, state)
        assert state_new.h is not None
        assert state_new.h.shape == hi.shape

    def test_update_returns_new_state_when_h_is_none(self, lattice, electric_params):
        from tud_lbm.operators.force._extra_state import ElectricExtraStatePlugin

        setup = make_electric_setup(lattice, electric_params)
        state_no_h = make_state(lattice, rho_value=1.0, h=None)
        result = ElectricExtraStatePlugin.update_state(setup, state_no_h, state_no_h)
        assert result is state_no_h

    def test_update_returns_new_state_when_no_electric_force_spec(self, lattice):
        from tud_lbm.operators.force._extra_state import ElectricExtraStatePlugin

        setup = SimpleNamespace(forces=None)
        state = make_state(lattice)
        result = ElectricExtraStatePlugin.update_state(setup, state, state)  # ty: ignore[invalid-argument-type]
        assert result is state

    def test_update_raises_when_streaming_fn_none(self, lattice, electric_params):
        from tud_lbm.operators.force import ForceParams
        from tud_lbm.operators.force import ForceSetup
        from tud_lbm.operators.force._extra_state import ElectricExtraStatePlugin

        hi = jnp.ones((NX, NY, NZ, 9, 1))
        state = make_state(lattice, rho_value=1.0, h=hi)
        specs = (ForceParams(name="electric_force", compute_fn=None, precomputed=electric_params),)
        setup = SimpleNamespace(
            grid_shape=(NX, NY, NZ),
            lattice=lattice,
            streaming_fn=None,
            forces=ForceSetup(specs=specs, source_term=lambda *_a, **_k: None),
        )
        with pytest.raises(TypeError, match="streaming_fn is required"):
            ElectricExtraStatePlugin.update_state(setup, state, state)  # ty: ignore[invalid-argument-type]


# =====================================================================
# Electric is_active + compute error branches
# =====================================================================


class TestElectricIsActive:
    """ElectricExtraStatePlugin.is_active reflects config.electric_force."""

    def test_active_when_electric_force_set(self):
        from tud_lbm.operators.force._extra_state import ElectricExtraStatePlugin

        cfg = SimpleNamespace(electric_force={"strength": 1.0})
        assert ElectricExtraStatePlugin.is_active(cfg) is True  # ty: ignore[invalid-argument-type]

    def test_inactive_when_electric_force_none(self):
        from tud_lbm.operators.force._extra_state import ElectricExtraStatePlugin

        assert ElectricExtraStatePlugin.is_active(SimpleNamespace(electric_force=None)) is False  # ty: ignore[invalid-argument-type]

    def test_inactive_when_attribute_absent(self):
        from tud_lbm.operators.force._extra_state import ElectricExtraStatePlugin

        assert ElectricExtraStatePlugin.is_active(SimpleNamespace()) is False  # ty: ignore[invalid-argument-type]


class TestElectricForceSetupWiring:
    """build_setup wires gradient_standard for electric-force runs.

    Regression guard: the electric force needs the standard gradient closure
    from build_diff_ops, which requires the differential operator subpackage
    to have been auto-loaded before setup completes.  If import order ever
    regresses, gradient_standard would be missing and every electric compute
    would raise at step time.
    """

    @pytest.fixture(scope="class")
    def electric_setup(self):
        from tud_lbm.config.adapter_dict import DictAdapter
        from tud_lbm.pipeline.setup import build_setup

        config = DictAdapter().load(
            {
                "grid_shape": (NX, NY),
                "nt": 10,
                "electric_force": {
                    "permittivity_liquid": 80.0,
                    "permittivity_vapour": 1.0,
                    "conductivity_liquid": 0.01,
                    "conductivity_vapour": 0.001,
                    "voltage_top": 1.0,
                    "voltage_bottom": 0.0,
                },
            }
        )
        return build_setup(config)

    def test_gradient_standard_is_built(self, electric_setup):
        assert electric_setup.gradient_standard is not None
        assert callable(electric_setup.gradient_standard)

    def test_electric_force_spec_registered(self, electric_setup):
        assert [spec.name for spec in electric_setup.forces.specs] == ["electric_force"]

    def test_total_force_computes_from_setup(self, electric_setup):
        from tud_lbm.operators.force import compute_total_force_ext
        from tud_lbm.pipeline.runner import init_state

        state = init_state(electric_setup)
        total_force, _ = compute_total_force_ext(electric_setup, state, electric_setup.forces)
        assert total_force is not None
        assert total_force.shape == (NX, NY, NZ, 1, 2)
        assert bool(jnp.all(jnp.isfinite(total_force)))


class TestElectricForceComputeErrors:
    """ElectricForceModule.compute raises TypeError on missing prerequisites."""

    def test_raises_without_gradient_standard(self, lattice, sim_config):
        from tud_lbm.operators.force._electric import ElectricForceModule

        params = ElectricForceModule.build(
            {
                "permittivity_liquid": 80.0,
                "permittivity_vapour": 1.0,
                "conductivity_liquid": 0.01,
                "conductivity_vapour": 0.001,
            },
            (NX, NY, NZ),
            config=sim_config,
            lattice=lattice,
        )
        state = make_state(lattice)
        with pytest.raises(TypeError, match="gradient_standard is required"):
            ElectricForceModule.compute(state, params)

    def test_raises_when_h_is_none(self, lattice, sim_config, electric_params):
        from tud_lbm.operators.differential import build_diff_ops
        from tud_lbm.operators.force._electric import ElectricForceModule

        gradient_standard, *_ = build_diff_ops(sim_config, mp_params=None, lattice=lattice)
        state = make_state(lattice, h=None)
        with pytest.raises(TypeError, match=r"state\.h"):
            ElectricForceModule.compute(state, electric_params, gradient_standard=gradient_standard)
