"""Tests for force operators — gravity and electric."""

from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.config.simulation_config import SimulationConfig
from src.lattice.lattice import build_lattice
from src.operators.force import ForceParams
from src.operators.force import ForceSetup
from src.pipeline.state.state import State

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
    from src.operators.force._electric import ElectricForceModule

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
    from src.operators.streaming._streaming import stream

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
        from src.operators.force._gravity import GravityForceModule

        template = GravityForceModule.build({"force_g": 0.001}, (NX, NY, NZ), config=sim_config, lattice=lattice)
        assert template.shape == (NX, NY, NZ, 1, 2)

    def test_vertical_gravity(self, lattice, sim_config):
        from src.operators.force._gravity import GravityForceModule

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
        from src.operators.force._gravity import GravityForceModule

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
        from src.operators.force._gravity import GravityForceModule

        template = GravityForceModule.build({"force_g": 0.001}, (NX, NY, NZ), config=sim_config, lattice=lattice)
        state = make_state(lattice, rho_value=1.0)
        force = GravityForceModule.compute(state, template)
        assert force.shape == (NX, NY, NZ, 1, 2)

    def test_compute_gravity_force_value(self, lattice, sim_config):
        from src.operators.force._gravity import GravityForceModule

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
        from src.operators.force._gravity import GravityForceModule

        template = GravityForceModule.build({"force_g": 0.001}, (NX, NY, NZ), config=sim_config, lattice=lattice)
        state = make_state(lattice, rho_value=1.0)
        force = jax.jit(lambda s: GravityForceModule.compute(s, template))(state)
        assert force.shape == (NX, NY, NZ, 1, 2)


class TestGravityMaskedForce:
    """GravityForceModule with phase mask based on rho_v/rho_l references."""

    def test_compute_matches_masked_formula(self, lattice):
        from src.operators.force._gravity_masked import GravityForceModule

        cfg = SimpleNamespace(rho_l=1.0, rho_v=0.5)
        precomputed = GravityForceModule.build({"force_g": 0.001}, (NX, NY, NZ), config=cfg, lattice=lattice)
        state = make_state(lattice, rho_value=0.75)
        force = GravityForceModule.compute(state, precomputed)

        rho = jnp.sum(state.f, axis=-2, keepdims=True)
        mask = jnp.clip((rho - cfg.rho_v) / (cfg.rho_l - cfg.rho_v), 0.0, 1.0)
        expected = -precomputed.template * rho * mask
        np.testing.assert_allclose(np.array(force), np.array(expected), atol=1e-12)

    def test_compute_without_phase_refs_matches_single_phase(self, lattice):
        from src.operators.force._gravity_masked import GravityForceModule

        precomputed = GravityForceModule.build({"force_g": 0.001}, (NX, NY, NZ), config=None, lattice=lattice)
        state = make_state(lattice, rho_value=1.2)
        force = GravityForceModule.compute(state, precomputed)
        expected = -precomputed.template * 1.2
        np.testing.assert_allclose(np.array(force), np.array(expected), atol=1e-12)


class TestGravityMaskedDispersedPhase:
    """The masked body force must drive the dispersed phase, bubble or droplet.

    A droplet run (liquid dispersed in vapour) keeps the historical liquid mask;
    a bubble run (vapour dispersed in liquid) takes its complement. Anything
    else spends the force on the continuous phase, which in a periodic domain
    accelerates the whole ambient body instead of the inclusion.
    """

    RHO_L = 1.0
    RHO_V = 0.33

    def _build(self, lattice, dispersed=None, *, initialisation=None, init_type=None):
        from src.operators.force._gravity_masked import GravityForceModule

        cfg = SimpleNamespace(
            rho_l=self.RHO_L,
            rho_v=self.RHO_V,
            initialisation=initialisation,
            init_type=init_type,
        )
        params = {"force_g": 0.001}
        if dispersed is not None:
            params["dispersed"] = dispersed
        return GravityForceModule.build(params, (NX, NY, NZ), config=cfg, lattice=lattice)

    def _two_phase_state(self, lattice):
        """Liquid in the left third, vapour in the right third, interface between."""
        rho_2d = np.full((NX, NY), self.RHO_V)
        rho_2d[: NX // 3, :] = self.RHO_L
        rho_2d[NX // 3 : 2 * NX // 3, :] = 0.5 * (self.RHO_L + self.RHO_V)
        rho = jnp.asarray(rho_2d.reshape(NX, NY, NZ, 1, 1))
        f = jnp.broadcast_to(rho / lattice.q, (NX, NY, NZ, lattice.q, 1))
        return State(
            f=f,
            rho=rho,
            u=jnp.zeros((NX, NY, NZ, 1, lattice.d)),
            t=jnp.array(0),
        )

    def test_bubble_drives_the_vapour_not_the_liquid(self, lattice):
        from src.operators.force._gravity_masked import GravityForceModule

        precomputed = self._build(lattice, "vapour")
        state = self._two_phase_state(lattice)
        force = np.array(GravityForceModule.compute(state, precomputed))

        # Bulk liquid gets nothing; bulk vapour carries the full -rho*g.
        np.testing.assert_allclose(force[: NX // 3], 0.0, atol=1e-12)
        expected_vapour = -np.array(precomputed.template)[2 * NX // 3 :] * self.RHO_V
        np.testing.assert_allclose(force[2 * NX // 3 :], expected_vapour, atol=1e-12)

    def test_droplet_drives_the_liquid_not_the_vapour(self, lattice):
        from src.operators.force._gravity_masked import GravityForceModule

        precomputed = self._build(lattice, "liquid")
        state = self._two_phase_state(lattice)
        force = np.array(GravityForceModule.compute(state, precomputed))

        np.testing.assert_allclose(force[2 * NX // 3 :], 0.0, atol=1e-12)
        expected_liquid = -np.array(precomputed.template)[: NX // 3] * self.RHO_L
        np.testing.assert_allclose(force[: NX // 3], expected_liquid, atol=1e-12)

    def test_the_two_masks_are_complementary(self, lattice):
        """Droplet plus bubble force recovers the unmasked ``-rho*g`` everywhere."""
        from src.operators.force._gravity_masked import GravityForceModule

        state = self._two_phase_state(lattice)
        liquid_force = GravityForceModule.compute(state, self._build(lattice, "liquid"))
        vapour_force = GravityForceModule.compute(state, self._build(lattice, "vapour"))

        template = self._build(lattice, "liquid").template
        rho = jnp.sum(state.f, axis=-2, keepdims=True)
        np.testing.assert_allclose(
            np.array(liquid_force + vapour_force),
            np.array(-template * rho),
            atol=1e-12,
        )

    def test_topology_comes_from_the_initialisation_section(self, lattice):
        precomputed = self._build(
            lattice,
            initialisation={"dispersed": "vapour"},
            init_type="multiphase_bubbles",
        )
        assert precomputed.dispersed == "vapour"

    def test_force_section_key_overrides_the_initialisation_section(self, lattice):
        precomputed = self._build(
            lattice,
            "liquid",
            initialisation={"dispersed": "vapour"},
            init_type="multiphase_bubbles",
        )
        assert precomputed.dispersed == "liquid"

    @pytest.mark.parametrize("init_type", ["multiphase_bubbles", "multiphase_bubble_top"])
    def test_bubble_initialisers_infer_vapour_without_an_explicit_key(self, lattice, init_type):
        """These initialisers place a vapour inclusion unless told otherwise."""
        precomputed = self._build(lattice, initialisation={}, init_type=init_type)
        assert precomputed.dispersed == "vapour"

    @pytest.mark.parametrize("init_type", ["standard", "wetting", None])
    def test_unknown_topology_falls_back_to_liquid(self, lattice, init_type):
        """Pre-bubble behaviour: droplet and single-phase runs drive the liquid."""
        precomputed = self._build(lattice, initialisation=None, init_type=init_type)
        assert precomputed.dispersed == "liquid"

    def test_invalid_dispersed_raises(self, lattice):
        with pytest.raises(ValueError, match="dispersed"):
            self._build(lattice, "solid")

    def test_jittable(self, lattice):
        from src.operators.force._gravity_masked import GravityForceModule

        precomputed = self._build(lattice, "vapour")
        state = self._two_phase_state(lattice)
        force = jax.jit(lambda s: GravityForceModule.compute(s, precomputed))(state)
        assert force.shape == (NX, NY, NZ, 1, 2)


# =====================================================================
# Electric params
# =====================================================================


class TestElectricParams:
    """ElectricForceModule.build creates a valid NamedTuple pytree."""

    def test_creation(self, lattice, sim_config):
        from src.operators.force._electric import ElectricForceModule

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
        from src.operators.force._electric import ElectricForceModule

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
        from src.operators.force._electric import ElectricForceModule

        assert not hasattr(ElectricForceModule, "init_state")
        assert not hasattr(ElectricForceModule, "update_state")


# =====================================================================
# Electric init_hi
# =====================================================================


class TestElectricExtraStateInit:
    """ElectricExtraStatePlugin.init_state produces a valid initial distribution."""

    def test_shape(self, lattice, electric_params):
        from src.operators.force._extra_state import ElectricExtraStatePlugin

        setup = make_electric_setup(lattice, electric_params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        assert hi.shape == (NX, NY, NZ, 9, 1)

    def test_linear_profile(self, lattice, sim_config):
        from src.operators.force._electric import ElectricForceModule
        from src.operators.force._extra_state import ElectricExtraStatePlugin

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
        from src.operators.differential import build_diff_ops
        from src.operators.force._electric import ElectricForceModule
        from src.operators.force._extra_state import ElectricExtraStatePlugin

        gradient_standard, *_ = build_diff_ops(sim_config, mp_params=None, lattice=lattice)
        setup = make_electric_setup(lattice, electric_params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        force = ElectricForceModule.compute(state, electric_params, gradient_standard=gradient_standard)
        assert force.shape == (NX, NY, NZ, 1, 2)

    def test_zero_voltage_zero_force(self, lattice, sim_config):
        from src.operators.differential import build_diff_ops
        from src.operators.force._electric import ElectricForceModule
        from src.operators.force._extra_state import ElectricExtraStatePlugin

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
        from src.operators.differential import build_diff_ops
        from src.operators.force._electric import ElectricForceModule
        from src.operators.force._extra_state import ElectricExtraStatePlugin

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
        from src.operators.force._extra_state import ElectricExtraStatePlugin

        setup = make_electric_setup(lattice, electric_params)
        hi = ElectricExtraStatePlugin.init_state(setup)["h"]
        state = make_state(lattice, rho_value=1.0, h=hi)

        state_new = ElectricExtraStatePlugin.update_state(setup, state, state)
        assert state_new.h is not None
        assert state_new.h.shape == hi.shape

    def test_update_returns_new_state_when_h_is_none(self, lattice, electric_params):
        from src.operators.force._extra_state import ElectricExtraStatePlugin

        setup = make_electric_setup(lattice, electric_params)
        state_no_h = make_state(lattice, rho_value=1.0, h=None)
        result = ElectricExtraStatePlugin.update_state(setup, state_no_h, state_no_h)
        assert result is state_no_h

    def test_update_returns_new_state_when_no_electric_force_spec(self, lattice):
        from src.operators.force._extra_state import ElectricExtraStatePlugin

        setup = SimpleNamespace(forces=None)
        state = make_state(lattice)
        result = ElectricExtraStatePlugin.update_state(setup, state, state)  # ty: ignore[invalid-argument-type]
        assert result is state

    def test_update_raises_when_streaming_fn_none(self, lattice, electric_params):
        from src.operators.force import ForceParams
        from src.operators.force import ForceSetup
        from src.operators.force._extra_state import ElectricExtraStatePlugin

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
        from src.operators.force._extra_state import ElectricExtraStatePlugin

        cfg = SimpleNamespace(electric_force={"strength": 1.0})
        assert ElectricExtraStatePlugin.is_active(cfg) is True  # ty: ignore[invalid-argument-type]

    def test_inactive_when_electric_force_none(self):
        from src.operators.force._extra_state import ElectricExtraStatePlugin

        assert ElectricExtraStatePlugin.is_active(SimpleNamespace(electric_force=None)) is False  # ty: ignore[invalid-argument-type]

    def test_inactive_when_attribute_absent(self):
        from src.operators.force._extra_state import ElectricExtraStatePlugin

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
        from src.config.adapter_dict import DictAdapter
        from src.pipeline.setup import build_setup

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
        from src.operators.force import compute_total_force_ext
        from src.pipeline.runner import init_state

        state = init_state(electric_setup)
        total_force, _ = compute_total_force_ext(electric_setup, state, electric_setup.forces)
        assert total_force is not None
        assert total_force.shape == (NX, NY, NZ, 1, 2)
        assert bool(jnp.all(jnp.isfinite(total_force)))


class TestElectricForceComputeErrors:
    """ElectricForceModule.compute raises TypeError on missing prerequisites."""

    def test_raises_without_gradient_standard(self, lattice, sim_config):
        from src.operators.force._electric import ElectricForceModule

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
        from src.operators.differential import build_diff_ops
        from src.operators.force._electric import ElectricForceModule

        gradient_standard, *_ = build_diff_ops(sim_config, mp_params=None, lattice=lattice)
        state = make_state(lattice, h=None)
        with pytest.raises(TypeError, match=r"state\.h"):
            ElectricForceModule.compute(state, electric_params, gradient_standard=gradient_standard)
