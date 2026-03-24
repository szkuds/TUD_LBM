"""Tests for:
- ``operators.wetting.contact_angle.compute_contact_angle``
- ``operators.wetting.contact_line.compute_contact_line_location``
- ``operators.wetting.hysteresis`` (WettingParams, clamp, cost fns,
optimise routines, update_wetting_state)
- Wiring into ``step_multiphase`` via ``WettingState``
"""

from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

# =====================================================================
# Helpers — build a synthetic droplet rho field
# =====================================================================


def _droplet_rho(
    nx: int,
    ny: int,
    rho_l: float,
    rho_v: float,
    centre_x: float | None = None,
    radius: float | None = None,
):
    """Create a 2D density field with a semicircular droplet on y=0.

    Returns shape ``(nx, ny, 1, 1)``.
    """
    if centre_x is None:
        centre_x = nx / 2.0
    if radius is None:
        radius = nx / 4.0

    _x = jnp.arange(nx, dtype=jnp.float32)
    _y = jnp.arange(ny, dtype=jnp.float32)
    x, y = jnp.meshgrid(_x, _y, indexing="ij")  # (nx, ny)

    dist = jnp.sqrt((x - centre_x) ** 2 + y**2)
    rho_2d = jnp.where(dist < radius, rho_l, rho_v)
    return rho_2d[:, :, None, None]


NX, NY = 64, 32
RHO_L, RHO_V = 1.0, 0.33
RHO_MEAN = (RHO_L + RHO_V) / 2.0


# =====================================================================
# compute_contact_angle
# =====================================================================


class TestComputeContactAngle:
    """Pure-function contact angle computation."""

    def test_returns_two_scalars(self):
        from operators.wetting.contact_angle import compute_contact_angle

        rho = _droplet_rho(NX, NY, RHO_L, RHO_V)
        ca_l, ca_r = compute_contact_angle(rho, RHO_MEAN)
        assert ca_l.shape == ()
        assert ca_r.shape == ()

    def test_angles_in_reasonable_range(self):
        from operators.wetting.contact_angle import compute_contact_angle

        rho = _droplet_rho(NX, NY, RHO_L, RHO_V)
        ca_l, ca_r = compute_contact_angle(rho, RHO_MEAN)
        # A semicircular droplet should give angles roughly around 90°
        assert 30.0 < float(ca_l) < 150.0
        assert 30.0 < float(ca_r) < 150.0

    def test_symmetric_droplet_symmetric_angles(self):
        """A droplet centred on the grid should give equal left/right angles."""
        from operators.wetting.contact_angle import compute_contact_angle

        rho = _droplet_rho(NX, NY, RHO_L, RHO_V, centre_x=NX / 2.0)
        ca_l, ca_r = compute_contact_angle(rho, RHO_MEAN)
        np.testing.assert_allclose(float(ca_l), float(ca_r), atol=2.0)

    def test_jittable(self):
        from operators.wetting.contact_angle import compute_contact_angle

        rho = _droplet_rho(NX, NY, RHO_L, RHO_V)
        jitted = jax.jit(partial(compute_contact_angle, rho_mean=RHO_MEAN))
        ca_l, ca_r = jitted(rho)
        assert not jnp.isnan(ca_l)
        assert not jnp.isnan(ca_r)


# =====================================================================
# compute_contact_line_location
# =====================================================================


class TestComputeContactLineLocation:
    """Pure-function contact-line-location computation."""

    def test_returns_two_scalars(self):
        from operators.wetting.contact_angle import compute_contact_angle
        from operators.wetting.contact_line import compute_contact_line_location

        rho = _droplet_rho(NX, NY, RHO_L, RHO_V)
        ca_l, ca_r = compute_contact_angle(rho, RHO_MEAN)
        cll_l, cll_r = compute_contact_line_location(rho, ca_l, ca_r, RHO_MEAN)
        assert cll_l.shape == ()
        assert cll_r.shape == ()

    def test_left_less_than_right(self):
        from operators.wetting.contact_angle import compute_contact_angle
        from operators.wetting.contact_line import compute_contact_line_location

        rho = _droplet_rho(NX, NY, RHO_L, RHO_V)
        ca_l, ca_r = compute_contact_angle(rho, RHO_MEAN)
        cll_l, cll_r = compute_contact_line_location(rho, ca_l, ca_r, RHO_MEAN)
        assert float(cll_l) < float(cll_r)

    def test_jittable(self):
        from operators.wetting.contact_angle import compute_contact_angle
        from operators.wetting.contact_line import compute_contact_line_location

        rho = _droplet_rho(NX, NY, RHO_L, RHO_V)
        ca_l, ca_r = compute_contact_angle(rho, RHO_MEAN)

        jitted = jax.jit(partial(compute_contact_line_location, rho_mean=RHO_MEAN))
        cll_l, cll_r = jitted(rho, ca_l, ca_r)
        assert not jnp.isnan(cll_l)
        assert not jnp.isnan(cll_r)


# =====================================================================
# WettingParams / helpers
# =====================================================================


class TestWettingParamsHelpers:
    """WettingParams, clamp, cost functions."""

    def test_wetting_params_is_pytree(self):
        from operators.wetting.hysteresis import WettingParams

        p = WettingParams(
            d_rho_left=jnp.array(0.1),
            d_rho_right=jnp.array(0.1),
            phi_left=jnp.array(1.2),
            phi_right=jnp.array(1.2),
        )
        leaves, treedef = jax.tree.flatten(p)
        p2 = treedef.unflatten(leaves)
        assert isinstance(p2, WettingParams)
        np.testing.assert_allclose(float(p2.d_rho_left), 0.1)

    def test_clamp_params(self):
        from operators.wetting.hysteresis import WettingParams
        from operators.wetting.hysteresis import _clamp_params

        p = WettingParams(
            d_rho_left=jnp.array(-0.5),
            d_rho_right=jnp.array(0.5),
            phi_left=jnp.array(0.5),
            phi_right=jnp.array(2.0),
        )
        clamped = _clamp_params(p)
        np.testing.assert_allclose(float(clamped.d_rho_left), 0.0, atol=1e-6)
        np.testing.assert_allclose(float(clamped.d_rho_right), 0.2, atol=1e-6)
        np.testing.assert_allclose(float(clamped.phi_left), 1.0, atol=1e-6)
        np.testing.assert_allclose(float(clamped.phi_right), 1.5, atol=1e-6)

    def test_cost_cll(self):
        from operators.wetting.hysteresis import _cost_cll

        assert float(_cost_cll(jnp.array(5.0), jnp.array(3.0))) == 2.0

    def test_cost_ca(self):
        from operators.wetting.hysteresis import _cost_ca

        assert float(_cost_ca(jnp.array(90.0), jnp.array(85.0))) == 5.0


# =====================================================================
# _optimise_single_param
# =====================================================================


class TestOptimiseSingleParam:
    """Inner optimisation loop."""

    def test_reduces_loss(self):
        import optax
        from operators.wetting.hysteresis import WettingParams
        from operators.wetting.hysteresis import _optimise_single_param

        # Simple quadratic objective: minimise (d_rho_left - 0.1)^2
        target = 0.1

        def objective(p):
            return (p.d_rho_left - target) ** 2

        def mask_fn(g):
            return WettingParams(
                g.d_rho_left,
                jnp.zeros_like(g.d_rho_right),
                jnp.zeros_like(g.phi_left),
                jnp.zeros_like(g.phi_right),
            )

        p0 = WettingParams(
            d_rho_left=jnp.array(0.15),
            d_rho_right=jnp.array(0.1),
            phi_left=jnp.array(1.2),
            phi_right=jnp.array(1.2),
        )
        opt = optax.adam(0.01)
        _p_final, loss_final = _optimise_single_param(objective, p0, mask_fn, opt, 50)
        initial_loss = float(objective(p0))
        assert float(loss_final) < initial_loss

    def test_jittable(self):
        import optax
        from operators.wetting.hysteresis import WettingParams
        from operators.wetting.hysteresis import _optimise_single_param

        def objective(p):
            return (p.d_rho_left - 0.1) ** 2

        def mask_fn(g):
            return WettingParams(
                g.d_rho_left,
                jnp.zeros_like(g.d_rho_right),
                jnp.zeros_like(g.phi_left),
                jnp.zeros_like(g.phi_right),
            )

        p0 = WettingParams(
            d_rho_left=jnp.array(0.15),
            d_rho_right=jnp.array(0.1),
            phi_left=jnp.array(1.2),
            phi_right=jnp.array(1.2),
        )
        opt = optax.adam(0.01)

        @jax.jit
        def run_opt(initial_params):
            return _optimise_single_param(objective, initial_params, mask_fn, opt, 10)

        _p_final, loss = run_opt(p0)
        assert not jnp.isnan(loss)


# =====================================================================
# update_wetting_state
# =====================================================================


class TestUpdateWettingState:
    """Top-level ``update_wetting_state`` integration tests."""

    @staticmethod
    def _make_setup():
        from config.simulation_config import SimulationConfig
        from setup.simulation_setup import build_setup

        cfg = SimulationConfig(
            sim_type="multiphase",
            grid_shape=(NX, NY),
            tau=0.99,
            nt=5,
            eos="double-well",
            kappa=0.017,
            rho_l=RHO_L,
            rho_v=RHO_V,
            interface_width=4,
            hysteresis_config={
                "ca_advancing": 120.0,
                "ca_receding": 60.0,
                "learning_rate": 0.01,
                "max_iterations": 5,
            },
        )
        return build_setup(cfg)

    @staticmethod
    def _make_wetting_state():
        from state.state import WettingState

        return WettingState(
            d_rho_left=jnp.array(0.05),
            d_rho_right=jnp.array(0.05),
            phi_left=jnp.array(1.2),
            phi_right=jnp.array(1.2),
            ca_left=jnp.array(90.0),
            ca_right=jnp.array(90.0),
            cll_left=jnp.array(16.0),
            cll_right=jnp.array(48.0),
        )

    def test_returns_wetting_state(self):
        from operators.wetting.hysteresis import update_wetting_state
        from state.state import WettingState

        setup = self._make_setup()
        rho = _droplet_rho(NX, NY, RHO_L, RHO_V)
        f_bc = jnp.ones((NX, NY, 9, 1)) * (1.0 / 9.0)
        force = jnp.zeros((NX, NY, 1, 2))
        wetting = self._make_wetting_state()

        new_wetting = update_wetting_state(wetting, rho, setup, f_bc, force)
        assert isinstance(new_wetting, WettingState)

    def test_ca_fields_updated(self):
        from operators.wetting.contact_angle import compute_contact_angle
        from operators.wetting.hysteresis import update_wetting_state

        setup = self._make_setup()
        rho = _droplet_rho(NX, NY, RHO_L, RHO_V)
        f_bc = jnp.ones((NX, NY, 9, 1)) * (1.0 / 9.0)
        force = jnp.zeros((NX, NY, 1, 2))
        wetting = self._make_wetting_state()

        new_wetting = update_wetting_state(wetting, rho, setup, f_bc, force)
        # ca_left and ca_right should reflect the actual droplet angles
        # measured by compute_contact_angle (not the initial placeholder)
        expected_ca_l, expected_ca_r = compute_contact_angle(rho, RHO_MEAN)
        np.testing.assert_allclose(
            float(new_wetting.ca_left),
            float(expected_ca_l),
            atol=1e-4,
        )
        np.testing.assert_allclose(
            float(new_wetting.ca_right),
            float(expected_ca_r),
            atol=1e-4,
        )

    def test_cll_fields_updated(self):
        from operators.wetting.hysteresis import update_wetting_state

        setup = self._make_setup()
        rho = _droplet_rho(NX, NY, RHO_L, RHO_V)
        f_bc = jnp.ones((NX, NY, 9, 1)) * (1.0 / 9.0)
        force = jnp.zeros((NX, NY, 1, 2))
        wetting = self._make_wetting_state()

        new_wetting = update_wetting_state(wetting, rho, setup, f_bc, force)
        # CLL should reflect actual droplet footprint
        assert float(new_wetting.cll_left) < float(new_wetting.cll_right)

    def test_params_stay_clamped(self):
        from operators.wetting.hysteresis import update_wetting_state

        setup = self._make_setup()
        rho = _droplet_rho(NX, NY, RHO_L, RHO_V)
        f_bc = jnp.ones((NX, NY, 9, 1)) * (1.0 / 9.0)
        force = jnp.zeros((NX, NY, 1, 2))
        wetting = self._make_wetting_state()

        new_wetting = update_wetting_state(wetting, rho, setup, f_bc, force)
        assert 0.0 <= float(new_wetting.d_rho_left) <= 0.2
        assert 0.0 <= float(new_wetting.d_rho_right) <= 0.2
        assert 1.0 <= float(new_wetting.phi_left) <= 1.5
        assert 1.0 <= float(new_wetting.phi_right) <= 1.5

    def test_no_nan(self):
        from operators.wetting.hysteresis import update_wetting_state
        from state.state import WettingState

        setup = self._make_setup()
        rho = _droplet_rho(NX, NY, RHO_L, RHO_V)
        f_bc = jnp.ones((NX, NY, 9, 1)) * (1.0 / 9.0)
        force = jnp.zeros((NX, NY, 1, 2))
        wetting = self._make_wetting_state()

        new_wetting = update_wetting_state(wetting, rho, setup, f_bc, force)
        for field_name in WettingState._fields[:8]:  # skip opt_state
            val = getattr(new_wetting, field_name)
            if val is not None:
                assert not jnp.isnan(val).any(), f"NaN in {field_name}"


# =====================================================================
# step_multiphase with wetting
# =====================================================================


class TestStepMultiphaseWithWetting:
    """``step_multiphase`` correctly propagates WettingState."""

    @staticmethod
    def _setup_and_state():
        from config.simulation_config import SimulationConfig
        from operators.equilibrium.equilibrium import compute_equilibrium
        from runner.run import init_state
        from setup.simulation_setup import build_setup
        from state.state import WettingState

        cfg = SimulationConfig(
            sim_type="multiphase",
            grid_shape=(NX, NY),
            tau=0.99,
            nt=3,
            eos="double-well",
            kappa=0.017,
            rho_l=RHO_L,
            rho_v=RHO_V,
            interface_width=4,
            hysteresis_config={
                "ca_advancing": 120.0,
                "ca_receding": 60.0,
                "learning_rate": 0.01,
                "max_iterations": 3,
            },
        )
        setup = build_setup(cfg)

        # Initialise with a droplet so compute_contact_angle finds an interface
        rho = _droplet_rho(NX, NY, RHO_L, RHO_V)
        u = jnp.zeros((NX, NY, 1, 2))
        lattice = setup.lattice
        feq = compute_equilibrium(rho, u, lattice)
        state = init_state(setup, f=feq)

        # Attach a WettingState
        wetting = WettingState(
            d_rho_left=jnp.array(0.05),
            d_rho_right=jnp.array(0.05),
            phi_left=jnp.array(1.2),
            phi_right=jnp.array(1.2),
            ca_left=jnp.array(90.0),
            ca_right=jnp.array(90.0),
            cll_left=jnp.array(16.0),
            cll_right=jnp.array(48.0),
        )
        state = state._replace(wetting=wetting)
        return setup, state

    def test_wetting_state_propagated(self):
        from runner.step import step_multiphase
        from state.state import WettingState

        setup, state = self._setup_and_state()
        new_state = step_multiphase(setup, state)

        assert new_state.wetting is not None
        assert isinstance(new_state.wetting, WettingState)

    def test_step_increments_t(self):
        from runner.step import step_multiphase

        setup, state = self._setup_and_state()
        new_state = step_multiphase(setup, state)
        assert int(new_state.t) == 1

    def test_wetting_fields_no_nan(self):
        from runner.step import step_multiphase

        setup, state = self._setup_and_state()
        new_state = step_multiphase(setup, state)

        for field_name in (
            "d_rho_left",
            "d_rho_right",
            "phi_left",
            "phi_right",
            "ca_left",
            "ca_right",
            "cll_left",
            "cll_right",
        ):
            val = getattr(new_state.wetting, field_name)
            assert not jnp.isnan(val).any(), f"NaN in wetting.{field_name}"

    def test_without_wetting_state_unchanged(self):
        """When wetting is None, step should not fail."""
        from config.simulation_config import SimulationConfig
        from runner.run import init_state
        from runner.step import step_multiphase
        from setup.simulation_setup import build_setup

        cfg = SimulationConfig(
            sim_type="multiphase",
            grid_shape=(16, 16),
            tau=0.99,
            nt=3,
            eos="double-well",
            kappa=0.017,
            rho_l=RHO_L,
            rho_v=RHO_V,
            interface_width=4,
        )
        setup = build_setup(cfg)
        state = init_state(setup)
        assert state.wetting is None

        new_state = step_multiphase(setup, state)
        assert new_state.wetting is None
        assert int(new_state.t) == 1


# =====================================================================
# Backward compat: legacy tests still pass
# =====================================================================


class TestLegacyPhase3Unbroken:
    """Phase 3 functional API should still pass."""

    def test_phase3_functional_step(self):
        from config.simulation_config import SimulationConfig
        from runner.run import init_state
        from runner.step import step_single_phase
        from setup.simulation_setup import build_setup

        cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=10)
        setup = build_setup(cfg)
        state = init_state(setup)

        new_state = step_single_phase(setup, state)
        assert int(new_state.t) == 1
