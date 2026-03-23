"""Tests — pure-function operators.

Tests for:
    - ``operators.collision.bgk.collide_bgk``
    - ``operators.collision.mrt.collide_mrt``
    - ``operators.collision.factory.get_collision_fn``
    - ``operators.streaming.streaming.stream``
    - ``operators.equilibrium.equilibrium.compute_equilibrium``
    - ``operators.macroscopic.single_phase.compute_macroscopic``
    - ``operators.macroscopic.multiphase.compute_macroscopic_multiphase``
    - ``operators.boundary.bounce_back.apply_bounce_back``
    - ``operators.boundary.symmetry.apply_symmetry``
    - ``operators.boundary.periodic.apply_periodic``
    - ``operators.boundary.composite.build_composite_bc``

Each operator is verified to be jittable on an 8×8 (or 16×16) grid
without any class instance.
"""

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from setup.lattice import build_lattice

# ── Shared helpers ───────────────────────────────────────────────────

NX, NY = 8, 8


@pytest.fixture
def lattice():
    return build_lattice("D2Q9")


@pytest.fixture
def rest_state(lattice):
    """Uniform density=1, velocity=0 populations at equilibrium."""
    q = lattice.q
    rho = jnp.ones((NX, NY, 1, 1))
    u = jnp.zeros((NX, NY, 1, 2))
    # At rest: feq_0 = rho - sum_rest, feq_i = w_i * rho for i>0
    # Actually, compute equilibrium properly
    from operators.equilibrium.equilibrium import compute_equilibrium

    feq = compute_equilibrium(rho, u, lattice)
    return feq, rho, u


# =====================================================================
# Collision — BGK
# =====================================================================


class TestCollideBGK:
    """``collide_bgk`` is a correct, jittable pure function."""

    def test_no_source(self, lattice, rest_state):
        from operators.collision.bgk import collide_bgk

        feq, rho, u = rest_state
        f = feq  # at equilibrium
        tau = 0.8

        f_col = collide_bgk(f, feq, tau)

        # At equilibrium f == feq → f_col should == f
        np.testing.assert_allclose(np.array(f_col), np.array(f), atol=1e-6)

    def test_with_source(self, lattice, rest_state):
        from operators.collision.bgk import collide_bgk

        feq, rho, u = rest_state
        f = feq
        tau = 0.8
        source = jnp.ones_like(f) * 0.01

        f_col = collide_bgk(f, feq, tau, source=source)

        # f_col = f + (1 - 1/(2*tau)) * source
        omega = 1.0 / tau
        expected = f + (1.0 - 0.5 * omega) * source
        np.testing.assert_allclose(np.array(f_col), np.array(expected), atol=1e-6)

    def test_shape_preserved(self, lattice, rest_state):
        from operators.collision.bgk import collide_bgk

        feq, _, _ = rest_state
        f_col = collide_bgk(feq, feq, 0.8)
        assert f_col.shape == feq.shape

    def test_jittable(self, lattice, rest_state):
        from operators.collision.bgk import collide_bgk

        feq, _, _ = rest_state
        jitted = jax.jit(collide_bgk)
        f_col = jitted(feq, feq, 0.8)
        np.testing.assert_allclose(np.array(f_col), np.array(feq), atol=1e-6)

    def test_matches_legacy_formula(self, lattice):
        """Verify against the explicit legacy BGK formula."""
        from operators.collision.bgk import collide_bgk

        key = jax.random.PRNGKey(42)
        f = jax.random.uniform(key, (NX, NY, 9, 1))
        feq = jax.random.uniform(key, (NX, NY, 9, 1))
        tau = 0.9
        source = jax.random.uniform(key, (NX, NY, 9, 1)) * 0.01

        result = collide_bgk(f, feq, tau, source)

        omega = 1.0 / tau
        expected = (1.0 - omega) * f + omega * feq + (1.0 - 0.5 * omega) * source
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-6)


# =====================================================================
# Collision — MRT
# =====================================================================


class TestCollideMRT:
    """``collide_mrt`` is a correct, jittable pure function."""

    def test_at_equilibrium(self, lattice, rest_state):
        from operators.collision.mrt import collide_mrt

        feq, _, _ = rest_state
        tau = 0.8

        f_col = collide_mrt(feq, feq, tau, source=None)

        # f_col = f + M_inv K M (feq - f) = f when feq == f
        np.testing.assert_allclose(np.array(f_col), np.array(feq), atol=1e-5)

    def test_with_source(self, lattice, rest_state):
        from operators.collision.mrt import collide_mrt

        feq, _, _ = rest_state
        source = jnp.ones_like(feq) * 0.001

        f_col = collide_mrt(feq, feq, 0.8, source=source)

        # Should differ from feq by the source contribution
        diff = jnp.abs(f_col - feq)
        assert float(jnp.max(diff)) > 0.0

    def test_shape_preserved(self, lattice, rest_state):
        from operators.collision.mrt import collide_mrt

        feq, _, _ = rest_state
        f_col = collide_mrt(feq, feq, 0.8)
        assert f_col.shape == feq.shape

    def test_jittable(self, lattice, rest_state):
        from operators.collision.mrt import collide_mrt

        feq, _, _ = rest_state
        jitted = jax.jit(collide_mrt)
        f_col = jitted(feq, feq, 0.8)
        assert f_col.shape == feq.shape

    def test_custom_k_diag(self, lattice, rest_state):
        from operators.collision.mrt import collide_mrt

        feq, _, _ = rest_state
        k_diag = jnp.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.8, 0.8])
        f_col = collide_mrt(feq, feq, 0.8, k_diag=k_diag)
        assert f_col.shape == feq.shape


# =====================================================================
# Collision — factory
# =====================================================================


class TestCollisionFactory:
    """``get_collision_fn`` dispatches to the correct function."""

    def test_bgk(self):
        from operators.collision.bgk import collide_bgk
        from operators.collision.factory import build_collision_fn

        assert build_collision_fn("bgk") is collide_bgk

    def test_mrt(self):
        from operators.collision.factory import build_collision_fn
        from operators.collision.mrt import collide_mrt

        assert build_collision_fn("mrt") is collide_mrt

    def test_unknown_raises(self):
        from operators.collision.factory import build_collision_fn

        with pytest.raises(ValueError, match="Unknown collision scheme"):
            build_collision_fn("invalid")


# =====================================================================
# Streaming
# =====================================================================


class TestStream:
    """``stream`` propagates populations correctly."""

    def test_rest_direction_unchanged(self, lattice):
        """Direction 0 (zero velocity) should not move."""
        from operators.streaming.streaming import stream

        f = jnp.zeros((NX, NY, 9, 1))
        f = f.at[3, 3, 0, 0].set(1.0)

        f_s = stream(f, lattice)

        # Direction 0 has c = (0,0) → no shift
        assert float(f_s[3, 3, 0, 0]) == 1.0

    def test_direction_1_shifts_right(self, lattice):
        """Direction 1 (cx=1, cy=0) shifts +1 in x."""
        from operators.streaming.streaming import stream

        f = jnp.zeros((NX, NY, 9, 1))
        f = f.at[3, 3, 1, 0].set(1.0)

        f_s = stream(f, lattice)

        assert float(f_s[4, 3, 1, 0]) == 1.0
        assert float(f_s[3, 3, 1, 0]) == 0.0

    def test_periodic_wrap(self, lattice):
        """Streaming wraps around the domain (periodic)."""
        from operators.streaming.streaming import stream

        f = jnp.zeros((NX, NY, 9, 1))
        # Put pulse at right edge, direction 1 (cx=1)
        f = f.at[NX - 1, 3, 1, 0].set(1.0)

        f_s = stream(f, lattice)

        # Should wrap to x=0
        assert float(f_s[0, 3, 1, 0]) == 1.0

    def test_shape_preserved(self, lattice):
        from operators.streaming.streaming import stream

        f = jnp.ones((NX, NY, 9, 1))
        f_s = stream(f, lattice)
        assert f_s.shape == f.shape

    def test_jittable(self, lattice):
        from operators.streaming.streaming import stream

        f = jnp.ones((NX, NY, 9, 1))
        jitted_stream = jax.jit(partial(stream, lattice=lattice))
        f_s = jitted_stream(f)
        assert f_s.shape == f.shape

    def test_mass_conservation(self, lattice):
        """Total mass should be conserved after streaming."""
        from operators.streaming.streaming import stream

        key = jax.random.PRNGKey(0)
        f = jax.random.uniform(key, (NX, NY, 9, 1))

        f_s = stream(f, lattice)

        np.testing.assert_allclose(float(jnp.sum(f)), float(jnp.sum(f_s)), rtol=1e-6)


# =====================================================================
# Equilibrium
# =====================================================================


class TestComputeEquilibrium:
    """``compute_equilibrium`` matches the legacy WB equilibrium."""

    def test_rest_state(self, lattice):
        from operators.equilibrium.equilibrium import compute_equilibrium

        rho = jnp.ones((NX, NY, 1, 1))
        u = jnp.zeros((NX, NY, 1, 2))

        feq = compute_equilibrium(rho, u, lattice)

        assert feq.shape == (NX, NY, 9, 1)
        # Mass conservation: sum over q should equal rho
        np.testing.assert_allclose(
            np.array(jnp.sum(feq, axis=2, keepdims=True)),
            np.array(rho),
            atol=1e-6,
        )

    def test_mass_conservation_with_velocity(self, lattice):
        from operators.equilibrium.equilibrium import compute_equilibrium

        rho = jnp.ones((NX, NY, 1, 1)) * 1.5
        u = jnp.ones((NX, NY, 1, 2)) * 0.05

        feq = compute_equilibrium(rho, u, lattice)

        # sum_q feq_q = rho everywhere
        np.testing.assert_allclose(
            np.array(jnp.sum(feq, axis=2, keepdims=True)),
            np.array(rho),
            atol=1e-6,
        )

    def test_shape_preserved(self, lattice):
        from operators.equilibrium.equilibrium import compute_equilibrium

        rho = jnp.ones((NX, NY, 1, 1))
        u = jnp.zeros((NX, NY, 1, 2))

        feq = compute_equilibrium(rho, u, lattice)
        assert feq.shape == (NX, NY, 9, 1)

    def test_jittable(self, lattice):
        from operators.equilibrium.equilibrium import compute_equilibrium

        rho = jnp.ones((NX, NY, 1, 1))
        u = jnp.zeros((NX, NY, 1, 2))

        jitted_eq = jax.jit(partial(compute_equilibrium, lattice=lattice))
        feq = jitted_eq(rho, u)
        assert feq.shape == (NX, NY, 9, 1)


# =====================================================================
# Macroscopic — single phase
# =====================================================================


class TestComputeMacroscopic:
    """``compute_macroscopic`` extracts density and velocity."""

    def test_rest_state(self, lattice, rest_state):
        from operators.macroscopic.single_phase import compute_macroscopic

        feq, rho_expected, u_expected = rest_state
        rho, u = compute_macroscopic(feq, lattice)

        np.testing.assert_allclose(np.array(rho), np.array(rho_expected), atol=1e-6)
        np.testing.assert_allclose(np.array(u), np.array(u_expected), atol=1e-6)

    def test_density_is_sum(self, lattice):
        from operators.macroscopic.single_phase import compute_macroscopic

        key = jax.random.PRNGKey(1)
        f = jax.random.uniform(key, (NX, NY, 9, 1), minval=0.05)

        rho, u = compute_macroscopic(f, lattice)

        expected_rho = jnp.sum(f, axis=2, keepdims=True)
        np.testing.assert_allclose(np.array(rho), np.array(expected_rho), atol=1e-6)

    def test_with_force_returns_three(self, lattice, rest_state):
        from operators.macroscopic.single_phase import compute_macroscopic

        feq, _, _ = rest_state
        force = jnp.ones((NX, NY, 1, 2)) * 0.001

        result = compute_macroscopic(feq, lattice, force=force)
        assert len(result) == 3
        rho, u_eq, force_out = result
        assert rho.shape == (NX, NY, 1, 1)
        assert u_eq.shape == (NX, NY, 1, 2)

    def test_shape_preserved(self, lattice, rest_state):
        from operators.macroscopic.single_phase import compute_macroscopic

        feq, _, _ = rest_state
        rho, u = compute_macroscopic(feq, lattice)
        assert rho.shape == (NX, NY, 1, 1)
        assert u.shape == (NX, NY, 1, 2)

    def test_jittable(self, lattice, rest_state):
        from operators.macroscopic.single_phase import compute_macroscopic

        feq, _, _ = rest_state
        jitted_mac = jax.jit(partial(compute_macroscopic, lattice=lattice))
        rho, u = jitted_mac(feq)
        assert rho.shape == (NX, NY, 1, 1)


# =====================================================================
# Macroscopic — multiphase
# =====================================================================


class TestComputeMacroscopicMultiphase:
    """``compute_macroscopic_multiphase`` returns (rho, u_eq, force)."""

    def _mp_params(self):
        from setup.simulation_setup import MultiphaseParams

        return MultiphaseParams(
            eos="double-well",
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
        )

    def _diff_ops(self, lattice):
        from operators.differential.config import DifferentialConfig
        from operators.differential.factory import build_differential_operators

        cfg = DifferentialConfig(
            w=lattice.w,
            c=lattice.c,
            pad_modes=["wrap", "wrap", "wrap", "wrap"],
        )
        return build_differential_operators(cfg)

    def test_returns_triple(self, lattice):
        from operators.macroscopic.multiphase import compute_macroscopic_multiphase

        mp = self._mp_params()
        diff_ops = self._diff_ops(lattice)
        f = jnp.ones((16, 16, 9, 1)) * (1.0 / 9.0)

        rho, u_eq, force_total = compute_macroscopic_multiphase(
            f,
            lattice,
            mp,
            diff_ops=diff_ops,
        )

        assert rho.shape == (16, 16, 1, 1)
        assert u_eq.shape == (16, 16, 1, 2)
        assert force_total.shape == (16, 16, 1, 2)

    def test_uniform_field_zero_force(self, lattice):
        """A perfectly uniform density field → zero interaction force."""
        from operators.macroscopic.multiphase import compute_macroscopic_multiphase

        mp = self._mp_params()
        diff_ops = self._diff_ops(lattice)
        # Uniform density = rho_l
        rho_0 = mp.rho_l
        f = jnp.ones((16, 16, 9, 1)) * (rho_0 / 9.0)

        rho, u_eq, force_total = compute_macroscopic_multiphase(
            f,
            lattice,
            mp,
            diff_ops=diff_ops,
        )

        # Interaction force should be ~0 for uniform field (gradients vanish)
        np.testing.assert_allclose(np.array(force_total), 0.0, atol=1e-5)

    def test_jittable(self, lattice):
        from operators.macroscopic.multiphase import compute_macroscopic_multiphase

        mp = self._mp_params()
        diff_ops = self._diff_ops(lattice)
        f = jnp.ones((16, 16, 9, 1)) * (1.0 / 9.0)

        jitted_mp = jax.jit(
            partial(
                compute_macroscopic_multiphase,
                lattice=lattice,
                mp=mp,
                diff_ops=diff_ops,
            ),
        )
        rho, u_eq, force = jitted_mp(f)
        assert rho.shape == (16, 16, 1, 1)

    def test_with_external_force(self, lattice):
        from operators.macroscopic.multiphase import compute_macroscopic_multiphase

        mp = self._mp_params()
        diff_ops = self._diff_ops(lattice)
        f = jnp.ones((16, 16, 9, 1)) * (1.0 / 9.0)
        force_ext = jnp.ones((16, 16, 1, 2)) * 0.001

        rho, u_eq, force_total = compute_macroscopic_multiphase(
            f,
            lattice,
            mp,
            force_ext=force_ext,
            diff_ops=diff_ops,
        )

        # Force total should include the external contribution
        # For uniform field, force_int ≈ 0, so force_total ≈ force_ext
        np.testing.assert_allclose(
            np.array(force_total),
            np.array(force_ext),
            atol=1e-4,
        )


# =====================================================================
# Boundary — bounce-back
# =====================================================================


class TestApplyBounceBack:
    """``apply_bounce_back`` applies bounce-back on a single edge."""

    def test_bottom_edge(self, lattice):
        from operators.boundary.bounce_back import apply_bounce_back

        key = jax.random.PRNGKey(10)
        f_s = jax.random.uniform(key, (NX, NY, 9, 1))
        f_c = jax.random.uniform(key, (NX, NY, 9, 1)) * 2.0

        f_out = apply_bounce_back(f_s, f_c, lattice, "bottom")

        # For incoming directions at bottom (top_indices), the bottom row
        # should have values from the opposite directions of f_c
        opp = np.array(lattice.opp_indices)
        for idx in np.array(lattice.top_indices):
            np.testing.assert_allclose(
                np.array(f_out[:, 0, idx, 0]),
                np.array(f_c[:, 0, opp[idx], 0]),
            )

    def test_top_edge(self, lattice):
        from operators.boundary.bounce_back import apply_bounce_back

        key = jax.random.PRNGKey(11)
        f_s = jax.random.uniform(key, (NX, NY, 9, 1))
        f_c = jax.random.uniform(key, (NX, NY, 9, 1)) * 2.0

        f_out = apply_bounce_back(f_s, f_c, lattice, "top")

        opp = np.array(lattice.opp_indices)
        for idx in np.array(lattice.bottom_indices):
            np.testing.assert_allclose(
                np.array(f_out[:, -1, idx, 0]),
                np.array(f_c[:, -1, opp[idx], 0]),
            )

    def test_shape_preserved(self, lattice):
        from operators.boundary.bounce_back import apply_bounce_back

        f = jnp.ones((NX, NY, 9, 1))
        f_out = apply_bounce_back(f, f, lattice, "bottom")
        assert f_out.shape == f.shape

    def test_jittable(self, lattice):
        from operators.boundary.bounce_back import apply_bounce_back

        f = jnp.ones((NX, NY, 9, 1))
        jitted_bb = jax.jit(
            partial(
                apply_bounce_back,
                lattice=lattice,
                edge="bottom",
            ),
        )
        f_out = jitted_bb(f, f)
        assert f_out.shape == f.shape

    def test_unknown_edge_raises(self, lattice):
        from operators.boundary.bounce_back import apply_bounce_back

        f = jnp.ones((NX, NY, 9, 1))
        with pytest.raises(ValueError, match="Unknown edge"):
            apply_bounce_back(f, f, lattice, "diagonal")


# =====================================================================
# Boundary — symmetry
# =====================================================================


class TestApplySymmetry:
    """``apply_symmetry`` mirrors distributions on a single edge."""

    def test_bottom_edge_shape(self, lattice):
        from operators.boundary.symmetry import apply_symmetry

        f = jnp.ones((NX, NY, 9, 1))
        f_out = apply_symmetry(f, f, lattice, "bottom")
        assert f_out.shape == f.shape

    def test_jittable(self, lattice):
        from operators.boundary.symmetry import apply_symmetry

        f = jnp.ones((NX, NY, 9, 1))
        jitted_sym = jax.jit(
            partial(
                apply_symmetry,
                lattice=lattice,
                edge="bottom",
            ),
        )
        f_out = jitted_sym(f, f)
        assert f_out.shape == f.shape

    def test_unknown_edge_raises(self, lattice):
        from operators.boundary.symmetry import apply_symmetry

        f = jnp.ones((NX, NY, 9, 1))
        with pytest.raises(ValueError, match="Unknown edge"):
            apply_symmetry(f, f, lattice, "diagonal")


# =====================================================================
# Boundary — periodic (no-op)
# =====================================================================


class TestApplyPeriodic:
    """``apply_periodic`` is a no-op."""

    def test_returns_unchanged(self, lattice):
        from operators.boundary.periodic import apply_periodic

        key = jax.random.PRNGKey(20)
        f = jax.random.uniform(key, (NX, NY, 9, 1))

        f_out = apply_periodic(f, f, lattice, "top")

        np.testing.assert_array_equal(np.array(f_out), np.array(f))

    def test_jittable(self, lattice):
        from operators.boundary.periodic import apply_periodic

        f = jnp.ones((NX, NY, 9, 1))
        jitted_per = jax.jit(
            partial(
                apply_periodic,
                lattice=lattice,
                edge="top",
            ),
        )
        f_out = jitted_per(f, f)
        assert f_out.shape == f.shape


# =====================================================================
# Boundary — composite
# =====================================================================


class TestBuildCompositeBC:
    """``build_composite_bc`` chains per-edge BC functions."""

    def test_all_periodic(self, lattice):
        from operators.boundary.composite import build_composite_bc

        bc_fn = build_composite_bc(None, lattice)

        key = jax.random.PRNGKey(30)
        f = jax.random.uniform(key, (NX, NY, 9, 1))

        # All periodic → identity
        f_out = bc_fn(f, f, None)
        np.testing.assert_array_equal(np.array(f_out), np.array(f))

    def test_bounce_back_bottom(self, lattice):
        from operators.boundary.composite import build_composite_bc

        bc_config = {
            "top": "periodic",
            "bottom": "bounce-back",
            "left": "periodic",
            "right": "periodic",
        }
        bc_fn = build_composite_bc(bc_config, lattice)

        key = jax.random.PRNGKey(31)
        f_s = jax.random.uniform(key, (NX, NY, 9, 1))
        f_c = jax.random.uniform(key, (NX, NY, 9, 1)) * 2.0

        f_out = bc_fn(f_s, f_c, None)

        # Bottom row should have bounce-back applied
        opp = np.array(lattice.opp_indices)
        for idx in np.array(lattice.top_indices):
            np.testing.assert_allclose(
                np.array(f_out[:, 0, idx, 0]),
                np.array(f_c[:, 0, opp[idx], 0]),
            )

    def test_mixed_bcs(self, lattice):
        from operators.boundary.composite import build_composite_bc

        bc_config = {
            "top": "symmetry",
            "bottom": "bounce-back",
            "left": "periodic",
            "right": "periodic",
        }
        bc_fn = build_composite_bc(bc_config, lattice)

        f = jnp.ones((NX, NY, 9, 1))
        f_out = bc_fn(f, f, None)
        assert f_out.shape == f.shape

    def test_composite_jittable(self, lattice):
        """The composite BC closure is jittable."""
        from operators.boundary.composite import build_composite_bc

        bc_config = {
            "top": "symmetry",
            "bottom": "bounce-back",
            "left": "periodic",
            "right": "periodic",
        }
        bc_fn = build_composite_bc(bc_config, lattice)

        f = jnp.ones((NX, NY, 9, 1))
        f_out = jax.jit(bc_fn)(f, f, None)
        assert f_out.shape == f.shape


# =====================================================================
# End-to-end: full LBM step with pure functions (no class instances)
# =====================================================================


class TestEndToEndPureFunctions:
    """A complete LBM step using only pure functions — no class instances."""

    def test_single_step_mass_conservation(self, lattice):
        """One full step should conserve mass on a periodic domain."""
        from operators.collision.bgk import collide_bgk
        from operators.equilibrium.equilibrium import compute_equilibrium
        from operators.macroscopic.single_phase import compute_macroscopic
        from operators.streaming.streaming import stream

        rho = jnp.ones((NX, NY, 1, 1))
        u = jnp.zeros((NX, NY, 1, 2))
        f = compute_equilibrium(rho, u, lattice)
        tau = 0.8

        # Step: macroscopic → equilibrium → collision → streaming
        rho_new, u_new = compute_macroscopic(f, lattice)
        feq = compute_equilibrium(rho_new, u_new, lattice)
        f_col = collide_bgk(f, feq, tau)
        f_stream = stream(f_col, lattice)

        mass_before = float(jnp.sum(f))
        mass_after = float(jnp.sum(f_stream))

        np.testing.assert_allclose(mass_before, mass_after, rtol=1e-6)

    def test_jit_full_step(self, lattice):
        """A full step wrapped in jax.jit compiles and runs."""
        from operators.collision.bgk import collide_bgk
        from operators.equilibrium.equilibrium import compute_equilibrium
        from operators.macroscopic.single_phase import compute_macroscopic
        from operators.streaming.streaming import stream

        def one_step(f, tau):
            rho, u = compute_macroscopic(f, lattice)
            feq = compute_equilibrium(rho, u, lattice)
            f_col = collide_bgk(f, feq, tau)
            return stream(f_col, lattice)

        rho = jnp.ones((NX, NY, 1, 1))
        u = jnp.zeros((NX, NY, 1, 2))
        f = compute_equilibrium(rho, u, lattice)

        jitted_step = jax.jit(one_step, static_argnums=(1,))
        f_new = jitted_step(f, 0.8)

        assert f_new.shape == f.shape
        np.testing.assert_allclose(float(jnp.sum(f_new)), float(jnp.sum(f)), rtol=1e-6)

    def test_step_with_bounce_back(self, lattice):
        """Full step with bounce-back BCs compiles and runs."""
        from operators.boundary.composite import build_composite_bc
        from operators.collision.bgk import collide_bgk
        from operators.equilibrium.equilibrium import compute_equilibrium
        from operators.macroscopic.single_phase import compute_macroscopic
        from operators.streaming.streaming import stream

        bc_config = {
            "top": "bounce-back",
            "bottom": "bounce-back",
            "left": "periodic",
            "right": "periodic",
        }
        bc_fn = build_composite_bc(bc_config, lattice)

        rho = jnp.ones((NX, NY, 1, 1))
        u = jnp.zeros((NX, NY, 1, 2))
        f = compute_equilibrium(rho, u, lattice)
        tau = 0.8

        rho_n, u_n = compute_macroscopic(f, lattice)
        feq = compute_equilibrium(rho_n, u_n, lattice)
        f_col = collide_bgk(f, feq, tau)
        f_stream = stream(f_col, lattice)
        f_bc = bc_fn(f_stream, f_col, None)

        assert f_bc.shape == f.shape
