"""Tests for operators/differential — compute_gradient, make_wetting_gradient, compute_laplacian.

Verifies:
- Shape correctness on small grids
- Mathematical correctness (linear field → constant gradient; quadratic → nonzero Laplacian)
- JIT-compatibility
- Wetting closure: shape and ghost-cell effect
- Registry entries
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from setup.lattice import build_lattice

NX, NY = 16, 16


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D2Q9")


@pytest.fixture(scope="module")
def periodic_pad():
    """All-periodic padding modes."""
    return ["wrap", "wrap", "wrap", "wrap"]


@pytest.fixture(scope="module")
def const_field():
    """Uniform field of 1.0, shape (NX, NY, 1, 1)."""
    return jnp.ones((NX, NY, 1, 1))


@pytest.fixture(scope="module")
def linear_x_field():
    """f(i,j) = i, so df/dx = 1, df/dy = 0 (periodic wrap)."""
    xs = jnp.arange(NX, dtype=jnp.float32)
    return jnp.broadcast_to(xs[:, None, None, None], (NX, NY, 1, 1))


# =====================================================================
# compute_gradient
# =====================================================================


class TestComputeGradient:
    """``compute_gradient`` output shape and basic maths."""

    def test_output_shape(self, lattice, const_field, periodic_pad):
        from operators.differential.gradient import compute_gradient

        out = compute_gradient(const_field, lattice.w, lattice.c, periodic_pad)
        assert out.shape == (NX, NY, 1, 2)

    def test_constant_field_zero_gradient(self, lattice, const_field, periodic_pad):
        from operators.differential.gradient import compute_gradient

        out = compute_gradient(const_field, lattice.w, lattice.c, periodic_pad)
        np.testing.assert_allclose(np.array(out), 0.0, atol=1e-5)

    def test_2d_input_accepted(self, lattice, periodic_pad):
        """Also accepts a bare (nx, ny) array."""
        from operators.differential.gradient import compute_gradient

        field_2d = jnp.ones((NX, NY))
        out = compute_gradient(field_2d, lattice.w, lattice.c, periodic_pad)
        assert out.shape == (NX, NY, 1, 2)

    def test_x_gradient_nonzero_for_x_varying_field(
        self, lattice, linear_x_field, periodic_pad
    ):
        from operators.differential.gradient import compute_gradient

        out = compute_gradient(linear_x_field, lattice.w, lattice.c, periodic_pad)
        # gx should be nonzero almost everywhere (periodic wrap creates edge artefacts)
        gx = np.array(out[:, :, 0, 0])
        # Interior columns should have positive gx
        assert float(np.mean(np.abs(gx[1:-1, :]))) > 0.1

    def test_jittable(self, lattice, const_field, periodic_pad):
        from operators.differential.gradient import compute_gradient

        jitted = jax.jit(compute_gradient, static_argnames=("pad_mode",))
        out = jitted(const_field, lattice.w, lattice.c, pad_mode=tuple(periodic_pad))
        assert out.shape == (NX, NY, 1, 2)

    def test_registered_in_registry(self):
        from registry import get_operator_names

        assert "gradient" in get_operator_names("differential")


# =====================================================================
# compute_laplacian
# =====================================================================


class TestComputeLaplacian:
    """``compute_laplacian`` output shape and basic maths."""

    def test_output_shape(self, lattice, const_field, periodic_pad):
        from operators.differential.laplacian import compute_laplacian

        out = compute_laplacian(const_field, lattice.w, periodic_pad)
        assert out.shape == (NX, NY, 1, 1)

    def test_constant_field_zero_laplacian(self, lattice, const_field, periodic_pad):
        from operators.differential.laplacian import compute_laplacian

        out = compute_laplacian(const_field, lattice.w, periodic_pad)
        np.testing.assert_allclose(np.array(out), 0.0, atol=1e-5)

    def test_2d_input_accepted(self, lattice, periodic_pad):
        from operators.differential.laplacian import compute_laplacian

        out = compute_laplacian(jnp.ones((NX, NY)), lattice.w, periodic_pad)
        assert out.shape == (NX, NY, 1, 1)

    def test_quadratic_field_nonzero_laplacian(self, lattice, periodic_pad):
        """f(i,j) = i² — Laplacian should be ~2 in the interior."""
        from operators.differential.laplacian import compute_laplacian

        xs = jnp.arange(NX, dtype=jnp.float32)
        field = (xs**2)[:, None, None, None] * jnp.ones((NX, NY, 1, 1))
        out = compute_laplacian(field, lattice.w, periodic_pad)
        # Interior (avoid periodic wrap artefacts at boundaries)
        lap_interior = np.array(out[2:-2, 2:-2, 0, 0])
        assert float(np.mean(np.abs(lap_interior))) > 0.5

    def test_jittable(self, lattice, const_field, periodic_pad):
        from operators.differential.laplacian import compute_laplacian

        jitted = jax.jit(compute_laplacian, static_argnames=("pad_mode",))
        out = jitted(const_field, lattice.w, pad_mode=tuple(periodic_pad))
        assert out.shape == (NX, NY, 1, 1)

    def test_registered_in_registry(self):
        from registry import get_operator_names

        assert "laplacian" in get_operator_names("differential")


# =====================================================================
# make_wetting_gradient
# =====================================================================


class TestMakeWettingGradient:
    """``make_wetting_gradient`` returns a jitted closure with correct behaviour."""

    @pytest.fixture
    def wetting_params(self):
        return {
            "rho_l": 1.0,
            "rho_v": 0.1,
            "width": 4,
            "phi_l": 1.2,
            "phi_r": 1.3,
            "d_rho_l": 0.05,
            "d_rho_r": 0.05,
        }

    def test_returns_callable(self, lattice, periodic_pad, wetting_params):
        from operators.differential.gradient import make_wetting_gradient

        fn = make_wetting_gradient(lattice.w, lattice.c, periodic_pad, wetting_params)
        assert callable(fn)

    def test_output_shape(self, lattice, periodic_pad, wetting_params, const_field):
        from operators.differential.gradient import make_wetting_gradient

        fn = make_wetting_gradient(lattice.w, lattice.c, periodic_pad, wetting_params)
        out = fn(const_field)
        assert out.shape == (NX, NY, 1, 2)

    def test_differs_from_plain_gradient_on_nonuniform_field(
        self, lattice, periodic_pad, wetting_params
    ):
        """Ghost-cell correction should change the result for a non-constant field."""
        from operators.differential.gradient import (
            compute_gradient,
            make_wetting_gradient,
        )

        # A non-trivial density field
        rho = jnp.linspace(0.3, 1.0, NX)[:, None, None, None] * jnp.ones((NX, NY, 1, 1))

        plain = compute_gradient(rho, lattice.w, lattice.c, periodic_pad)
        wetting_fn = make_wetting_gradient(
            lattice.w, lattice.c, periodic_pad, wetting_params
        )
        with_wetting = wetting_fn(rho)

        # They should not be identical (ghost cells differ)
        assert not jnp.allclose(plain, with_wetting, atol=1e-9)

    def test_jittable_result(self, lattice, periodic_pad, wetting_params, const_field):
        from operators.differential.gradient import make_wetting_gradient

        fn = make_wetting_gradient(lattice.w, lattice.c, periodic_pad, wetting_params)
        # The closure is already jitted; calling it again should use the compiled version
        out = fn(const_field)
        out2 = fn(const_field)
        np.testing.assert_array_equal(np.array(out), np.array(out2))

    def test_chemical_step_variant(self, lattice, periodic_pad, const_field):
        """make_wetting_gradient with chemical_step uses per-step wetting fields."""
        from operators.differential.gradient import make_wetting_gradient

        params_array = {
            "rho_l": 1.0,
            "rho_v": 0.1,
            "width": 4,
            "phi": [1.2, 1.4],
            "d_rho": [0.03, 0.07],
        }
        fn = make_wetting_gradient(
            lattice.w, lattice.c, periodic_pad, params_array, chemical_step=0
        )
        out = fn(const_field)
        assert out.shape == (NX, NY, 1, 2)


# =====================================================================
# wetting_util helpers
# =====================================================================


class TestWettingUtil:
    """resolve_wetting_fields and apply_wetting_to_all_edges."""

    def test_resolve_scalar_layout(self):
        from operators.wetting.wetting_util import resolve_wetting_fields

        p = {"phi_l": 1.2, "phi_r": 1.4, "d_rho_l": 0.05, "d_rho_r": 0.06}
        phi_l, phi_r, drho_l, drho_r = resolve_wetting_fields(p)
        assert phi_l == 1.2
        assert phi_r == 1.4
        assert drho_l == 0.05
        assert drho_r == 0.06

    def test_resolve_array_layout_step0(self):
        from operators.wetting.wetting_util import resolve_wetting_fields

        p = {"phi": [1.2, 1.4], "d_rho": [0.03, 0.07]}
        phi_l, phi_r, d_rho_l, d_rho_r = resolve_wetting_fields(p, chemical_step=0)
        assert phi_l == 1.2
        assert phi_r == 1.4

    def test_resolve_array_layout_step1(self):
        from operators.wetting.wetting_util import resolve_wetting_fields

        p = {"phi": [1.2, 1.4], "d_rho": [0.03, 0.07]}
        phi_l, phi_r, d_rho_l, d_rho_r = resolve_wetting_fields(p, chemical_step=1)
        # step=1 swaps sides
        assert phi_l == 1.4
        assert phi_r == 1.2

    def test_apply_wetting_changes_bottom_row(self):
        from operators.wetting.wetting_util import apply_wetting_to_all_edges

        gp = jnp.zeros((NX + 2, NY + 2))
        gp_out = apply_wetting_to_all_edges(gp, 1.0, 0.1, 1.2, 1.3, 0.05, 0.05, 4)
        # Bottom ghost row (index 0) interior columns should be nonzero
        bottom = np.array(gp_out[1:-1, 0])
        assert float(np.mean(np.abs(bottom))) > 0.0

    def test_apply_wetting_does_not_touch_top_row(self):
        from operators.wetting.wetting_util import apply_wetting_to_all_edges

        gp = jnp.zeros((NX + 2, NY + 2))
        gp_out = apply_wetting_to_all_edges(gp, 1.0, 0.1, 1.2, 1.3, 0.05, 0.05, 4)
        # Top ghost row (index -1) should still be zero
        np.testing.assert_array_equal(np.array(gp_out[:, -1]), 0.0)
