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
from operators.wetting import build_wetting_fn
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
        from operators.differential._gradient import compute_gradient

        out = compute_gradient(const_field, lattice.w, lattice.c, periodic_pad)
        assert out.shape == (NX, NY, 1, 2)

    def test_constant_field_zero_gradient(self, lattice, const_field, periodic_pad):
        from operators.differential._gradient import compute_gradient

        out = compute_gradient(const_field, lattice.w, lattice.c, periodic_pad)
        np.testing.assert_allclose(np.array(out), 0.0, atol=1e-5)

    def test_2d_input_accepted(self, lattice, periodic_pad):
        """Also accepts a bare (nx, ny) array."""
        from operators.differential._gradient import compute_gradient

        field_2d = jnp.ones((NX, NY))
        out = compute_gradient(field_2d, lattice.w, lattice.c, periodic_pad)
        assert out.shape == (NX, NY, 1, 2)

    def test_x_gradient_nonzero_for_x_varying_field(
        self,
        lattice,
        linear_x_field,
        periodic_pad,
    ):
        from operators.differential._gradient import compute_gradient

        out = compute_gradient(linear_x_field, lattice.w, lattice.c, periodic_pad)
        # gx should be nonzero almost everywhere (periodic wrap creates edge artefacts)
        gx = np.array(out[:, :, 0, 0])
        # Interior columns should have positive gx
        assert float(np.mean(np.abs(gx[1:-1, :]))) > 0.1

    def test_jittable(self, lattice, const_field, periodic_pad):
        from operators.differential._gradient import compute_gradient

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
        from operators.differential._laplacian import compute_laplacian

        out = compute_laplacian(const_field, lattice.w, periodic_pad)
        assert out.shape == (NX, NY, 1, 1)

    def test_constant_field_zero_laplacian(self, lattice, const_field, periodic_pad):
        from operators.differential._laplacian import compute_laplacian

        out = compute_laplacian(const_field, lattice.w, periodic_pad)
        np.testing.assert_allclose(np.array(out), 0.0, atol=1e-5)

    def test_2d_input_accepted(self, lattice, periodic_pad):
        from operators.differential._laplacian import compute_laplacian

        out = compute_laplacian(jnp.ones((NX, NY)), lattice.w, periodic_pad)
        assert out.shape == (NX, NY, 1, 1)

    def test_quadratic_field_nonzero_laplacian(self, lattice, periodic_pad):
        """f(i,j) = i² — Laplacian should be ~2 in the interior."""
        from operators.differential._laplacian import compute_laplacian

        xs = jnp.arange(NX, dtype=jnp.float32)
        field = (xs**2)[:, None, None, None] * jnp.ones((NX, NY, 1, 1))
        out = compute_laplacian(field, lattice.w, periodic_pad)
        # Interior (avoid periodic wrap artefacts at boundaries)
        lap_interior = np.array(out[2:-2, 2:-2, 0, 0])
        assert float(np.mean(np.abs(lap_interior))) > 0.5

    def test_jittable(self, lattice, const_field, periodic_pad):
        from operators.differential._laplacian import compute_laplacian

        jitted = jax.jit(compute_laplacian, static_argnames=("pad_mode",))
        out = jitted(const_field, lattice.w, pad_mode=tuple(periodic_pad))
        assert out.shape == (NX, NY, 1, 1)

    def test_registered_in_registry(self):
        from registry import get_operator_names

        assert "laplacian" in get_operator_names("differential")


# =====================================================================
# make_wetting_gradient
# =====================================================================


class TestBuildWettingGradient:
    """``build_wetting_gradient`` returns a closure with correct behaviour."""

    @pytest.fixture
    def wetting_params(self):
        return {
            "rho_l": 1.0,
            "rho_v": 0.1,
            "width": 4,
            "phi_l": 1.2,
            "phi_r": 1.2,
            "d_rho_l": 0.0,
            "d_rho_r": 0.0,
        }

    def _call_wetting(self, fn, grid, params):
        """Invoke the wetting closure with only dynamic params (static ones baked in)."""
        return fn(
            grid,
            params["phi_l"],
            params["phi_r"],
            params["d_rho_l"],
            params["d_rho_r"],
        )

    def test_returns_callable(self, lattice, periodic_pad, wetting_params):
        from operators.differential._gradient_wetting import build_wetting_gradient

        fn = build_wetting_gradient(
            lattice.w,
            lattice.c,
            periodic_pad,
            rho_l=wetting_params["rho_l"],
            rho_v=wetting_params["rho_v"],
            width=wetting_params["width"],
        )
        assert callable(fn)

    def test_output_shape(self, lattice, periodic_pad, wetting_params, const_field):
        from operators.differential._gradient_wetting import build_wetting_gradient

        fn = build_wetting_gradient(
            lattice.w,
            lattice.c,
            periodic_pad,
            rho_l=wetting_params["rho_l"],
            rho_v=wetting_params["rho_v"],
            width=wetting_params["width"],
        )
        out = self._call_wetting(fn, const_field, wetting_params)
        assert out.shape == (NX, NY, 1, 2)

    def test_differs_from_plain_gradient_on_nonuniform_field(
        self,
        lattice,
        wetting_params,
    ):
        """Wetting correction changes the gradient when a droplet interface meets the wall."""
        from operators.differential._gradient import compute_gradient
        from operators.differential._gradient_wetting import build_wetting_gradient

        # Pad modes matching bottom=wetting, top=bounce-back, left/right=periodic
        wetting_pad = ("wrap", "edge", "edge", "wrap")

        rho_l = wetting_params["rho_l"]
        rho_v = wetting_params["rho_v"]

        # Droplet on the bottom wall: liquid in the center, vapour at the sides.
        # The tanh along x creates an interface that intersects the bottom row,
        # which is exactly where the wetting modification acts.
        xs = jnp.linspace(-1, 1, NX)
        droplet_x = 0.5 * (rho_l + rho_v) + 0.5 * (rho_l - rho_v) * (
            jnp.tanh((xs + 0.4) / 0.15) - jnp.tanh((xs - 0.4) / 0.15) - 1.0
        )
        # Taper toward vapour at the top
        ys = jnp.linspace(0, 1, NY)
        taper_y = 0.5 * (1.0 + jnp.tanh((0.5 - ys) / 0.15))
        rho_2d = rho_v + (droplet_x[:, None] - rho_v) * taper_y[None, :]
        rho = rho_2d[:, :, None, None]

        plain = compute_gradient(rho, lattice.w, lattice.c, wetting_pad)
        wetting_fn = build_wetting_gradient(
            lattice.w,
            lattice.c,
            wetting_pad,
            bc_config={"bottom": "wetting", "top": "bounce-back",
                       "left": "periodic", "right": "periodic"},
            rho_l=rho_l,
            rho_v=rho_v,
            width=wetting_params["width"],
        )
        with_wetting = self._call_wetting(wetting_fn, rho, wetting_params)

        assert not jnp.allclose(plain, with_wetting, atol=1e-9)

    def test_deterministic_result(self, lattice, periodic_pad, wetting_params, const_field):
        from operators.differential._gradient_wetting import build_wetting_gradient

        fn = build_wetting_gradient(
            lattice.w,
            lattice.c,
            periodic_pad,
            rho_l=wetting_params["rho_l"],
            rho_v=wetting_params["rho_v"],
            width=wetting_params["width"],
        )
        out = self._call_wetting(fn, const_field, wetting_params)
        out2 = self._call_wetting(fn, const_field, wetting_params)
        np.testing.assert_array_equal(np.array(out), np.array(out2))

    def test_chemical_step_variant(self, lattice, periodic_pad, const_field):
        """build_wetting_gradient with resolved chemical-step wetting fields."""
        from operators.differential._gradient_wetting import build_wetting_gradient

        params_array = {
            "phi": [1.2, 1.4],
            "d_rho": [0.03, 0.07],
        }
        _resolve_wetting_fields = build_wetting_fn("resolve_wetting_fields")
        phi_l, phi_r, d_rho_l, d_rho_r = _resolve_wetting_fields(params_array, chemical_step=0)

        fn = build_wetting_gradient(
            lattice.w,
            lattice.c,
            periodic_pad,
            rho_l=1.0,
            rho_v=0.1,
            width=4,
        )
        out = fn(
            const_field,
            phi_l,
            phi_r,
            d_rho_l,
            d_rho_r,
        )
        assert out.shape == (NX, NY, 1, 2)

    def test_registered_in_registry(self):
        from registry import get_operator_names

        assert "gradient_wetting" in get_operator_names("differential")


# =====================================================================
# wetting helpers
# =====================================================================


class TestWettingUtil:
    """Tests for the refactored wetting utility architecture.

    Covers:
    - resolve_wetting_fields (scalar and chemical-step layouts)
    - Ghost-cell reconstruction correctness (_reconstruct_ghost_row)
    - Interface wetting modification (_apply_wetting_modification)
    - Per-edge application via build_wetting_applicator
    - Corner handling with periodic vs non-periodic perpendicular BCs
    """

    # --- resolve_wetting_fields -------------------------------------------

    def test_resolve_scalar_layout(self):

        p = {"phi_l": 1.2, "phi_r": 1.4, "d_rho_l": 0.05, "d_rho_r": 0.06}
        _resolve_wetting_fields = build_wetting_fn("resolve_wetting_fields")
        phi_l, phi_r, drho_l, drho_r = _resolve_wetting_fields(p)
        assert phi_l == 1.2
        assert phi_r == 1.4
        assert drho_l == 0.05
        assert drho_r == 0.06

    def test_resolve_array_layout_step0(self):

        p = {"phi": [1.2, 1.4], "d_rho": [0.03, 0.07]}
        _resolve_wetting_fields = build_wetting_fn("resolve_wetting_fields")
        phi_l, phi_r, _d_rho_l, _d_rho_r = _resolve_wetting_fields(p, chemical_step=0)
        assert phi_l == 1.2
        assert phi_r == 1.4

    def test_resolve_array_layout_step1(self):

        p = {"phi": [1.2, 1.4], "d_rho": [0.03, 0.07]}
        _resolve_wetting_fields = build_wetting_fn("resolve_wetting_fields")
        phi_l, phi_r, _d_rho_l, _d_rho_r = _resolve_wetting_fields(p, chemical_step=1)
        # step=1 swaps sides
        assert phi_l == 1.4
        assert phi_r == 1.2

    # --- Ghost-cell reconstruction ----------------------------------------

    def test_reconstruction_uses_d2q9_weights(self):
        """Ghost row should be a D2Q9-weighted average of interior neighbour."""
        from operators.wetting._ghost_reconstruction import _W_CARDINAL
        from operators.wetting._ghost_reconstruction import _W_DIAGONAL
        from operators.wetting._ghost_reconstruction import _W_TOTAL
        from operators.wetting._ghost_reconstruction import _reconstruct_ghost_row

        # 6 rows along the wall, 4 columns (2 interior + 2 ghost)
        arr = jnp.zeros((6, 4))
        # Fill the interior column next to ghost_idx=0 (column 1)
        vals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        arr = arr.at[:, 1].set(vals)

        out = _reconstruct_ghost_row(arr, ghost_idx=0, interior_offset=1,
                                     wrap_start=True, wrap_end=True)
        # Interior point at index 2: cardinal=3.0, diag_minus=2.0, diag_plus=4.0
        expected = (_W_CARDINAL * 3.0 + _W_DIAGONAL * (2.0 + 4.0)) / _W_TOTAL
        np.testing.assert_allclose(float(out[2, 0]), expected, atol=1e-6)

    def test_reconstruction_corner_periodic(self):
        """Start corner wraps to last row when periodic."""
        from operators.wetting._ghost_reconstruction import _W_CARDINAL
        from operators.wetting._ghost_reconstruction import _W_DIAGONAL
        from operators.wetting._ghost_reconstruction import _W_TOTAL
        from operators.wetting._ghost_reconstruction import _reconstruct_ghost_row

        arr = jnp.zeros((6, 4))
        vals = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        arr = arr.at[:, 1].set(vals)

        out = _reconstruct_ghost_row(arr, ghost_idx=0, interior_offset=1,
                                     wrap_start=True, wrap_end=True)
        # Start corner (index 0): wrap_start → uses arr[-1, 1] = 60.0
        expected_start = (_W_CARDINAL * 10.0 + _W_DIAGONAL * (60.0 + 20.0)) / _W_TOTAL
        np.testing.assert_allclose(float(out[0, 0]), expected_start, atol=1e-6)

        # End corner (index -1): wrap_end → uses arr[0, 1] = 10.0
        expected_end = (_W_CARDINAL * 60.0 + _W_DIAGONAL * (50.0 + 10.0)) / _W_TOTAL
        np.testing.assert_allclose(float(out[-1, 0]), expected_end, atol=1e-6)

    def test_reconstruction_corner_non_periodic(self):
        """Non-periodic corners mirror the adjacent interior value."""
        from operators.wetting._ghost_reconstruction import _W_CARDINAL
        from operators.wetting._ghost_reconstruction import _W_DIAGONAL
        from operators.wetting._ghost_reconstruction import _W_TOTAL
        from operators.wetting._ghost_reconstruction import _reconstruct_ghost_row

        arr = jnp.zeros((6, 4))
        vals = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        arr = arr.at[:, 1].set(vals)

        out = _reconstruct_ghost_row(arr, ghost_idx=0, interior_offset=1,
                                     wrap_start=False, wrap_end=False)
        # Start corner: non-periodic → uses arr[1, 1] = 20.0 instead of arr[-1, 1]
        expected_start = (_W_CARDINAL * 10.0 + _W_DIAGONAL * (20.0 + 20.0)) / _W_TOTAL
        np.testing.assert_allclose(float(out[0, 0]), expected_start, atol=1e-6)

        # End corner: non-periodic → uses arr[-2, 1] = 50.0 instead of arr[0, 1]
        expected_end = (_W_CARDINAL * 60.0 + _W_DIAGONAL * (50.0 + 50.0)) / _W_TOTAL
        np.testing.assert_allclose(float(out[-1, 0]), expected_end, atol=1e-6)

    # --- Wetting modification ---------------------------------------------

    def test_modification_in_interface_region(self):
        """Modification should only affect densities inside the interface band."""
        from operators.wetting._wetting_modification import _apply_wetting_modification

        rho_l, rho_v = 1.0, 0.1
        upper = 0.95 * rho_l + 0.05 * rho_v  # 0.955
        lower = 0.05 * rho_l + 0.95 * rho_v  # 0.145

        # Create a slice with values spanning the full density range
        n = 32
        edge = jnp.linspace(rho_v, rho_l, n)
        result = _apply_wetting_modification(edge, rho_l, rho_v,
                                             1.2, 1.2, 0.05, 0.05, 4)

        # Values clearly outside the interface should be unchanged
        outside_mask = (np.array(edge) >= upper) | (np.array(edge) <= lower)
        np.testing.assert_array_equal(
            np.array(result)[outside_mask],
            np.array(edge)[outside_mask],
        )

    def test_modification_clamps_to_bounds(self):
        """Modified values inside the interface should be clamped to density bounds."""
        from operators.wetting._wetting_modification import _apply_wetting_modification

        rho_l, rho_v = 1.0, 0.1
        upper = 0.95 * rho_l + 0.05 * rho_v
        lower = 0.05 * rho_l + 0.95 * rho_v

        edge = jnp.linspace(rho_v, rho_l, 32)
        # Use extreme phi values to force clamping
        result = _apply_wetting_modification(edge, rho_l, rho_v,
                                             10.0, 10.0, 0.0, 0.0, 4)

        # Identify which values were actually modified (inside the interface)
        edge_np = np.array(edge)
        result_np = np.array(result)
        modified_mask = ~np.isclose(edge_np, result_np)

        if np.any(modified_mask):
            # All modified values must be clamped within [lower, upper]
            assert float(np.max(result_np[modified_mask])) <= upper + 1e-6
            assert float(np.min(result_np[modified_mask])) >= lower - 1e-6

    # --- Per-edge application via build_wetting_applicator -----------------

    def test_bottom_wetting_changes_bottom_ghost_row(self):
        """Bottom-only wetting should modify the bottom ghost row."""
        bc = {"bottom": "wetting", "top": "bounce-back"}
        _build_wetting_applicator = build_wetting_fn("applicator")
        fn = _build_wetting_applicator(rho_l=1.0, rho_v=0.1, width=4,
                                      bc_config=bc)
        gp = jnp.ones((NX + 2, NY + 2)) * 0.5
        gp_out = fn(gp, 1.2, 1.3, 0.05, 0.05)
        # Bottom ghost row should have been modified
        bottom = np.array(gp_out[1:-1, 0])
        assert not np.allclose(bottom, 0.5)

    def test_top_wetting_only(self):
        """Top-only wetting should modify only the top ghost row."""
        bc = {"bottom": "bounce-back", "top": "wetting"}
        _build_wetting_applicator = build_wetting_fn("applicator")
        fn = _build_wetting_applicator(rho_l=1.0, rho_v=0.1, width=4,
                                       bc_config=bc)
        gp = jnp.ones((NX + 2, NY + 2)) * 0.5
        gp_out = fn(gp, 1.2, 1.3, 0.05, 0.05)
        # Bottom ghost row should be unchanged
        np.testing.assert_array_equal(np.array(gp_out[1:-1, 0]), 0.5)
        # Top ghost row should be modified
        top = np.array(gp_out[1:-1, -1])
        assert not np.allclose(top, 0.5)

    def test_left_right_wetting_uses_transpose(self):
        """Left/right wetting should modify the left/right ghost columns."""
        bc = {"left": "wetting", "right": "wetting",
              "bottom": "bounce-back", "top": "bounce-back"}
        _build_wetting_applicator = build_wetting_fn("applicator")
        fn = _build_wetting_applicator(rho_l=1.0, rho_v=0.1, width=4,
                                       bc_config=bc)
        gp = jnp.ones((NX + 2, NY + 2)) * 0.5
        gp_out = fn(gp, 1.2, 1.3, 0.05, 0.05)
        # Left and right ghost columns should be modified
        left_col = np.array(gp_out[0, 1:-1])
        right_col = np.array(gp_out[-1, 1:-1])
        assert not np.allclose(left_col, 0.5)
        assert not np.allclose(right_col, 0.5)
        # Top/bottom ghost rows should be unchanged
        np.testing.assert_array_equal(np.array(gp_out[1:-1, 0]), 0.5)
        np.testing.assert_array_equal(np.array(gp_out[1:-1, -1]), 0.5)

    def test_no_wetting_edges_leaves_array_unchanged(self):
        """An empty bc_config should leave the array entirely unchanged."""
        _build_wetting_applicator = build_wetting_fn("applicator")
        fn = _build_wetting_applicator(rho_l=1.0, rho_v=0.1, width=4,
                                       bc_config={})
        gp = jnp.ones((NX + 2, NY + 2)) * 0.5
        gp_out = fn(gp, 1.2, 1.3, 0.05, 0.05)
        np.testing.assert_array_equal(np.array(gp_out), np.array(gp))

    def test_corner_periodic_vs_non_periodic(self):
        """Perpendicular periodic BCs affect corner ghost-cell values."""
        """Perpendicular periodic BCs affect corner ghost-cell values."""
        from operators.wetting import build_wetting_fn

        # Periodic perpendicular (default for unspecified edges)
        _build_wetting_applicator = build_wetting_fn("applicator")
        fn_periodic = _build_wetting_applicator(rho_l=1.0, rho_v=0.1, width=4,
                                                bc_config={"bottom": "wetting"})

        # Non-periodic perpendicular (bounce-back on left/right)
        fn_nonperiodic = _build_wetting_applicator(
            rho_l=1.0, rho_v=0.1, width=4,
            bc_config={"bottom": "wetting", "left": "bounce-back",
                       "right": "bounce-back"},
        )

        # For a uniform field they happen to be the same; use a non-uniform
        # field to see the difference.
        gp2 = jnp.ones((NX + 2, NY + 2)) * 0.5
        gp2 = gp2.at[1, 1].set(0.8)   # break symmetry near start corner
        gp2 = gp2.at[-2, 1].set(0.2)  # break symmetry near end corner

        out_p2 = fn_periodic(gp2, 1.0, 1.0, 0.0, 0.0)
        out_np2 = fn_nonperiodic(gp2, 1.0, 1.0, 0.0, 0.0)
        # With asymmetric interior, periodic and non-periodic corners differ
        assert float(out_p2[0, 0]) != float(out_np2[0, 0]) or \
               float(out_p2[-1, 0]) != float(out_np2[-1, 0])
