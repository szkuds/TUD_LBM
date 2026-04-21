"""Tests for initialisation operators — pure functions.

Tests for all ``operators.initialise`` functions:
    - Shape correctness on small grids
    - Mass conservation (density sums correctly)
    - Factory registry lookup
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from operators.initialise import build_initialise_fn
from registry import get_operators
from setup.lattice import build_lattice

NX, NY = 16, 16


@pytest.fixture
def lattice():
    return build_lattice("D2Q9")


# =====================================================================
# Factory
# =====================================================================


class TestInitFactory:
    """``build_initialise_fn`` returns the correct callable."""

    def test_known_type(self):
        fn = build_initialise_fn("standard")
        assert callable(fn)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown initialise scheme"):
            build_initialise_fn("nonexistent_type")

    def test_all_registry_entries_callable(self):
        init_ops = get_operators("initialise")
        for name, entry in init_ops.items():
            assert callable(entry.target), f"{name} is not callable"


# =====================================================================
# Standard
# =====================================================================


class TestInitStandard:
    """``init_standard`` produces correct shapes and densities."""

    def test_shape(self, lattice):
        f = build_initialise_fn("standard")(NX, NY, lattice)
        assert f.shape == (NX, NY, 9, 1)

    def test_density_default(self, lattice):
        f = build_initialise_fn("standard")(NX, NY, lattice, density=1.0)
        rho = jnp.sum(f, axis=2, keepdims=True)
        np.testing.assert_allclose(np.array(rho), 1.0, atol=1e-12)

    def test_density_custom(self, lattice):
        f = build_initialise_fn("standard")(NX, NY, lattice, density=2.5)
        rho = jnp.sum(f, axis=2, keepdims=True)
        np.testing.assert_allclose(np.array(rho), 2.5, atol=1e-12)

    def test_jittable(self, lattice):
        from functools import partial

        fn = build_initialise_fn("standard")
        # lattice contains string (name) — close over it; nx, ny are static
        jitted = jax.jit(partial(fn, lattice=lattice), static_argnums=(0, 1))
        f = jitted(NX, NY)
        assert f.shape == (NX, NY, 9, 1)


# =====================================================================
# Multiphase initialisers — shape tests
# =====================================================================


_MULTIPHASE_TYPES = [
    "multiphase_bubbles",
]


class TestMultiphaseInitShape:
    """All multiphase initialisers produce (nx, ny, q, 1) arrays."""

    @pytest.mark.parametrize("init_type", _MULTIPHASE_TYPES)
    def test_shape(self, lattice, init_type):
        fn = build_initialise_fn(init_type)
        f = fn(NX, NY, lattice, rho_l=1.0, rho_v=0.33, interface_width=4, centres=[[0.5, 0.5]], radii=[0.2])
        assert f.shape == (NX, NY, 9, 1)

    @pytest.mark.parametrize("init_type", _MULTIPHASE_TYPES)
    def test_density_range(self, lattice, init_type):
        """Density should be between rho_v and rho_l everywhere."""
        fn = build_initialise_fn(init_type)
        f = fn(NX, NY, lattice, rho_l=1.0, rho_v=0.33, interface_width=4, centres=[[0.5, 0.5]], radii=[0.2])
        rho = jnp.sum(f, axis=2, keepdims=True)
        assert float(jnp.min(rho)) >= 0.33 - 0.01
        assert float(jnp.max(rho)) <= 1.0 + 0.01


# =====================================================================
# Wetting initialisers
# =====================================================================


_WETTING_TYPES = ["wetting", "wetting_chem_step"]


class TestWettingInitShape:
    """Wetting initialisers produce correct shapes."""

    @pytest.mark.parametrize("init_type", _WETTING_TYPES)
    def test_shape(self, lattice, init_type):
        fn = build_initialise_fn(init_type)
        f = fn(NX, NY, lattice, rho_l=1.0, rho_v=0.33, interface_width=4)
        assert f.shape == (NX, NY, 9, 1)


# =====================================================================
# Mass conservation for bubble
# =====================================================================


class TestMassConservation:
    """Total mass is conserved (equals sum of rho over the domain)."""

    def test_bubble_mass_positive(self, lattice):
        fn = build_initialise_fn("multiphase_bubbles")
        f = fn(32, 32, lattice, rho_l=1.0, rho_v=0.33, interface_width=4, centres=[[0.5, 0.5]], radii=[0.2])
        rho = jnp.sum(f, axis=2, keepdims=True)
        total_mass = float(jnp.sum(rho))
        # Mass must be positive and between rho_v * N and rho_l * N
        assert total_mass > 0.33 * 32 * 32
        assert total_mass < 1.0 * 32 * 32

    def test_droplet_mass_positive(self, lattice):
        fn = build_initialise_fn("multiphase_bubbles")
        f = fn(
            32,
            32,
            lattice,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
            centres=[[0.5, 0.5]],
            radii=[0.2],
            dispersed="liquid",
        )
        rho = jnp.sum(f, axis=2, keepdims=True)
        total_mass = float(jnp.sum(rho))
        assert total_mass > 0.33 * 32 * 32
        assert total_mass < 1.0 * 32 * 32


# =====================================================================
# General multiphase bubbles
# =====================================================================


class TestMultiphaseBubbles:
    """``init_multiphase_bubbles`` accepts multiple centres/radii."""

    def test_multiple_inclusions(self, lattice):
        fn = build_initialise_fn("multiphase_bubbles")
        f = fn(
            32,
            32,
            lattice,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
            centres=[[0.25, 0.5], [0.75, 0.5]],
            radii=[0.12, 0.12],
        )
        assert f.shape == (32, 32, 9, 1)

    def test_invalid_dispersed_raises(self, lattice):
        fn = build_initialise_fn("multiphase_bubbles")
        with pytest.raises(ValueError, match="dispersed"):
            fn(
                32,
                32,
                lattice,
                rho_l=1.0,
                rho_v=0.33,
                interface_width=4,
                centres=[[0.5, 0.5]],
                radii=[0.2],
                dispersed="invalid",
            )

    def test_mismatched_centres_and_radii_raises(self, lattice):
        fn = build_initialise_fn("multiphase_bubbles")
        with pytest.raises(ValueError, match="same length"):
            fn(32, 32, lattice, rho_l=1.0, rho_v=0.33, interface_width=4, centres=[[0.5, 0.5]], radii=[0.2, 0.1])

    def test_empty_centres_raises(self, lattice):
        fn = build_initialise_fn("multiphase_bubbles")
        with pytest.raises(ValueError, match="non-empty"):
            fn(32, 32, lattice, rho_l=1.0, rho_v=0.33, interface_width=4, centres=[], radii=[])

    def test_liquid_dispersed_mode(self, lattice):
        fn = build_initialise_fn("multiphase_bubbles")
        f = fn(
            32,
            32,
            lattice,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
            centres=[[0.5, 0.5]],
            radii=[0.2],
            dispersed="liquid",
        )
        assert f.shape == (32, 32, 9, 1)
