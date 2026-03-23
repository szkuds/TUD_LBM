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

from setup.lattice import build_lattice
from operators.initialise.factory import get_init_fn
from registry import get_operators

NX, NY = 16, 16


@pytest.fixture
def lattice():
    return build_lattice("D2Q9")


# =====================================================================
# Factory
# =====================================================================


class TestInitFactory:
    """``get_init_fn`` returns the correct callable."""

    def test_known_type(self):
        fn = get_init_fn("standard")
        assert callable(fn)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown init_type"):
            get_init_fn("nonexistent_type")

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
        f = get_init_fn("standard")(NX, NY, lattice)
        assert f.shape == (NX, NY, 9, 1)

    def test_density_default(self, lattice):
        f = get_init_fn("standard")(NX, NY, lattice, density=1.0)
        rho = jnp.sum(f, axis=2, keepdims=True)
        np.testing.assert_allclose(np.array(rho), 1.0, atol=1e-12)

    def test_density_custom(self, lattice):
        f = get_init_fn("standard")(NX, NY, lattice, density=2.5)
        rho = jnp.sum(f, axis=2, keepdims=True)
        np.testing.assert_allclose(np.array(rho), 2.5, atol=1e-12)

    def test_jittable(self, lattice):
        from functools import partial

        fn = get_init_fn("standard")
        # lattice contains string (name) — close over it; nx, ny are static
        jitted = jax.jit(partial(fn, lattice=lattice), static_argnums=(0, 1))
        f = jitted(NX, NY)
        assert f.shape == (NX, NY, 9, 1)


# =====================================================================
# Multiphase initialisers — shape tests
# =====================================================================


_MULTIPHASE_TYPES = [
    "multiphase_bubble",
    "multiphase_bubble_bot",
    "multiphase_bubble_bubble",
    "multiphase_droplet",
    "multiphase_droplet_top",
    "multiphase_droplet_variable_radius",
    "multiphase_lateral_bubble",
]


class TestMultiphaseInitShape:
    """All multiphase initialisers produce (nx, ny, q, 1) arrays."""

    @pytest.mark.parametrize("init_type", _MULTIPHASE_TYPES)
    def test_shape(self, lattice, init_type):
        fn = get_init_fn(init_type)
        f = fn(NX, NY, lattice, rho_l=1.0, rho_v=0.33, interface_width=4)
        assert f.shape == (NX, NY, 9, 1)

    @pytest.mark.parametrize("init_type", _MULTIPHASE_TYPES)
    def test_density_range(self, lattice, init_type):
        """Density should be between rho_v and rho_l everywhere."""
        fn = get_init_fn(init_type)
        f = fn(NX, NY, lattice, rho_l=1.0, rho_v=0.33, interface_width=4)
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
        fn = get_init_fn(init_type)
        f = fn(NX, NY, lattice, rho_l=1.0, rho_v=0.33, interface_width=4)
        assert f.shape == (NX, NY, 9, 1)


# =====================================================================
# Mass conservation for bubble
# =====================================================================


class TestMassConservation:
    """Total mass is conserved (equals sum of rho over the domain)."""

    def test_bubble_mass_positive(self, lattice):
        fn = get_init_fn("multiphase_bubble")
        f = fn(32, 32, lattice, rho_l=1.0, rho_v=0.33, interface_width=4)
        rho = jnp.sum(f, axis=2, keepdims=True)
        total_mass = float(jnp.sum(rho))
        # Mass must be positive and between rho_v * N and rho_l * N
        assert total_mass > 0.33 * 32 * 32
        assert total_mass < 1.0 * 32 * 32

    def test_droplet_mass_positive(self, lattice):
        fn = get_init_fn("multiphase_droplet")
        f = fn(32, 32, lattice, rho_l=1.0, rho_v=0.33, interface_width=4)
        rho = jnp.sum(f, axis=2, keepdims=True)
        total_mass = float(jnp.sum(rho))
        assert total_mass > 0.33 * 32 * 32
        assert total_mass < 1.0 * 32 * 32


# =====================================================================
# Variable radius droplet
# =====================================================================


class TestVariableRadius:
    """``init_multiphase_droplet_variable_radius`` respects custom radius."""

    def test_custom_radius(self, lattice):
        fn = get_init_fn("multiphase_droplet_variable_radius")
        f = fn(32, 32, lattice, rho_l=1.0, rho_v=0.33, interface_width=4, radius=5.0)
        assert f.shape == (32, 32, 9, 1)

    def test_default_radius(self, lattice):
        fn = get_init_fn("multiphase_droplet_variable_radius")
        f = fn(32, 32, lattice, rho_l=1.0, rho_v=0.33, interface_width=4)
        assert f.shape == (32, 32, 9, 1)
