"""Tests for force operators — gravity and electric.

Tests for:
    - ``operators.force.gravity.build_gravity_force``
    - ``operators.force.gravity.compute_gravity_force``
    - ``operators.force.electric.*``
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from setup.lattice import build_lattice

NX, NY = 8, 8


@pytest.fixture
def lattice():
    return build_lattice("D2Q9")


# =====================================================================
# Gravity force
# =====================================================================


class TestGravityForce:
    """``build_gravity_force`` and ``compute_gravity_force`` are correct."""

    def test_template_shape(self):
        from operators.force.gravity import build_gravity_force

        template = build_gravity_force((NX, NY), force_g=0.001)
        assert template.shape == (NX, NY, 1, 2)

    def test_vertical_gravity(self):
        from operators.force.gravity import build_gravity_force

        template = build_gravity_force(
            (NX, NY),
            force_g=0.001,
            inclination_angle_deg=0.0,
        )
        # At 0 degrees: force_x = 0, force_y = force_g
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

    def test_inclined_gravity(self):
        from operators.force.gravity import build_gravity_force

        template = build_gravity_force(
            (NX, NY),
            force_g=0.001,
            inclination_angle_deg=90.0,
        )
        # At 90 degrees: force_x = -force_g, force_y ≈ 0
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

    def test_compute_gravity_force_shape(self):
        from operators.force.gravity import build_gravity_force
        from operators.force.gravity import compute_gravity_force

        template = build_gravity_force((NX, NY), force_g=0.001)
        rho = jnp.ones((NX, NY, 1, 1))
        force = compute_gravity_force(template, rho)
        assert force.shape == (NX, NY, 1, 2)

    def test_compute_gravity_force_value(self):
        from operators.force.gravity import build_gravity_force
        from operators.force.gravity import compute_gravity_force

        template = build_gravity_force((NX, NY), force_g=0.001)
        rho = jnp.ones((NX, NY, 1, 1)) * 2.0
        force = compute_gravity_force(template, rho)
        expected = -template * 2.0
        np.testing.assert_allclose(
            np.array(force),
            np.array(expected),
            atol=1e-12,
        )

    def test_jittable(self):
        from operators.force.gravity import build_gravity_force
        from operators.force.gravity import compute_gravity_force

        template = build_gravity_force((NX, NY), force_g=0.001)
        rho = jnp.ones((NX, NY, 1, 1))
        force = jax.jit(compute_gravity_force)(template, rho)
        assert force.shape == (NX, NY, 1, 2)


# =====================================================================
# Electric params
# =====================================================================


class TestElectricParams:
    """``build_electric_params`` creates a valid NamedTuple pytree."""

    def test_creation(self):
        from operators.force.electric import build_electric_params

        ep = build_electric_params(
            permittivity_liquid=80.0,
            permittivity_vapour=1.0,
            conductivity_liquid=0.01,
            conductivity_vapour=0.001,
        )
        assert ep.permittivity_liquid == 80.0
        assert ep.permittivity_vapour == 1.0

    def test_is_pytree(self):
        from operators.force.electric import build_electric_params

        ep = build_electric_params(
            permittivity_liquid=80.0,
            permittivity_vapour=1.0,
            conductivity_liquid=0.01,
            conductivity_vapour=0.001,
        )
        leaves, treedef = jax.tree_util.tree_flatten(ep)
        ep2 = treedef.unflatten(leaves)
        assert ep2.permittivity_liquid == ep.permittivity_liquid


# =====================================================================
# Electric init_hi
# =====================================================================


class TestInitHi:
    """``init_hi`` produces a valid initial electric distribution."""

    def test_shape(self, lattice):
        from operators.force.electric import init_hi

        hi = init_hi(NX, NY, lattice)
        assert hi.shape == (NX, NY, 9, 1)

    def test_linear_profile(self, lattice):
        from operators.force.electric import init_hi

        hi = init_hi(NX, NY, lattice, voltage_top=1.0, voltage_bottom=0.0)
        # Macroscopic potential should be linear from bottom to top
        potential = jnp.sum(hi, axis=2, keepdims=True)
        # Bottom row (y=0) should be ~0, top row (y=NY-1) should be ~1
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
    """``compute_electric_force`` returns correct shape and is jittable."""

    def test_shape(self, lattice):
        from operators.force.electric import build_electric_params
        from operators.force.electric import compute_electric_force
        from operators.force.electric import init_hi

        ep = build_electric_params(
            permittivity_liquid=80.0,
            permittivity_vapour=1.0,
            conductivity_liquid=0.01,
            conductivity_vapour=0.001,
        )
        rho = jnp.ones((NX, NY, 1, 1))
        hi = init_hi(NX, NY, lattice, voltage_top=1.0)

        force = compute_electric_force(rho, hi, ep, lattice)
        assert force.shape == (NX, NY, 1, 2)

    def test_zero_voltage_zero_force(self, lattice):
        from operators.force.electric import build_electric_params
        from operators.force.electric import compute_electric_force
        from operators.force.electric import init_hi

        ep = build_electric_params(
            permittivity_liquid=1.0,
            permittivity_vapour=1.0,
            conductivity_liquid=0.0,
            conductivity_vapour=0.0,
        )
        rho = jnp.ones((NX, NY, 1, 1))
        hi = init_hi(NX, NY, lattice, voltage_top=0.0, voltage_bottom=0.0)

        force = compute_electric_force(rho, hi, ep, lattice)
        np.testing.assert_allclose(np.array(force), 0.0, atol=1e-10)

    def test_jittable(self, lattice):
        from functools import partial
        from operators.force.electric import build_electric_params
        from operators.force.electric import compute_electric_force
        from operators.force.electric import init_hi

        ep = build_electric_params(
            permittivity_liquid=80.0,
            permittivity_vapour=1.0,
            conductivity_liquid=0.01,
            conductivity_vapour=0.001,
        )
        rho = jnp.ones((NX, NY, 1, 1))
        hi = init_hi(NX, NY, lattice, voltage_top=1.0)

        # Close over ep and lattice (contain Python scalars/strings)
        jitted = jax.jit(
            partial(compute_electric_force, electric_params=ep, lattice=lattice),
        )
        force = jitted(rho, hi)
        assert force.shape == (NX, NY, 1, 2)


# =====================================================================
# Update hi
# =====================================================================


class TestUpdateHi:
    """``update_hi`` advances the electric potential distribution."""

    def test_shape(self, lattice):
        from operators.force.electric import build_electric_params
        from operators.force.electric import init_hi
        from operators.force.electric import update_hi
        from operators.streaming._streaming import stream

        ep = build_electric_params(
            permittivity_liquid=80.0,
            permittivity_vapour=1.0,
            conductivity_liquid=0.01,
            conductivity_vapour=0.001,
            voltage_top=1.0,
            voltage_bottom=0.0,
        )
        rho = jnp.ones((NX, NY, 1, 1))
        hi = init_hi(NX, NY, lattice, voltage_top=1.0)

        hi_new = update_hi(hi, rho, ep, lattice, stream)
        assert hi_new.shape == hi.shape
