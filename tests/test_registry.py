"""Tests for the central operator registry.

Tests for:
    - ``registry.register_operator`` (class + function registration)
    - ``registry.get_operators`` / ``get_operator_names`` / ``get_operator_category``
    - Duplicate detection
    - Convenience decorators
    - All expected operator kinds are populated after importing operators
"""

from __future__ import annotations
import pytest
import operators.boundary

# Import all operator packages to trigger registration decorators.
# In production code, these are imported by the factories/setup modules.
import operators.collision
import operators.differential
import operators.equilibrium
import operators.force
import operators.initialise
import operators.macroscopic
import operators.streaming
import operators.wetting  # noqa: F401
import setup.lattice  # noqa: F401
from registry import OPERATOR_REGISTRY
from registry import get_operator_category
from registry import get_operator_names
from registry import get_operators
from registry import register_operator
from registry import unregister_operator

# =====================================================================
# Registration mechanics
# =====================================================================


class TestRegisterOperator:
    """Core registration decorator tests."""

    def test_register_function_with_explicit_name(self):
        @register_operator("_test_kind", name="_test_fn")
        def my_fn():
            pass

        entry = OPERATOR_REGISTRY["_test_kind:_test_fn"]
        assert entry.name == "_test_fn"
        assert entry.kind == "_test_kind"
        assert entry.target is my_fn
        # cleanup
        unregister_operator("_test_kind", "_test_fn")

    def test_register_class_with_name_attr(self):
        @register_operator("_test_kind")
        class MyCls:
            name = "_test_cls"

        entry = OPERATOR_REGISTRY["_test_kind:_test_cls"]
        assert entry.target is MyCls
        unregister_operator("_test_kind", "_test_cls")

    def test_register_function_uses_dunder_name(self):
        @register_operator("_test_kind")
        def _test_auto_name():
            pass

        assert "_test_kind:_test_auto_name" in OPERATOR_REGISTRY
        unregister_operator("_test_kind", "_test_auto_name")

    def test_duplicate_raises(self):
        @register_operator("_test_dup", name="_dup")
        def fn1():
            pass

        with pytest.raises(ValueError, match="Duplicate"):

            @register_operator("_test_dup", name="_dup")
            def fn2():
                pass

        unregister_operator("_test_dup", "_dup")

    def test_metadata_stored(self):
        @register_operator("_test_meta", name="_meta", foo="bar")
        def fn():
            pass

        entry = OPERATOR_REGISTRY["_test_meta:_meta"]
        assert entry.metadata == {"foo": "bar"}
        unregister_operator("_test_meta", "_meta")


# =====================================================================
# Query helpers
# =====================================================================


class TestQueryHelpers:
    """get_operators / get_operator_names / get_operator_category."""

    def test_get_operators_returns_dict(self):
        ops = get_operators("collision_models")
        assert isinstance(ops, dict)
        assert "bgk" in ops
        assert "mrt" in ops

    def test_get_operator_names(self):
        names = get_operator_names("collision_models")
        assert "bgk" in names
        assert "mrt" in names

    def test_get_operator_category(self):
        cats = get_operator_category()
        assert "collision_models" in cats
        assert "lattice" in cats

    def test_empty_kind_returns_empty_dict(self):
        ops = get_operators("nonexistent_kind_xyz")
        assert ops == {}


# =====================================================================
# All expected kinds are populated
# =====================================================================


class TestRegistryPopulated:
    """After importing all operator packages, expected entries exist."""

    def test_collision_models_registered(self):
        names = get_operator_names("collision_models")
        assert names >= {"bgk", "mrt"}

    def test_boundary_conditions_registered(self):
        names = get_operator_names("boundary_condition")
        assert names >= {"bounce-back", "periodic", "symmetry"}

    def test_equilibrium_registered(self):
        names = get_operator_names("equilibrium")
        assert "wb" in names

    def test_streaming_registered(self):
        names = get_operator_names("stream")
        assert "standard" in names

    def test_macroscopic_registered(self):
        names = get_operator_names("macroscopic")
        assert names >= {"standard", "double-well"}

    def test_force_registered(self):
        names = get_operator_names("force")
        assert names >= {"source_term_wb", "gravity_force", "electric_force"}

    def test_initialise_registered(self):
        names = get_operator_names("initialise")
        expected = {
            "standard",
            "multiphase_bubble",
            "multiphase_bubble_bot",
            "multiphase_bubble_bubble",
            "multiphase_droplet",
            "multiphase_droplet_top",
            "multiphase_droplet_variable_radius",
            "multiphase_lateral_bubble",
            "wetting",
            "wetting_chem_step",
            "init_from_file",
        }
        assert names >= expected

    def test_wetting_registered(self):
        names = get_operator_names("wetting")
        assert names >= {"contact_angle", "contact_line_location", "hysteresis"}

    def test_differential_registered(self):
        names = get_operator_names("differential")
        assert names >= {"gradient", "laplacian"}

    def test_lattice_registered(self):
        names = get_operator_names("lattice")
        assert "D2Q9" in names


# =====================================================================
# Lattice via registry
# =====================================================================


class TestLatticeViaRegistry:
    """build_lattice uses the registry correctly."""

    def test_build_lattice_d2q9(self):
        from setup.lattice import build_lattice

        lat = build_lattice("D2Q9")
        assert lat.d == 2
        assert lat.q == 9

    def test_build_lattice_case_insensitive(self):
        from setup.lattice import build_lattice

        lat = build_lattice("d2q9")
        assert lat.d == 2

    def test_build_lattice_unsupported_raises(self):
        from setup.lattice import build_lattice

        with pytest.raises(ValueError, match="Unsupported lattice type"):
            build_lattice("D1Q3")

    def test_all_lattices_in_registry(self):
        names = get_operator_names("lattice")
        assert len(names) >= 1
        assert "D2Q9" in names


# =====================================================================
# Collision factory via registry
# =====================================================================


class TestCollisionFactoryViaRegistry:
    """build_collision_fn uses the registry correctly."""

    def test_get_bgk(self):
        from operators.collision import build_collision_fn

        fn = build_collision_fn("bgk")
        assert callable(fn)

    def test_get_mrt(self):
        from operators.collision import build_collision_fn

        fn = build_collision_fn("mrt")
        assert callable(fn)

    def test_unknown_raises(self):
        from operators.collision import build_collision_fn

        with pytest.raises(ValueError, match="Unknown collision"):
            build_collision_fn("nonexistent")


# =====================================================================
# End-to-end "no list to edit" acceptance tests
# =====================================================================


class TestDummyOperatorAutoExposure:
    """Prove that a decorator-only registration is visible end-to-end.

    No operator list, factory switch-case, or config allow-list is
    edited — the decorator alone is sufficient.
    """

    def test_new_dummy_collision_appears_without_list_edit(self):
        """Register a dummy collision model and verify it's discoverable."""
        from registry import collision_model

        try:

            @collision_model(name="_dummy_test_col")
            def _dummy_collide(f, feq, tau, source=None):
                return f  # identity — good enough for a test

            ops = get_operators("collision_models")
            assert "_dummy_test_col" in ops
            assert ops["_dummy_test_col"].target is _dummy_collide

            # Also visible via get_operator_names
            names = get_operator_names("collision_models")
            assert "_dummy_test_col" in names

            # The factory can resolve it
            from operators.collision import build_collision_fn

            fn = build_collision_fn("_dummy_test_col")
            assert fn is _dummy_collide
        finally:
            unregister_operator("collision_models", "_dummy_test_col")

    def test_new_dummy_force_appears_without_list_edit(self):
        """Register a dummy force builder and verify it's discoverable."""
        from registry import force_model

        try:

            @force_model(name="_dummy_test_force", result_field="gravity_template")
            def _dummy_force_builder(grid_shape=(4, 4)):
                import jax.numpy as jnp

                return jnp.zeros((*grid_shape, 1, 2))

            ops = get_operators("force")
            assert "_dummy_test_force" in ops
            assert ops["_dummy_test_force"].target is _dummy_force_builder

            # Metadata carries result_field
            assert ops["_dummy_test_force"].metadata["result_field"] == "gravity_template"
        finally:
            unregister_operator("force", "_dummy_test_force")

    def test_new_dummy_boundary_appears_without_list_edit(self):
        """Register a dummy boundary condition and verify it's discoverable."""
        from registry import boundary_condition

        try:

            @boundary_condition(name="_dummy_test_bc")
            def _dummy_bc(f, **kwargs):
                return f

            ops = get_operators("boundary_condition")
            assert "_dummy_test_bc" in ops
            assert ops["_dummy_test_bc"].target is _dummy_bc
        finally:
            unregister_operator("boundary_condition", "_dummy_test_bc")

    def test_unregister_removes_from_all_indices(self):
        """unregister_operator cleans both OPERATOR_REGISTRY and _KIND_INDEX."""
        from registry import _KIND_INDEX

        @register_operator("_test_cleanup", name="_cleanup_target")
        def _fn():
            pass

        assert "_test_cleanup:_cleanup_target" in OPERATOR_REGISTRY
        assert "_cleanup_target" in _KIND_INDEX.get("_test_cleanup", {})

        unregister_operator("_test_cleanup", "_cleanup_target")

        assert "_test_cleanup:_cleanup_target" not in OPERATOR_REGISTRY
        assert "_cleanup_target" not in _KIND_INDEX.get("_test_cleanup", {})
