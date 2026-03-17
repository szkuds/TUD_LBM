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

from registry import (
    OPERATOR_REGISTRY,
    get_operator_category,
    get_operator_names,
    get_operators,
    register_operator,
)

# Import all operator packages to trigger registration decorators.
# In production code, these are imported by the factories/setup modules.
import operators.collision  # noqa: F401
import operators.boundary  # noqa: F401
import operators.differential  # noqa: F401
import operators.equilibrium  # noqa: F401
import operators.streaming  # noqa: F401
import operators.macroscopic  # noqa: F401
import operators.force  # noqa: F401
import operators.initialise.factory  # noqa: F401
import operators.wetting  # noqa: F401
import setup.lattice  # noqa: F401

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
        del OPERATOR_REGISTRY["_test_kind:_test_fn"]

    def test_register_class_with_name_attr(self):
        @register_operator("_test_kind")
        class MyCls:
            name = "_test_cls"

        entry = OPERATOR_REGISTRY["_test_kind:_test_cls"]
        assert entry.target is MyCls
        del OPERATOR_REGISTRY["_test_kind:_test_cls"]

    def test_register_function_uses_dunder_name(self):
        @register_operator("_test_kind")
        def _test_auto_name():
            pass

        assert "_test_kind:_test_auto_name" in OPERATOR_REGISTRY
        del OPERATOR_REGISTRY["_test_kind:_test_auto_name"]

    def test_duplicate_raises(self):
        @register_operator("_test_dup", name="_dup")
        def fn1():
            pass

        with pytest.raises(ValueError, match="Duplicate"):

            @register_operator("_test_dup", name="_dup")
            def fn2():
                pass

        del OPERATOR_REGISTRY["_test_dup:_dup"]

    def test_metadata_stored(self):
        @register_operator("_test_meta", name="_meta", foo="bar")
        def fn():
            pass

        entry = OPERATOR_REGISTRY["_test_meta:_meta"]
        assert entry.metadata == {"foo": "bar"}
        del OPERATOR_REGISTRY["_test_meta:_meta"]


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
        assert names >= {"source_term_wb", "gravity_multiphase", "electric"}

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
    """get_collision_fn uses the registry correctly."""

    def test_get_bgk(self):
        from operators.collision.factory import build_collision_fn

        fn = build_collision_fn("bgk")
        assert callable(fn)

    def test_get_mrt(self):
        from operators.collision.factory import build_collision_fn

        fn = build_collision_fn("mrt")
        assert callable(fn)

    def test_unknown_raises(self):
        from operators.collision.factory import build_collision_fn

        with pytest.raises(ValueError, match="Unknown collision scheme"):
            build_collision_fn("nonexistent")
