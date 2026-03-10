"""Tests for the global operator registry."""

import pytest

from app_setup.registry import OPERATOR_REGISTRY, get_operators, register_operator, OperatorEntry


# Ensure all simulation_operators are imported (triggers registration)
import simulation_operators  # noqa: F401
import simulation_type  # noqa: F401
import update_timestep  # noqa: F401


class TestOperatorRegistry:
    """Tests for the global OPERATOR_REGISTRY dict."""

    def test_registry_is_populated(self):
        """After importing simulation_operators, the registry should contain entries."""
        assert len(OPERATOR_REGISTRY) > 0

    def test_registry_entries_are_operator_entry(self):
        for key, entry in OPERATOR_REGISTRY.items():
            assert isinstance(entry, OperatorEntry)
            assert isinstance(entry.name, str)
            assert isinstance(entry.kind, str)
            assert entry.cls is not None

    def test_registry_key_format(self):
        """Keys should be '{kind}:{name}'."""
        for key, entry in OPERATOR_REGISTRY.items():
            assert key == f"{entry.kind}:{entry.name}"


class TestGetOperators:
    """Tests for the get_operators() function."""

    def test_get_collision_operators(self):
        collision_ops = get_operators("collision_models")
        assert "bgk" in collision_ops
        assert "mrt" in collision_ops

    def test_get_macroscopic_operators(self):
        macroscopic_ops = get_operators("macroscopic")
        assert "standard" in macroscopic_ops
        assert "double-well" in macroscopic_ops
        assert "carnahan-starling" in macroscopic_ops

    def test_get_equilibrium_operators(self):
        eq_ops = get_operators("equilibrium")
        assert "wb" in eq_ops

    def test_get_stream_operators(self):
        stream_ops = get_operators("stream")
        assert "standard" in stream_ops

    def test_get_boundary_condition_operators(self):
        bc_ops = get_operators("boundary_condition")
        assert "standard" in bc_ops

    def test_get_initialise_operators(self):
        init_ops = get_operators("initialise")
        assert "standard" in init_ops

    def test_get_differential_operators(self):
        diff_ops = get_operators("differential")
        assert "gradient" in diff_ops
        assert "laplacian" in diff_ops

    def test_get_force_operators(self):
        force_ops = get_operators("force")
        assert "gravity_multiphase" in force_ops
        assert "electric" in force_ops
        assert "composite" in force_ops
        assert "source_term" in force_ops

    def test_get_wetting_operators(self):
        wetting_ops = get_operators("wetting")
        assert "contact_angle" in wetting_ops
        assert "contact_line_location" in wetting_ops

    def test_get_simulation_operators(self):
        sim_ops = get_operators("simulation_type")
        assert "single_phase" in sim_ops
        assert "multiphase" in sim_ops

    def test_get_update_operators(self):
        update_ops = get_operators("update_timestep")
        assert "single_phase" in update_ops
        assert "multiphase" in update_ops
        assert "multiphase_hysteresis" in update_ops

    def test_get_nonexistent_kind_returns_empty(self):
        ops = get_operators("nonexistent_kind_xyz")
        assert ops == {}

    def test_operator_entry_cls_is_correct(self):
        from simulation_operators.collision_models import CollisionBGK, CollisionMRT
        collision_ops = get_operators("collision_models")
        assert collision_ops["bgk"].cls is CollisionBGK
        assert collision_ops["mrt"].cls is CollisionMRT


class TestRegisterOperator:
    """Tests for the @register_operator decorator."""

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="must define a class attribute 'name'"):
            @register_operator("test_kind")
            class BadOperator:
                pass

    def test_duplicate_registration_raises(self):
        # First registration should succeed
        @register_operator("_test_duplicate")
        class Op1:
            name = "_dup_test"

        # Second registration with same kind:name should fail
        with pytest.raises(ValueError, match="Duplicate operator registration"):
            @register_operator("_test_duplicate")
            class Op2:
                name = "_dup_test"

        # Cleanup
        key = "_test_duplicate:_dup_test"
        OPERATOR_REGISTRY.pop(key, None)

    def test_successful_registration(self):
        @register_operator("_test_reg")
        class TestOp:
            name = "_test_op"

        key = "_test_reg:_test_op"
        assert key in OPERATOR_REGISTRY
        assert OPERATOR_REGISTRY[key].cls is TestOp
        assert OPERATOR_REGISTRY[key].name == "_test_op"
        assert OPERATOR_REGISTRY[key].kind == "_test_reg"

        # Cleanup
        OPERATOR_REGISTRY.pop(key, None)


class TestRegistryIntegration:
    """Integration tests: verify registry-based lookups produce usable classes."""

    def test_collision_bgk_class_is_callable(self):
        collision_ops = get_operators("collision_models")
        bgk_cls = collision_ops["bgk"].cls
        assert callable(bgk_cls)

    def test_all_collision_names_are_strings(self):
        collision_ops = get_operators("collision_models")
        for name in collision_ops:
            assert isinstance(name, str)

    def test_sorted_collision_names(self):
        collision_ops = get_operators("collision_models")
        names = sorted(collision_ops.keys())
        assert isinstance(names, list)
        assert len(names) >= 2  # at least bgk, mrt


