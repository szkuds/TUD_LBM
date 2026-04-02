"""TDD tests explaining unified operator factory refactoring.

This test module documents the desired behavior of a unified, generic
operator factory that serves all operator kinds (collision, streaming,
equilibrium, macroscopic, etc).

## Architecture Decision

PROBLEM: Current implementation has 6-8 separate factory.py files with
nearly identical code (~40 lines each). This is boilerplate duplication.

SOLUTION: Move to unified factory with type-safe wrappers in __init__.py:
  - src/operators/factory.py: Single generic build_operator(kind, scheme)
  - src/operators/{kind}/__init__.py: Thin wrappers for type safety

BENEFITS:
  ✓ DRY: Single factory implementation
  ✓ Type-safe: Wrappers provide protocol return types for type-checkers
  ✓ Extensible: Add new operator kind → add 5-line wrapper
  ✓ Clear: Registry logic in one place, not scattered across files

## Test Structure

Tests are organized by what they explain:
  1. Generic factory finds operators in registry
  2. Type-safe wrappers delegate to generic factory
  3. Error handling is consistent
  4. All existing operator types still work
"""

# Ensure src/ is importable
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from registry import get_operators
from registry import register_operator
from registry import unregister_operator

# ── PART 1: Generic Factory Tests ─────────────────────────────────────────


class TestGenericFactory:
    """Generic factory resolves scheme names to operators via registry.

    The generic factory is the SINGLE SOURCE OF TRUTH for operator
    lookup. It takes (kind, scheme) and returns the registered operator.
    All specialized factories (build_collision_fn, etc) delegate to this.
    """

    def test_generic_factory_resolves_registered_operator(self):
        """Generic factory returns the correct operator from registry.

        This test documents:
          build_operator("collision_models", "bgk")
            ↓ (looks up in registry)
            ↓ (finds OperatorEntry with target=collide_bgk)
            ↓ (returns the function)
          collide_bgk ✓
        """
        from operators.collision._bgk import collide_bgk
        from operators.factory import build_operator

        result = build_operator("collision_models", "bgk")
        assert result is collide_bgk

    def test_generic_factory_raises_on_unknown_kind(self):
        """Generic factory raises ValueError for unknown operator kind.

        This test documents error handling when the operator kind
        itself is not registered (e.g., "unknown_operator_type").
        """
        from operators.factory import build_operator

        with pytest.raises(ValueError, match="No operators registered"):
            build_operator("nonexistent_kind", "some_scheme")

    def test_generic_factory_raises_on_unknown_scheme(self):
        """Generic factory raises ValueError for unknown scheme within a kind.

        This test documents error handling when the scheme is not
        registered under the given kind (e.g., "collision_models:invalid").
        """
        from operators.factory import build_operator

        with pytest.raises(ValueError, match=r"Unknown.*scheme"):
            build_operator("collision_models", "invalid_scheme_xyz")

    def test_generic_factory_error_message_lists_valid_schemes(self):
        """Error message includes available schemes for debugging.

        When a scheme is unknown, the error should help the developer
        by listing what schemes ARE available. This is helpful UX.
        """
        from operators.factory import build_operator

        with pytest.raises(ValueError) as exc_info:
            build_operator("collision_models", "not_a_real_scheme")

        error_msg = str(exc_info.value)
        # Should mention what schemes are actually available
        assert "bgk" in error_msg or "Valid" in error_msg


# ── PART 2: Type-Safe Wrapper Tests ──────────────────────────────────


class TestTypesSafeCollisionWrapper:
    """Type-safe wrapper for collision operators.

    The wrapper in operators/collision/__init__.py:
      1. Delegates to generic factory (no logic duplication)
      2. Returns strongly-typed CollisionOperator (for IDE + type-checker)
      3. Has clear docstring explaining what schemes exist

    This documents the NEW PATTERN for all operator kinds.
    """

    def test_wrapper_returns_collision_operator_protocol(self):
        """Wrapper returns a CollisionOperator satisfying the protocol.

        The return type is annotated as CollisionOperator, which tells:
          - Type-checker (mypy): "This function returns CollisionOperator"
          - IDE: Autocomplete knows the function signature
          - Developer: Knows it satisfies the protocol contract
        """
        from operators.collision import build_collision_fn

        collision_op = build_collision_fn("bgk")

        # Should be callable with the CollisionOperator signature
        assert callable(collision_op)

    def test_wrapper_delegates_to_generic_factory(self):
        """Wrapper doesn't reimplement logic — delegates to generic factory.

        This test documents that the wrapper is THIN:
          def build_collision_fn(scheme: str) -> CollisionOperator:
              return build_operator("collision_models", scheme)

        It adds type safety without duplicating factory logic.
        """
        from operators.collision import build_collision_fn
        from operators.collision._bgk import collide_bgk
        from operators.factory import build_operator

        # Both should return the same underlying function
        wrapper_result = build_collision_fn("bgk")
        generic_result = build_operator("collision_models", "bgk")

        assert wrapper_result is generic_result is collide_bgk

    def test_wrapper_raises_same_errors_as_generic(self):
        """Wrapper error handling matches generic factory.

        Since the wrapper delegates to the generic factory, error
        handling is consistent across all operator types.
        """
        from operators.collision import build_collision_fn

        with pytest.raises(ValueError):
            build_collision_fn("invalid_scheme")


class TestTypeSafeStreamingWrapper:
    """Type-safe wrapper for streaming operators.

    Documents the same pattern as collision:
      1. Thin wrapper in operators/streaming/__init__.py
      2. Delegates to generic factory
      3. Returns strongly-typed StreamingOperator
    """

    def test_wrapper_returns_streaming_operator(self):
        """Wrapper returns a StreamingOperator satisfying the protocol."""
        from operators.streaming import build_streaming_fn

        streaming_op = build_streaming_fn("standard")
        assert callable(streaming_op)

    def test_wrapper_delegates_to_generic_factory(self):
        """Wrapper delegates to generic factory without duplication."""
        from operators.factory import build_operator
        from operators.streaming import build_streaming_fn

        wrapper_result = build_streaming_fn("standard")
        generic_result = build_operator("stream", "standard")

        assert wrapper_result is generic_result


class TestTypeSafeEquilibriumWrapper:
    """Type-safe wrapper for equilibrium operators.

    Same pattern as collision and streaming.
    """

    def test_wrapper_returns_equilibrium_operator(self):
        """Wrapper returns an EquilibriumOperator satisfying the protocol."""
        from operators.equilibrium import build_equilibrium_fn

        eq_op = build_equilibrium_fn("wb")
        assert callable(eq_op)

    def test_wrapper_delegates_to_generic_factory(self):
        """Wrapper delegates to generic factory without duplication."""
        from operators.equilibrium import build_equilibrium_fn
        from operators.factory import build_operator

        wrapper_result = build_equilibrium_fn("wb")
        generic_result = build_operator("equilibrium", "wb")

        assert wrapper_result is generic_result


class TestTypeSafeMacroscopicWrapper:
    """Type-safe wrapper for macroscopic operators.

    Same pattern as collision, streaming, and equilibrium.
    """

    def test_wrapper_returns_macroscopic_operator(self):
        """Wrapper returns a MacroscopicOperator satisfying the protocol."""
        from operators.macroscopic import build_macroscopic_fn

        macro_op = build_macroscopic_fn("standard")
        assert callable(macro_op)

    def test_wrapper_delegates_to_generic_factory(self):
        """Wrapper delegates to generic factory without duplication."""
        from operators.factory import build_operator
        from operators.macroscopic import build_macroscopic_fn

        wrapper_result = build_macroscopic_fn("standard")
        generic_result = build_operator("macroscopic", "standard")

        assert wrapper_result is generic_result


# ── PART 3: API Compatibility Tests ──────────────────────────────────


class TestBackwardCompatibility:
    """Refactoring preserves existing API.

    The import location changed (factory.py → __init__.py), but
    the public API `build_collision_fn` is still available via both
    `from operators.collision import build_collision_fn` (recommended)
    and falls back gracefully for any old code.
    """

    def test_collision_init_import_recommended(self):
        """Recommended import: from operators.collision import build_collision_fn"""
        from operators.collision import build_collision_fn

        op = build_collision_fn("bgk")
        assert callable(op)

    def test_collision_factory_module_no_longer_exists(self):
        """Old factory.py files were deleted. Import from __init__.py instead."""
        with pytest.raises(ModuleNotFoundError):
            from operators.collision.factory import build_collision_fn  # noqa: F401

    def test_all_schemes_still_available(self):
        """All previously available schemes are still accessible.

        Refactoring doesn't change which operators are registered,
        just how they're accessed.
        """
        from operators.collision import build_collision_fn

        bgk_op = build_collision_fn("bgk")
        mrt_op = build_collision_fn("mrt")

        assert callable(bgk_op)
        assert callable(mrt_op)


# ── PART 4: Integration Tests ────────────────────────────────────────


class TestOperatorIntegration:
    """Refactored operators work together in realistic scenarios.

    These tests verify that the refactored factories integrate
    properly with existing code (step.py, simulation runners, etc).
    """

    def test_all_operator_kinds_accessible_via_factories(self):
        """Every operator kind has a factory function."""
        from operators.collision import build_collision_fn
        from operators.equilibrium import build_equilibrium_fn
        from operators.macroscopic import build_macroscopic_fn
        from operators.streaming import build_streaming_fn

        assert callable(build_collision_fn("bgk"))
        assert callable(build_streaming_fn("standard"))
        assert callable(build_equilibrium_fn("wb"))
        assert callable(build_macroscopic_fn("standard"))

    def test_step_module_can_use_factories(self):
        """step.py can construct operators using factory pattern.

        This verifies the primary use case: step.py builds operators
        at the start of each timestep using factories.
        """
        from operators.collision import build_collision_fn
        from operators.equilibrium import build_equilibrium_fn
        from operators.macroscopic import build_macroscopic_fn
        from operators.streaming import build_streaming_fn

        # Simulate what step.py does
        collision_fn = build_collision_fn("bgk")
        streaming_fn = build_streaming_fn("standard")
        equilibrium_fn = build_equilibrium_fn("wb")
        macroscopic_fn = build_macroscopic_fn("standard")

        # All operators should be available for use
        assert callable(collision_fn)
        assert callable(streaming_fn)
        assert callable(equilibrium_fn)
        assert callable(macroscopic_fn)


# ── PART 5: Documentation via Tests ─────────────────────────────────


class TestFactoryPattern:
    """Tests that explain the new factory pattern.

    These tests serve as living documentation of:
      1. How to add a new operator kind
      2. How the registry and factory interact
      3. Why this pattern is better than separate factories
    """

    def test_adding_new_operator_kind_requires_only_wrapper(self):
        """Extending the system: Adding a new operator kind.

        To add a new operator kind (e.g., "stress_tensor"):

        1. Register it like any other operator:
           @register_operator("stress_tensor", name="simple")
           def compute_stress(...): ...

        2. Create a thin wrapper in src/operators/stress/__init__.py:
           def build_stress_fn(scheme: str) -> StressOperator:
               return build_operator("stress_tensor", scheme)

        3. Done! No factory.py file needed.

        This test documents that the pattern is minimal and repeatable.
        """

        # Register a test operator
        @register_operator("_test_operator_kind", name="_test_scheme")
        def test_op():
            return "test"

        # Generic factory finds it
        from operators.factory import build_operator

        result = build_operator("_test_operator_kind", "_test_scheme")

        assert result() == "test"

        # Cleanup
        unregister_operator("_test_operator_kind", "_test_scheme")

    def test_registry_and_factory_are_properly_decoupled(self):
        """Registry and factory have clean separation of concerns.

        REGISTRY: Stores operator name → implementation mapping
        FACTORY: Reads registry, returns typed result
        This separation enables:
          - Multiple implementations of the same factory pattern
          - Registry as the single source of truth
          - Easy testing (can register test operators)
          - Plugin system (register operators dynamically)
        """
        from operators.factory import build_operator

        # Registry has all the data
        collision_ops = get_operators("collision_models")
        assert "bgk" in collision_ops
        assert "mrt" in collision_ops

        # Factory just reads from registry and returns
        bgk_fn = build_operator("collision_models", "bgk")
        assert bgk_fn is collision_ops["bgk"].target


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
