"""Unified operator factory — single source of truth for operator lookup.

This module provides a generic factory function that resolves operator names
to their implementations using the central registry. All operator-specific
factories (in collision/, streaming/, etc) delegate to this generic factory,
which eliminates boilerplate duplication.

## Architecture

The unified factory separates concerns:

1. REGISTRY (src/registry.py)
   - Stores: name → implementation mapping
   - Updated by: @register_operator decorators
   - Query: get_operators(kind) → {name: OperatorEntry}

2. GENERIC FACTORY (this module)
   - Resolves: (kind, scheme) → registered operator
   - Returns: OperatorTarget (the callable function/class)
   - Used by: All operator-specific factories

3. TYPE-SAFE WRAPPERS (src/operators/{kind}/__init__.py)
   - Delegate to: build_operator()
   - Return type: Annotated as specific Protocol (CollisionOperator, etc)
   - Purpose: IDE autocomplete, type-checker support, clear intent

## Example

```python
# Generic factory (this module)
def build_operator(kind: str, scheme: str) -> OperatorTarget:
    ops = get_operators(kind)
    return ops[scheme].target

# Type-safe wrapper (in operators/collision/__init__.py)
def build_collision_fn(scheme: str) -> CollisionOperator:
    return build_operator("collision_models", scheme)

# Usage
collision_fn = build_collision_fn("bgk")  # IDE knows it's CollisionOperator
f_col = collision_fn(f, feq, tau)
```

## Why Not Multiple Factories?

Before: 6-8 separate factory.py files (~40 lines each) = 240-320 lines boilerplate
After: 1 generic factory (~20 lines) + thin wrappers (~5 lines each) = 40-50 lines total

This follows: "Boilerplate in one place, business logic everywhere else"
"""

from __future__ import annotations

from registry import get_operators, OperatorTarget


def build_operator(kind: str, scheme: str) -> OperatorTarget:
    """Resolve operator name to implementation.
    
    This is the single source of truth for operator factory logic.
    All operator-specific factories delegate to this function.
    
    Args:
        kind: Operator kind ("collision_models", "stream", "equilibrium", etc)
        scheme: Operator name within that kind ("bgk", "standard", "wb", etc)
        
    Returns:
        OperatorTarget: The operator function/class satisfying the protocol
        
    Raises:
        ValueError: If kind is not registered or scheme is unknown
        
    Example:
        >>> collision_op = build_operator("collision_models", "bgk")
        >>> collision_op
        <function collide_bgk at 0x...>
        
    Example (for internal use in __init__.py):
        >>> from operators.factory import build_operator
        >>> def build_collision_fn(scheme: str) -> CollisionOperator:
        ...     return build_operator("collision_models", scheme)
    """
    ops = get_operators(kind)
    
    if not ops:
        raise ValueError(f"No operators registered for kind '{kind}'")
    
    try:
        return ops[scheme].target
    except KeyError as exc:
        # Include available schemes in error message for better UX
        valid_schemes = ", ".join(sorted(ops.keys()))
        raise ValueError(
            f"Unknown {kind} scheme '{scheme}'. Valid schemes: {valid_schemes}"
        ) from exc
