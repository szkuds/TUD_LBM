"""Pure-function operators for JAX-compliant LBM.

This package provides **module-level pure functions** that implement
the protocols defined in operators.protocols. All functions replace
the legacy mutable operator classes: no class instances, no ``self``,
no ``static_argnums`` — just arrays in, arrays out.

RECOMMENDED USAGE PATTERN:
    1. Import protocol interface:
        from operators.protocols import CollisionOperator
    
    2. Get operator via factory (primary API):
        from operators.collision import build_collision_fn
        op: CollisionOperator = build_collision_fn("bgk")
    
    3. Use operator via protocol interface:
        f_col = op(f, feq, tau)
    
    This keeps your code independent of specific implementations.

SUBPACKAGES:
    collision: CollisionOperator implementations + build_collision_fn factory
    streaming: StreamingOperator implementations + build_streaming_fn factory
    equilibrium: EquilibriumOperator implementations + factory
    macroscopic: MacroscopicOperator implementations + factory
    boundary: BoundaryConditionOperator implementations + factory
    initialise: InitialiserOperator implementations + factory
    
    (Additional: force, differential, wetting for advanced use)

DESIGN PRINCIPLE:
    - Protocols (operators.protocols) = stable interfaces
    - Implementations (bgk.py, mrt.py, etc) = internal details
    - Factories (build_collision_fn, etc) = public API
    - Users type-hint against protocols, not implementations
"""
