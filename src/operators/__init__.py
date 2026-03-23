"""Pure-function operators for JAX-compliant LBM.

This package provides **module-level pure functions** that replace the
legacy mutable operator classes.  No class instances, no ``self``, no
``static_argnums`` — just arrays in, arrays out.

Sub-packages:
    collision: BGK and MRT collision operators + factory.
    streaming: Lattice streaming (propagation).
    equilibrium: Equilibrium distribution computation.
    macroscopic: Density/velocity extraction (single-phase & multiphase).
    boundary: Boundary condition functions + composite builder.
"""
