"""Physics operators (collision, streaming, equilibrium, etc.).

Operators are organized by category:
- collision/     : Collision operators (BGK, MRT, source)
- streaming/     : Streaming operators
- equilibrium/   : Equilibrium distribution
- macroscopic/   : Macroscopic moment computation
- boundary/      : Boundary conditions
- differential/  : Differential operators (gradient, laplacian)
- force/         : Force models
- wetting/       : Wetting and contact angle
- initialise/    : Population initialization

Operators are auto-discovered and registered via registry.py.
Use registry.get_operators(category) to retrieve implementations.
"""

from tud_lbm.operators.protocols import (
    BoundaryOperator,
    CollisionOperator,
    DifferentialOperator,
    EquilibriumOperator,
    ForceOperator,
    InitialiserOperator,
    MacroscopicOperator,
    StreamingOperator,
)

__all__ = [
    "CollisionOperator",
    "StreamingOperator",
    "EquilibriumOperator",
    "MacroscopicOperator",
    "BoundaryOperator",
    "DifferentialOperator",
    "ForceOperator",
    "InitialiserOperator",
]
