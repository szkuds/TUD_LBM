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

from tud_lbm.operators.protocols import BoundaryOperator
from tud_lbm.operators.protocols import CollisionOperator
from tud_lbm.operators.protocols import DifferentialOperator
from tud_lbm.operators.protocols import EquilibriumOperator
from tud_lbm.operators.protocols import ForceOperator
from tud_lbm.operators.protocols import InitialiserOperator
from tud_lbm.operators.protocols import MacroscopicOperator
from tud_lbm.operators.protocols import StreamingOperator

__all__ = [
    "BoundaryOperator",
    "CollisionOperator",
    "DifferentialOperator",
    "EquilibriumOperator",
    "ForceOperator",
    "InitialiserOperator",
    "MacroscopicOperator",
    "StreamingOperator",
]
