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

from __future__ import annotations
import pkgutil
from operators._loader import auto_load_operators
from tud_lbm.operators.protocols import BoundaryOperator
from tud_lbm.operators.protocols import CollisionOperator
from tud_lbm.operators.protocols import DifferentialOperator
from tud_lbm.operators.protocols import EquilibriumOperator
from tud_lbm.operators.protocols import ForceOperator
from tud_lbm.operators.protocols import InitialiserOperator
from tud_lbm.operators.protocols import MacroscopicOperator
from tud_lbm.operators.protocols import StreamingOperator


def load_all() -> None:
    """Import every operator subpackage to trigger registry registration."""
    for _, subpkg_name, is_pkg in pkgutil.iter_modules(__path__):
        if is_pkg:
            auto_load_operators(f"operators.{subpkg_name}")


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
