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
from src.operators._loader import auto_load_operators
from src.operators.protocols import BoundaryOperator
from src.operators.protocols import CollisionOperator
from src.operators.protocols import DifferentialOperator
from src.operators.protocols import EOSFunction
from src.operators.protocols import EquilibriumOperator
from src.operators.protocols import ForceOperator
from src.operators.protocols import InitialiserOperator
from src.operators.protocols import MacroscopicOperator
from src.operators.protocols import StreamingOperator


def load_all() -> None:
    """Import every operator subpackage to trigger registry registration."""
    # Derived from __name__ rather than hardcoded: a hardcoded package prefix
    # is invisible to import rewriting and silently breaks discovery.
    for _, subpkg_name, is_pkg in pkgutil.iter_modules(__path__):
        if is_pkg:
            auto_load_operators(f"{__name__}.{subpkg_name}")


__all__ = [
    "BoundaryOperator",
    "CollisionOperator",
    "DifferentialOperator",
    "EOSFunction",
    "EquilibriumOperator",
    "ForceOperator",
    "InitialiserOperator",
    "MacroscopicOperator",
    "StreamingOperator",
]
