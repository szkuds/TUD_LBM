"""Automatic module loader for operator implementations.

Discovers and imports all private operator modules (_*.py) in a package
without hardcoding their names. This ensures registry registration occurs
without maintaining brittle import lists.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


def auto_load_operators(package_name: str) -> None:
    """Auto-discover and import all private operator modules in a package.

    Scans the package directory for modules matching '_*.py' and imports them
    to trigger their @register_operator decorators without hardcoding the list.

    Args:
        package_name: Full package name, e.g. 'operators.collision'

    Example:
        # In src/operators/collision/__init__.py:
        from operators._loader import auto_load_operators
        auto_load_operators('operators.collision')
    """
    try:
        module = importlib.import_module(package_name)
    except ImportError:
        return

    # Get the package path
    if not hasattr(module, '__path__'):
        return

    package_path = module.__path__[0]

    # Scan for private modules (_*.py)
    for _, module_name, is_pkg in pkgutil.iter_modules([package_path]):
        if not module_name.startswith('_'):
            continue
        if is_pkg:
            continue

        try:
            importlib.import_module(f"{package_name}.{module_name}")
        except ImportError:
            # Skip modules that fail to import
            pass
