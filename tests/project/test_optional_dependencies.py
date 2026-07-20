"""Behavioral tests for optional dependency handling.

Validates that the hysteresis path raises a clear, actionable ImportError
when optax is absent, rather than an obscure AttributeError or crash.
The ``mock_optax_missing`` fixture (defined in ``tests/conftest.py``)
simulates a missing optax installation via monkeypatching.
"""

from __future__ import annotations
import pytest


class TestOptaxOptionalDependency:
    """Hysteresis path raises ImportError with install hint when optax is absent."""

    def test_import_optax_raises_helpful_message_when_missing(self, mock_optax_missing) -> None:
        """_import_optax() raises ImportError with pip-install hint when optax is missing.

        Given: optax is not installed (simulated by mock_optax_missing)
        When: the lazy optax importer is called
        Then: ImportError is raised with an actionable message
        """
        from src.operators.wetting.hysteresis.hysteresis import _import_optax

        with pytest.raises(ImportError, match="pip install optax"):
            _import_optax()

    def test_optax_available_in_normal_operation(self, mock_optax_present) -> None:
        """_import_optax() returns the optax module when optax is installed.

        Given: optax is installed
        When: the lazy optax importer is called
        Then: the optax module is returned without error
        """
        from src.operators.wetting.hysteresis.hysteresis import _import_optax

        optax = _import_optax()
        assert optax is not None
