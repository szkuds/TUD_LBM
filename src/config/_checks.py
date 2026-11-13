"""Checks shared by the section modules."""

from __future__ import annotations
import importlib
from typing import Any
from src.registry import get_operator_names


def validate_positive(value: Any, name: str) -> None:  # noqa: ANN401
    """Reject a set *value* that is not positive."""
    if value is not None and value <= 0:
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


def validate_nonnegative(value: Any, name: str) -> None:  # noqa: ANN401
    """Reject a set *value* that is negative."""
    if value is not None and value < 0:
        msg = f"{name} must be non-negative, got {value}"
        raise ValueError(msg)


def registered_names(package: str, kind: str) -> set[str]:
    """Names registered under *kind*, after importing *package* for its registration side effect."""
    importlib.import_module(package)
    return get_operator_names(kind)
