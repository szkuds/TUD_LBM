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


#: Scalar fields that once took a list as a sweep axis.
_SCALAR_FIELDS = ("tau", "nt", "collision_scheme")
#: Sections whose keys once took a list as a sweep axis. No key there takes a list as a value.
_SCALAR_SECTIONS = ("multiphase", "wetting", "hysteresis", "chemical_step")
_SWEEP_HINT = "a list is not a value. Stage a parameter sweep with 'tud-lbm sweep stage'."


def _list_paths(prefix: str, table: dict[str, Any]) -> list[str]:
    """The dotted path of every list or tuple value in *table*, at any depth."""
    paths: list[str] = []
    for key, value in table.items():
        if isinstance(value, dict):
            paths.extend(_list_paths(f"{prefix}.{key}", value))
        elif isinstance(value, (list, tuple)):
            paths.append(f"{prefix}.{key}")
    return paths


def reject_sweep_lists(fields: dict[str, Any]) -> None:
    """Reject a list where a parameter sweep used to read one as its axis.

    *fields* maps a config field name to its value as given, before any default
    is filled in. A ``grid_shape`` is a list itself, so only a list of lists is
    rejected there.
    """
    paths = [name for name in _SCALAR_FIELDS if isinstance(fields.get(name), (list, tuple))]
    grid_shape = fields.get("grid_shape")
    if isinstance(grid_shape, (list, tuple)) and any(isinstance(n, (list, tuple)) for n in grid_shape):
        paths.append("grid_shape")
    for section in _SCALAR_SECTIONS:
        if isinstance(table := fields.get(section), dict):
            paths.extend(_list_paths(section, table))
    for name, params in (fields.get("forces") or {}).items():
        if isinstance(params, dict):
            paths.extend(_list_paths(name, params))
    if paths:
        msg = f"{', '.join(paths)}: {_SWEEP_HINT}"
        raise ValueError(msg)
