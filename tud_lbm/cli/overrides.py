"""Parsing and application of ``--override KEY=VALUE`` arguments."""

from __future__ import annotations
import tomllib
from typing import Any
from tud_lbm.cli._console import console

_SECTION_ALIAS_MAP = {
    "simulation_type": "",
    "multiphase": "",
    "output": "",
    "boundary_conditions": "bc_config",
    "wetting": "wetting_config",
    "hysteresis": "hysteresis_config",
    "chemical_step": "chemical_step_config",
}


def _parse_override_argument(raw_override: str) -> tuple[str, object]:
    """Parse one --override expression and return (path, typed_value).

    Supports two formats:
    1. Direct: path=value (e.g., tau=0.7, simulation_name="test")
    2. Legacy: (path, value) or override(path, value)

    Values are parsed as TOML literals, supporting:
    - Numbers: 0.7, 123, 1e-5
    - Strings: "text" (with quotes)
    - Booleans: true, false
    - Arrays: [1, 2, 3], [0.6, 0.7, 0.8]

    Args:
        raw_override: Override expression string.

    Returns:
        Tuple of (dotted_path, typed_value).

    Raises:
        ValueError: If format is invalid or value cannot be parsed as TOML.

    Examples:
        _parse_override_argument('tau=0.7')
        # ('tau', 0.7)

        _parse_override_argument('simulation_type.simulation_name="test"')
        # ('simulation_type.simulation_name', 'test')

        _parse_override_argument('tau=[0.6, 0.7, 0.8]')
        # ('tau', [0.6, 0.7, 0.8])
    """
    value_expr: str

    text = raw_override.strip()
    if not text:
        msg = "override expression cannot be empty"
        raise ValueError(msg)

    if "=" in text:
        path, value_expr = text.split("=", 1)
        path = path.strip()
        value_expr = value_expr.strip()
    else:
        # Also support forms like: (path, value) or override(path, value)
        if text.startswith("override(") and text.endswith(")"):
            text = text[len("override(") : -1].strip()
        elif text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        if "," not in text:
            msg = "invalid override format. Use 'path=value' (e.g. simulation_type.tau=0.7)."
            raise ValueError(
                msg,
            )
        path, value_expr = text.split(",", 1)
        path = path.strip()
        value_expr = value_expr.strip()

    if not path:
        msg = "override path cannot be empty"
        raise ValueError(msg)
    if not value_expr:
        msg = "override value cannot be empty"
        raise ValueError(msg)

    try:
        value = tomllib.loads(f"value = {value_expr}")["value"]
    except tomllib.TOMLDecodeError as exc:
        msg = f"invalid override value '{value_expr}'. Use a TOML literal (quoted strings, numbers, booleans, arrays)."
        raise ValueError(
            msg,
        ) from exc

    return path, value


def _normalise_override_path(path: str) -> list[str]:
    """Map TOML table paths to raw-config keys and split into segments.

    Normalizes TOML section aliases to their field names:
    - simulation_type.* → * (direct field)
    - boundary_conditions.* → bc_config.*
    - wetting.* → wetting_config.*
    - hysteresis.* → hysteresis_config.*
    - electric_force.* → electric_force.*
    - gravity_force.* → gravity_force.*

    Args:
        path: Dotted-path string (e.g., "simulation_type.tau" or "gravity_force.force_g").

    Returns:
        List of path segments (e.g., ["tau"] or ["gravity_force", "force_g"]).

    Raises:
        ValueError: If path is empty or becomes empty after normalisation.

    Examples:
        _normalise_override_path('simulation_type.tau')
        # ['tau']

        _normalise_override_path('gravity_force.force_g')
        # ['gravity_force', 'force_g']

        _normalise_override_path('boundary_conditions.top')
        # ['bc_config', 'top']
    """
    parts = [segment.strip() for segment in path.split(".") if segment.strip()]
    if not parts:
        msg = "override path cannot be empty"
        raise ValueError(msg)

    head = parts[0]
    if head in _SECTION_ALIAS_MAP:
        mapped = _SECTION_ALIAS_MAP[head]
        # SIM108: use ternary instead of if-else block
        parts = [mapped, *parts[1:]] if mapped else parts[1:]

    if not parts:
        msg = f"override path '{path}' does not reference a field"
        raise ValueError(msg)
    return parts


def _set_nested_override(raw_config: dict[str, Any], path: str, value: object) -> None:
    """Apply a typed override value to raw config using dotted-path syntax.

    Automatically creates nested dicts as needed. For example, to set
    gravity_force.force_g=5e-7, this will create raw_config['gravity_force']
    if it doesn't exist, then set its 'force_g' sub-key.

    Args:
        raw_config: The raw configuration dict to mutate.
        path: Dotted-path string (normalised or already valid).
        value: The typed value to assign.

    Raises:
        TypeError: If an intermediate key exists but is not a dict.

    Examples:
        raw = {}
        _set_nested_override(raw, 'tau', 0.7)
        # raw == {'tau': 0.7}

        raw = {}
        _set_nested_override(raw, 'gravity_force.force_g', 5e-7)
        # raw == {'gravity_force': {'force_g': 5e-7}}
    """
    parts = _normalise_override_path(path)

    if len(parts) == 1:
        raw_config[parts[0]] = value
        return

    cursor: dict[str, Any] = raw_config
    for key in parts[:-1]:
        existing = cursor.get(key)
        if existing is None:
            existing = {}
            cursor[key] = existing
        if not isinstance(existing, dict):
            dotted_prefix = ".".join(parts[:-1])
            # TRY004: use TypeError for invalid type
            msg = f"override path '{path}' is invalid: '{dotted_prefix}' is not a table"
            raise TypeError(msg)
        cursor = existing
    cursor[parts[-1]] = value


def _apply_overrides(raw_config: dict[str, Any], overrides: tuple[str, ...]) -> None:
    """Parse and apply all --override expressions in order.

    Each override is parsed, type-checked, and applied to raw_config before
    config expansion. This allows CLI users to override or create config
    fields without editing the file.

    Overrides are applied in the order provided, so later values override
    earlier ones for the same path.

    Args:
        raw_config: The configuration dict to mutate (in-place).
        overrides: Tuple of override expressions (e.g., ("tau=0.7", "nt=500")).

    Prints:
        Console output listing each override applied.

    Raises:
        ValueError: If any override has invalid format or TOML syntax.
        TypeError: If any override path conflicts with existing non-dict values.
    """
    if not overrides:
        return

    console.print("[cyan]Applying CLI overrides:[/cyan]")
    for raw_override in overrides:
        path, value = _parse_override_argument(raw_override)
        _set_nested_override(raw_config, path, value)
        console.print(f"  - {path} = {value!r}")
    console.print()
