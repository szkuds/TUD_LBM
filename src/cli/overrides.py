"""Parsing and application of ``--override KEY=VALUE`` arguments."""

from __future__ import annotations
import tomllib
from copy import deepcopy
from difflib import get_close_matches
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
import click
from rich.prompt import Prompt
from src.cli._console import console

if TYPE_CHECKING:
    from collections.abc import Callable

#: The value a confirm/override loop is deciding about — the expanded config (or
#: tuple of configs) built from the raw dict the override is applied to.
_T = TypeVar("_T")

#: TOML section name -> the ``SimulationConfig`` field it flattens to (``""`` for
#: sections merged straight into the top level). Mirrors the section handling in
#: :meth:`src.config.adapter_base.ConfigAdapter._merge_sections`, so an override
#: path can be written with the same section names the config file uses.
#: ``tests/cli/test_cli.py::TestSectionAliasMap`` pins the two together — the
#: ``obstacle`` entry was missing here for exactly that reason.
_SECTION_ALIAS_MAP = {
    "simulation_type": "",
    "multiphase": "",
    "output": "",
    "boundary_conditions": "bc_config",
    "wetting": "wetting_config",
    "hysteresis": "hysteresis_config",
    "chemical_step": "chemical_step_config",
    "obstacle": "obstacle_config",
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

    Normalises TOML section aliases to their field names:
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


def _known_override_heads() -> set[str]:
    """Every first segment an override path may start with.

    The ``SimulationConfig`` field names plus the section aliases above, so
    ``bc_config.left`` and ``boundary_conditions.left`` are both accepted.
    Imported lazily to keep this module free of a config dependency.
    """
    from src.config import SimulationConfig

    return set(SimulationConfig.__dataclass_fields__) | set(_SECTION_ALIAS_MAP)


def _reject_unknown_field(path: str, head: str) -> None:
    """Raise on an override path whose first segment names nothing.

    Caught here because neither layer below reports it usefully: the adapter
    sweeps an unknown key into ``config.extra`` and says nothing, while
    ``expand_config`` calls ``SimulationConfig(**raw)`` and surfaces a bare
    ``unexpected keyword argument``. Both leave the operator guessing at a
    typo, so the suggestion is made where the typo was actually typed.
    """
    known = _known_override_heads()
    if head in known:
        return
    suggestions = get_close_matches(head, sorted(known), n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    msg = f"unknown override field '{head}' in '{path}'.{hint}"
    raise ValueError(msg)


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
        # Validated here, at the operator-input boundary, rather than inside
        # _set_nested_override, which stays a plain dotted-path dict setter.
        _reject_unknown_field(path, _normalise_override_path(path)[0])
        _set_nested_override(raw_config, path, value)
        console.print(f"  - {path} = {value!r}")
    console.print()


def _ask_confirm(prompt_text: str) -> str:
    """Ask the y/n/o question and return 'yes', 'no' or 'override'."""
    choice = Prompt.ask(
        f"{prompt_text} [[green]y[/green]/[red]n[/red]/[cyan]o[/cyan]=override]",
        choices=["y", "n", "o"],
        default="y",
        show_choices=False,
    )
    return {"y": "yes", "n": "no", "o": "override"}[choice]


def confirm_or_override(
    raw_config: dict[str, Any] | None,
    value: _T,
    *,
    prompt: Callable[[_T], str],
    rebuild: Callable[[dict[str, Any]], _T],
) -> _T | None:
    """Confirm *value*, or apply an inline override and rebuild it.

    Loops until the operator answers yes or no, so a mistyped parameter is
    corrected in place rather than by cancelling and re-invoking the command.
    *rebuild* re-expands the mutated raw dict **and displays the new summary** —
    each caller shows its own — and *prompt* is re-evaluated each round because
    an override can change what is being confirmed (a scalar becoming a sweep).

    The override is applied to a copy and only committed into *raw_config* once
    *rebuild* succeeds: a rejected expression must not leave the raw dict in a
    state the next round builds on. The commit is in-place so callers holding a
    reference to *raw_config* see the accepted overrides.

    Returns the confirmed value, or ``None`` when the operator answers no.
    """
    while True:
        decision = _ask_confirm(prompt(value))
        if decision == "no":
            return None
        if decision == "yes":
            return value
        if raw_config is None:
            console.print("[yellow]Inline overrides require a config file.[/yellow]")
            continue

        # There is no shell at this prompt, so the value is read as a bare TOML
        # literal: a string needs its quotes. The one numeric example this used
        # to show gave no hint of that, and a section name is easier to recall
        # from a full path than from the field list.
        raw_expr = Prompt.ask(
            "[cyan]Enter override[/cyan] "
            '[dim](one key, TOML value: tau=0.7, boundary_conditions.left="periodic")[/dim]',
        )
        candidate = deepcopy(raw_config)
        try:
            _apply_overrides(candidate, (raw_expr,))
            value = rebuild(candidate)
        except (ValueError, TypeError, click.UsageError) as exc:
            console.print(f"[red]Invalid override: {exc}[/red]")
            continue
        raw_config.clear()
        raw_config.update(candidate)
