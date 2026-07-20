"""Interactive selection of plotting/analysis operators."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from rich.prompt import Prompt
from src.cli._console import console
from src.cli.display import _build_fields_table

if TYPE_CHECKING:
    from collections.abc import Iterable
    from src.simulation_io.plotting import FigureBuilder


@dataclass(frozen=True)
class OperatorChoice:
    """One selectable operator, and whether the run's config already lists it."""

    name: str
    description: str
    kind: str
    in_config: bool


def build_choices(available: dict, configured: Iterable[str]) -> list[OperatorChoice]:
    """Build the prompt's choices, listing configured operators first.

    Ordering is in-config first, then the rest, each alphabetical — so the
    relevant entries sit at the top and the typed numbering stays stable for a
    given config.
    """
    from src.cli.display import _operator_description

    in_config = set(configured)
    choices = [
        OperatorChoice(
            name=name,
            description=_operator_description(entry.target),
            kind=getattr(entry, "kind", ""),
            in_config=name in in_config,
        )
        for name, entry in available.items()
    ]
    return sorted(choices, key=lambda c: (not c.in_config, c.name))


def _resolve_token(token: str, names: list[str], available: dict) -> str | None:
    """Resolve a single user token (number or name) to an operator name.

    Returns the resolved name, or ``None`` when the token is invalid.
    """
    try:
        idx = int(token) - 1
    except ValueError:
        if token in available:
            return token
        console.print(f"[yellow]Unknown field '{token}' — skipped[/yellow]")
        return None
    else:
        if 0 <= idx < len(names):
            return names[idx]
        console.print(f"[yellow]Number {token} out of range — skipped[/yellow]")
        return None


def _parse_field_tokens(raw: str, names: list[str], available: dict) -> list[str]:
    """Parse a comma-separated user input string into a list of valid operator names."""
    selected: list[str] = []
    for raw_token in raw.split(","):
        token = raw_token.strip()
        if not token:
            continue
        resolved = _resolve_token(token, names, available)
        if resolved is not None:
            selected.append(resolved)
    return selected


def _prompt_fields(
    available: dict,
    current: list[str] | None,
    label: str,
) -> list[str] | None:
    """Interactively select plot operators from *available*.

    Args:
        available: ``{name: OperatorEntry}`` dict of all selectable operators.
        current: Pre-selected names (from config), or ``None`` for all.
        label: Human-readable context shown in the prompt header.

    Returns:
        Validated list of operator names, or ``None`` when the user accepts the
        default (meaning "use all available / whatever the builder defaults to").
    """
    names = sorted(available.keys())

    console.print()
    console.print(f"[bold cyan]Available {label}:[/bold cyan]")
    console.print(_build_fields_table(names, available))

    default_str = ", ".join(current) if current else "(all)"
    console.print(f"\n[dim]Current selection:[/dim] {default_str}")
    console.print("[dim]Enter comma-separated numbers (e.g. 1,3) or names (e.g. density,force).[/dim]")
    console.print("[dim]Press Enter to keep current selection.[/dim]")

    try:
        raw = Prompt.ask("Select fields", default="")
    except EOFError:
        return current

    if not raw.strip():
        return current  # None → builder default; list → exact config selection

    selected = _parse_field_tokens(raw, names, available)

    if not selected:
        console.print("[dim]No valid selection — keeping current.[/dim]")
        return current

    return selected


def prompt_fields_marked(
    available: dict,
    current: list[str] | None,
    *,
    configured: Iterable[str],
    label: str,
    config_label: str,
) -> list[str] | None:
    """Select operators, marking which ones the stored config already lists.

    Args:
        available: ``{name: OperatorEntry}`` of every selectable operator.
        current: Pre-selected names, or ``None`` to accept the builder default.
        configured: Names the run's config lists — used only for the marking.
        label: Human-readable context shown in the prompt header.
        config_label: What ``configured`` came from, named in the footer.

    Returns:
        The chosen operator names, or ``None`` when the user takes the default.
    """
    from src.cli.display import build_choices_table
    from src.cli.display import choices_footer

    choices = build_choices(available, configured)
    names = [choice.name for choice in choices]

    console.print()
    console.print(f"[bold cyan]Available {label}:[/bold cyan]")
    console.print(build_choices_table(choices))
    console.print(choices_footer(choices, config_label=config_label))

    default_str = ", ".join(current) if current else "(all)"
    console.print(f"\n[dim]Current selection:[/dim] {default_str}")
    console.print("[dim]Enter comma-separated numbers (e.g. 1,3) or names (e.g. density,force).[/dim]")
    console.print("[dim]Press Enter to keep current selection.[/dim]")

    try:
        raw = Prompt.ask("Select fields", default="")
    except EOFError:
        return current

    if not raw.strip():
        return current

    selected = _parse_field_tokens(raw, names, available)
    if not selected:
        console.print("[dim]No valid selection — keeping current.[/dim]")
        return current
    return selected


def _prompt_snapshot_timesteps(available: list[int]) -> list[int]:
    """Interactively collect the timesteps to snapshot for the ``snapshot_fig`` plot."""
    console.print(f"[dim]Available timesteps: {', '.join(str(t) for t in available)}[/dim]")
    try:
        raw_ts = Prompt.ask("Enter snapshot timesteps (comma-separated)")
    except EOFError:
        return []
    return [int(tok.strip()) for tok in raw_ts.split(",") if tok.strip()]


def _configure_snapshot_fig(builder: FigureBuilder, field_list: list[str] | None) -> None:
    """Prompt for and wire up snapshot timesteps when ``snapshot_fig`` is requested."""
    if not field_list or "snapshot_fig" not in field_list:
        return
    available = [t for t, _ in builder.sorted_timed_files()]
    requested_ts = _prompt_snapshot_timesteps(available)
    for op in builder.analysis_operators:
        if op.name == "snapshot_fig":
            op.timesteps = requested_ts
