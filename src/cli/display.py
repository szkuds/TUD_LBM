"""Rich tables and summary panels shown by the CLI."""

from __future__ import annotations
from typing import TYPE_CHECKING
from rich.panel import Panel
from rich.table import Table
from src.cli._console import console

if TYPE_CHECKING:
    from src.cli.field_select import OperatorChoice
    from src.config import SimulationConfig
    from src.config.array_expansion import ArrayParameterSet

_BOLD_CYAN = "bold cyan"

_VISUAL_KINDS = {
    "plotting": "Field plots - rendered per timestep snapshot",
    "analysis": "Analysis plots - computed over a run's snapshot history",
}

_ANALYSIS_USAGE = {
    "plotting": "tud-lbm animate, tud-lbm visualise",
    "analysis": "tud-lbm animate, tud-lbm visualise, tud-lbm compare",
}

_ANALYSIS_KINDS = frozenset({"plotting", "analysis"})


def _operator_description(target: object) -> str:
    """Return the first line of *target*'s docstring, or '—' when absent."""
    doc = (getattr(target, "__doc__", None) or "").strip().splitlines()
    return doc[0] if doc else "—"


def _new_operator_table(title: str, columns: tuple[tuple[str, dict], ...]) -> Table:
    """An operator listing table with a left-justified magenta *title*."""
    table = Table(
        title=title,
        show_header=True,
        header_style=_BOLD_CYAN,
        title_justify="left",
    )
    for header, style in columns:
        table.add_column(header, **style)
    return table


def _build_visual_table(kind: str, ops: dict, subtitle: str | None = None) -> Table:
    """Describe plotting/analysis operators by docstring and required data keys."""
    sub = subtitle if subtitle is not None else _VISUAL_KINDS.get(kind, "")
    table = _new_operator_table(
        f"[bold magenta]{kind}[/bold magenta]  [dim]{sub}[/dim]",
        (
            ("Name", {"style": "green", "no_wrap": True}),
            ("Description", {"style": "white"}),
            ("Required keys", {"style": "dim"}),
        ),
    )
    for name in sorted(ops):
        target = ops[name].target
        required = getattr(target, "required_keys", None)
        table.add_row(name, _operator_description(target), ", ".join(required) if required else "—")
    return table


def _build_standard_table(kind: str, ops: dict) -> Table:
    """Describe physics operators by import path and registration metadata."""
    table = _new_operator_table(
        f"[bold magenta]{kind}[/bold magenta]",
        (
            ("Name", {"style": "green", "no_wrap": True}),
            ("Target", {"style": "white"}),
            ("Metadata", {"style": "dim"}),
        ),
    )
    for name in sorted(ops):
        entry = ops[name]
        target = entry.target
        target_mod = getattr(target, "__module__", type(target).__module__)
        target_name = getattr(target, "__qualname__", getattr(target, "__name__", type(target).__name__))
        meta_str = ", ".join(f"{k}={v!r}" for k, v in entry.metadata.items()) if entry.metadata else "—"
        table.add_row(name, f"{target_mod}.{target_name}", meta_str)
    return table


def _display_operators(*, analysis: bool) -> None:
    """List every registered operator, split into the two CLI-facing groups.

    Analysis kinds (plotting/analysis) get docstring + required-key columns and
    a "used by" note; every other kind gets the import-path listing.
    """
    import src.simulation_io.plotting as _plotting_mod  # noqa: F401  registers plotting operators
    from src.operators import load_all
    from src.registry import get_operator_category
    from src.registry import get_operators

    load_all()

    label = "Analysis" if analysis else "Simulation"
    categories = sorted(c for c in get_operator_category() if (c in _ANALYSIS_KINDS) is analysis)
    if not categories:
        console.print(f"[yellow]No {label.lower()} operators registered.[/yellow]")
        return

    n_total = sum(len(get_operators(c)) for c in categories)
    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]{label} Operators[/bold blue]  ({n_total} total across {len(categories)} categories)",
        ),
    )
    console.print()

    for kind in categories:
        ops = get_operators(kind)
        if analysis:
            usage = _ANALYSIS_USAGE.get(kind, "")
            base = _VISUAL_KINDS.get(kind, "")
            subtitle = f"{base}  │  used by: {usage}" if usage else base
            console.print(_build_visual_table(kind, ops, subtitle=subtitle))
        else:
            console.print(_build_standard_table(kind, ops))
        console.print()


def _display_simulation_operators() -> None:
    """Display physics/model operators (excludes the plotting and analysis kinds)."""
    _display_operators(analysis=False)


def _display_analysis_operators() -> None:
    """Display plotting and analysis operators with their CLI usage context."""
    _display_operators(analysis=True)


def _build_fields_table(names: list[str], available: dict) -> Table:
    """Build a Rich table listing selectable operator fields."""
    table = Table(show_header=True, header_style=_BOLD_CYAN, box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="green", no_wrap=True)
    table.add_column("Description", style="white")
    for i, name in enumerate(names, 1):
        description = _operator_description(available[name].target)
        table.add_row(str(i), name, description)
    return table


def build_choices_table(choices: list[OperatorChoice]) -> Table:
    """List selectable operators, marking which the stored config already names."""
    table = Table(show_header=True, header_style=_BOLD_CYAN, box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="green", no_wrap=True)
    table.add_column("Kind", style="magenta", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Status", no_wrap=True)
    for i, choice in enumerate(choices, 1):
        status = "[green]in config[/green]" if choice.in_config else "[yellow]* not in config[/yellow]"
        table.add_row(str(i), choice.name, choice.kind, choice.description, status)
    return table


def choices_footer(choices: list[OperatorChoice], *, config_label: str) -> str:
    """One dim line summarising how many operators the config does not list."""
    n_missing = sum(1 for c in choices if not c.in_config)
    return f"[dim]{n_missing} of {len(choices)} operators are not listed in {config_label}.[/dim]"


def _display_config_summary(config: SimulationConfig | None) -> None:
    """Display a compact summary of the simulation configuration."""
    if config is None:
        console.print("[yellow]No configuration available.[/yellow]")
        return

    console.print()
    table = Table(
        title="Simulation Configuration",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Simulation Type", config.sim_type)
    table.add_row("Grid Shape", str(config.grid_shape))
    table.add_row("Lattice Type", config.lattice_type)
    table.add_row("Relaxation Time (tau)", str(config.tau))
    table.add_row("Time Steps", str(config.nt))
    table.add_row("Save Interval", str(config.save_interval))
    table.add_row("Results Directory", config.results_dir)
    if config.save_fields:
        table.add_row("Save Fields", ", ".join(config.save_fields))
    if config.plot_fields:
        table.add_row("Plot Fields", ", ".join(config.plot_fields))
    if config.animate_fields:
        table.add_row("Animate Fields", ", ".join(config.animate_fields))

    if config.is_multiphase:
        table.add_row("Kappa", str(config.kappa))
        table.add_row("Liquid Density", str(config.rho_l))
        table.add_row("Vapor Density", str(config.rho_v))
        table.add_row("Interface Width", str(config.interface_width))

    if config.force_enabled:
        active_forces = [
            f.name
            for f in config.__dataclass_fields__.values()
            if f.name.endswith("_force") and getattr(config, f.name) is not None
        ]
        table.add_row("Forces", ", ".join(active_forces) if active_forces else "enabled")

    console.print(table)
    console.print()


def _display_full_overview(config: SimulationConfig | None) -> None:
    """Display the full physical-parameter overview from build_overview()."""
    from rich.text import Text
    from src.simulation_io.analysis.physical_parameters import build_overview

    if config is None:
        console.print("[yellow]No configuration available.[/yellow]")
        return

    overview = build_overview(config)
    console.print(Panel(Text(overview), title="Simulation Overview", border_style="blue"))
    console.print()


def _display_sweep_summary(metadata: ArrayParameterSet) -> None:
    """Display detected parameter-sweep axes and total combinations."""
    console.print()
    table = Table(
        title="Parameter Sweep",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Values", style="green")

    for field_name in sorted(metadata.field_names):
        values = metadata.array_values.get(field_name, ())
        values_str = ", ".join(str(v) for v in values)
        table.add_row(field_name, values_str)

    console.print(table)
    console.print(f"[bold blue]Total combinations:[/bold blue] {metadata.total_combinations}")
    console.print()


def _display_summary(
    config: SimulationConfig | None,
    sweep_metadata: ArrayParameterSet | None,
    configs: list[SimulationConfig],
    *,
    overview: bool,
) -> None:
    """Display either a single-run config summary or a sweep summary."""
    if sweep_metadata is None:
        _display_config_summary(config)
        if overview:
            _display_full_overview(config)
    else:
        _display_sweep_summary(sweep_metadata)
        console.print("[dim]Preview of the first expanded configuration:[/dim]")
        _display_config_summary(configs[0])
        if overview:
            _display_full_overview(configs[0])


def _print_dry_run_message(sweep_metadata: ArrayParameterSet | None) -> None:
    if sweep_metadata is None:
        console.print("[yellow]Dry run mode - simulation not started[/yellow]")
    else:
        console.print("[yellow]Dry run mode - parameter sweep not started[/yellow]")
