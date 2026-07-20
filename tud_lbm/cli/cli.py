"""Command-line interface for TUD-LBM simulations.

Example Python usage::

    from config import SimulationConfig
    from setup import build_setup
    from runner import run, init_state

    config = SimulationConfig(grid_shape=(100, 100), tau=0.6, nt=10000)
    setup = build_setup(config)
    state = init_state(setup)
    final_state, trajectory = run(setup, state)
"""

import os
import sys
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.prompt import Prompt
from rich.table import Table
from tud_lbm.config import SimulationConfig
from tud_lbm.config.array_expansion import ArrayParameterSet

if TYPE_CHECKING:
    from tud_lbm.io.analysis.accelerations import Smoothing
    from tud_lbm.io.plotting import FigureBuilder

console = Console()
_CLI_SUBTITLE = "Delft University of Technology"

_SECTION_ALIAS_MAP = {
    "simulation_type": "",
    "multiphase": "",
    "output": "",
    "boundary_conditions": "bc_config",
    "wetting": "wetting_config",
    "hysteresis": "hysteresis_config",
    "chemical_step": "chemical_step_config",
}

_VISUAL_KINDS = {
    "plotting": "Field plots - rendered per timestep snapshot",
    "analysis": "Analysis plots - computed over a run's snapshot history",
}

# Which CLI commands consume each analysis kind — shown in --list-simulation-analysis output.
_ANALYSIS_USAGE = {
    "plotting": "tud-lbm animate, tud-lbm visualise",
    "analysis": "tud-lbm animate, tud-lbm visualise, tud-lbm compare",
}

# Number of timesteps for the no-gravity wetting equilibration phase.
_WETTING_INIT_NT = 50_000
_BOLD_CYAN = "bold cyan"


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


def _build_visual_table(kind: str, ops: dict, subtitle: str | None = None) -> Table:
    sub = subtitle if subtitle is not None else _VISUAL_KINDS.get(kind, "")
    table = Table(
        title=f"[bold magenta]{kind}[/bold magenta]  [dim]{sub}[/dim]",
        show_header=True,
        header_style=_BOLD_CYAN,
        title_justify="left",
    )
    table.add_column("Name", style="green", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Required keys", style="dim")
    for name in sorted(ops):
        target = ops[name].target
        doc = (getattr(target, "__doc__", None) or "").strip().splitlines()
        description = doc[0] if doc else "—"
        required = getattr(target, "required_keys", None)
        keys_str = ", ".join(required) if required else "—"
        table.add_row(name, description, keys_str)
    return table


def _build_standard_table(kind: str, ops: dict) -> Table:
    table = Table(
        title=f"[bold magenta]{kind}[/bold magenta]",
        show_header=True,
        header_style=_BOLD_CYAN,
        title_justify="left",
    )
    table.add_column("Name", style="green", no_wrap=True)
    table.add_column("Target", style="white")
    table.add_column("Metadata", style="dim")
    for name in sorted(ops):
        entry = ops[name]
        target = entry.target
        target_mod = getattr(target, "__module__", type(target).__module__)
        target_name = getattr(target, "__qualname__", getattr(target, "__name__", type(target).__name__))
        target_str = f"{target_mod}.{target_name}"
        meta_str = ", ".join(f"{k}={v!r}" for k, v in entry.metadata.items()) if entry.metadata else "—"
        table.add_row(name, target_str, meta_str)
    return table


_ANALYSIS_KINDS = frozenset({"plotting", "analysis"})


def _display_simulation_operators() -> None:
    """Display physics/model operators (excludes plotting and comparison kinds)."""
    import tud_lbm.io.plotting as _plotting_mod  # noqa: F401
    from tud_lbm.operators import load_all
    from tud_lbm.registry import get_operator_category
    from tud_lbm.registry import get_operators

    load_all()

    categories = sorted(c for c in get_operator_category() if c not in _ANALYSIS_KINDS)
    if not categories:
        console.print("[yellow]No simulation operators registered.[/yellow]")
        return

    n_total = sum(len(get_operators(c)) for c in categories)
    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]Simulation Operators[/bold blue]  ({n_total} total across {len(categories)} categories)",
        ),
    )
    console.print()

    for kind in categories:
        ops = get_operators(kind)
        console.print(_build_standard_table(kind, ops))
        console.print()


def _display_analysis_operators() -> None:
    """Display plotting and comparison operators with their CLI usage context."""
    import tud_lbm.io.plotting as _plotting_mod  # noqa: F401
    from tud_lbm.operators import load_all
    from tud_lbm.registry import get_operator_category
    from tud_lbm.registry import get_operators

    load_all()

    categories = sorted(c for c in get_operator_category() if c in _ANALYSIS_KINDS)
    if not categories:
        console.print("[yellow]No analysis operators registered.[/yellow]")
        return

    n_total = sum(len(get_operators(c)) for c in categories)
    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]Analysis Operators[/bold blue]  ({n_total} total across {len(categories)} categories)",
        ),
    )
    console.print()

    for kind in categories:
        ops = get_operators(kind)
        usage = _ANALYSIS_USAGE.get(kind, "")
        base = _VISUAL_KINDS.get(kind, "")
        subtitle = f"{base}  │  used by: {usage}" if usage else base
        console.print(_build_visual_table(kind, ops, subtitle=subtitle))
        console.print()


def _operator_description(target: object) -> str:
    """Return the first line of *target*'s docstring, or '—' when absent."""
    doc = (getattr(target, "__doc__", None) or "").strip().splitlines()
    return doc[0] if doc else "—"


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


def _prompt_snapshot_timesteps(available: list[int]) -> list[int]:
    """Interactively collect the timesteps to snapshot for the ``snapshot_fig`` plot."""
    console.print(f"[dim]Available timesteps: {', '.join(str(t) for t in available)}[/dim]")
    try:
        raw_ts = Prompt.ask("Enter snapshot timesteps (comma-separated)")
    except EOFError:
        return []
    return [int(tok.strip()) for tok in raw_ts.split(",") if tok.strip()]


def _configure_snapshot_fig(builder: "FigureBuilder", field_list: list[str] | None) -> None:
    """Prompt for and wire up snapshot timesteps when ``snapshot_fig`` is requested."""
    if not field_list or "snapshot_fig" not in field_list:
        return
    available = [t for t, _ in builder.sorted_timed_files()]
    requested_ts = _prompt_snapshot_timesteps(available)
    for op in builder.analysis_operators:
        if op.name == "snapshot_fig":
            op.timesteps = requested_ts


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
    from tud_lbm.io.analysis.physical_parameters import build_overview

    if config is None:
        console.print("[yellow]No configuration available.[/yellow]")
        return

    overview = build_overview(config)
    console.print(Panel(Text(overview), title="Simulation Overview", border_style="blue"))
    console.print()


def _run_simulation(config: SimulationConfig) -> str:
    """Run the simulation with the given configuration.

    Returns:
        The data directory path (``io.data_dir``) where snapshots were written.
    """
    from tud_lbm.config.jax_config import configure_jax

    configure_jax()

    from tud_lbm.io import SimulationIO
    from tud_lbm.pipeline.runner import init_state
    from tud_lbm.pipeline.runner import run
    from tud_lbm.pipeline.setup import build_setup

    simulation_setup = build_setup(config)
    state = init_state(simulation_setup)

    if config.init_type == "init_from_file" and int(state.t) > 0:
        console.print(f"[dim]Resuming from snapshot: t={int(state.t)}[/dim]")
        console.print()

    # Build the IO handler for streaming snapshots to disk.
    io = SimulationIO(
        base_dir=config.results_dir,
        config=config,
        simulation_name=config.simulation_name,
    )

    from tud_lbm.io.analysis.surface_tension import record_surface_tension

    config = record_surface_tension(config, io.run_dir)

    console.print("[bold green]Starting simulation...[/bold green]")
    console.print(f"[dim]Results directory: {io.run_dir}[/dim]")
    console.print()

    final_state, _ = run(
        simulation_setup,
        state,
        save_interval=config.save_interval,
        io_handler=io,
        skip_interval=config.skip_interval,
        save_fields=tuple(config.save_fields) if config.save_fields else None,
    )

    console.print("[bold green]Simulation completed![/bold green]")
    console.print(f"  Final timestep     : {int(final_state.t)}")
    console.print(f"  Snapshots saved to : {io.data_dir}")

    if config.plot_fields:
        from tud_lbm.io.plotting import FigureBuilder

        console.print("[dim]Generating plots...[/dim]")
        builder = FigureBuilder(config, io.run_dir)
        builder.build_all()
        console.print(
            "[bold green]Plotting complete![/bold green]",
        )
    return io.data_dir


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


def _run_parallel_sweep(
    configs: list[SimulationConfig],
    parameters_list: list[dict[str, Any]],
    *,
    max_workers: int | None,
    continue_on_error: bool,
) -> list[Any]:
    """Run a parameter sweep in parallel and save a manifest."""
    from tud_lbm.pipeline.parallel_runner import run_parallel_simulations
    from tud_lbm.pipeline.parallel_runner import save_sweep_log

    console.print("[bold green]Starting parallel parameter sweep...[/bold green]")
    console.print(f"[dim]Simulations: {len(configs)}[/dim]")
    console.print(f"[dim]Max workers: {max_workers if max_workers is not None else 'auto'}[/dim]")
    console.print()

    results = run_parallel_simulations(
        configs,
        max_workers=max_workers,
        parameters_list=parameters_list,
        verbose=True,
        continue_on_error=continue_on_error,
    )

    manifest_dir = Path(configs[0].results_dir).expanduser() / "sweep_manifest"
    save_sweep_log(results, manifest_dir)

    successful = sum(1 for result in results if result.status == "success")
    failed = sum(1 for result in results if result.status == "failed")

    console.print()
    console.print("[bold green]Parallel sweep completed.[/bold green]")
    console.print(f"  Successful runs : {successful}")
    console.print(f"  Failed runs     : {failed}")
    console.print(f"  Manifest folder : {manifest_dir}")

    return results


def _validate_cli_args(
    overrides: tuple[str, ...],
    config_path: str | None,
    *,
    init_wetting: bool = False,
    init_dir: str | None = None,
) -> None:
    """TRY301: raise lives here, outside the try-block in run()."""
    if overrides and not config_path:
        msg = "--override requires CONFIG_PATH"
        raise click.UsageError(msg)
    if init_wetting and not config_path:
        msg = "--init-wetting requires CONFIG_PATH"
        raise click.UsageError(msg)
    if init_dir and not config_path:
        msg = "--init-dir requires CONFIG_PATH"
        raise click.UsageError(msg)


def _load_raw_config(
    config_path: str,
    overrides: tuple[str, ...],
    *,
    init_dir: str | None = None,
) -> dict[str, Any]:
    """Load a TOML file, apply init_dir defaults and CLI overrides; return the raw dict."""
    from tud_lbm.config.adapter_toml import TomlAdapter

    console.print(f"[cyan]Loading configuration from:[/cyan] {config_path}")
    raw_config = TomlAdapter().load_raw(config_path) or {}
    if init_dir is not None:
        raw_config["init_dir"] = str(init_dir)
        raw_config["init_type"] = "init_from_file"
    _apply_overrides(raw_config, overrides)
    return raw_config


def _expand_raw_config(
    raw_config: dict[str, Any],
) -> tuple[list[SimulationConfig], SimulationConfig | None, ArrayParameterSet | None, list[dict[str, Any]] | None]:
    """Expand a raw config dict; return (configs, config, sweep_metadata, parameters_list)."""
    from tud_lbm.config.array_expansion import enumerate_configs
    from tud_lbm.config.array_expansion import expand_config

    configs, sweep_metadata = expand_config(raw_config)
    if sweep_metadata is None:
        return configs, configs[0], None, None
    parameters_list = [params for _, params, _ in enumerate_configs(raw_config)]
    return configs, None, sweep_metadata, parameters_list


def _load_config_interactive() -> tuple[list[SimulationConfig], SimulationConfig, None, None]:
    """Collect simulation parameters interactively; return the same 4-tuple as _load_config_from_file."""
    from tud_lbm.config import SimulationConfig

    console.print("[cyan]Interactive mode - creating default simulation config[/cyan]")

    grid_x = int(Prompt.ask("Grid size X", default="100"))
    grid_y = int(Prompt.ask("Grid size Y", default="100"))
    tau = float(Prompt.ask("Relaxation time (tau)", default="0.6"))
    nt = int(Prompt.ask("Number of timesteps", default="1000"))
    save_interval = int(Prompt.ask("Save interval", default=str(nt // 10)))

    config = SimulationConfig(
        grid_shape=(grid_x, grid_y),
        tau=tau,
        nt=nt,
        save_interval=save_interval,
    )
    return [config], config, None, None


_WETTING_PARAM_DEFAULTS: dict[str, float] = {
    "phi_left": 1.0,
    "phi_right": 1.0,
    "d_rho_left": 0.0,
    "d_rho_right": 0.0,
}


def _prompt_wetting_params(base_raw: dict[str, Any], *, no_prompt: bool) -> dict[str, float]:
    """Return wetting params, reading from config and only prompting for missing ones."""
    from_config: dict[str, Any] = base_raw.get("wetting_config") or {}
    missing = [k for k in _WETTING_PARAM_DEFAULTS if k not in from_config]

    if missing and not no_prompt:
        console.print("[cyan]Wetting init - enter missing wetting boundary parameters:[/cyan]")

    params: dict[str, float] = {}
    for key, default in _WETTING_PARAM_DEFAULTS.items():
        if key in from_config:
            params[key] = float(from_config[key])
        elif no_prompt:
            params[key] = default
        else:
            params[key] = float(Prompt.ask(f"  {key}", default=str(default)))

    if missing and not no_prompt:
        console.print()

    return params


def _expand_single_phase(raw_config: dict[str, Any], phase_name: str) -> SimulationConfig:
    from tud_lbm.config.array_expansion import expand_config

    configs, _ = expand_config(raw_config)
    if len(configs) != 1:
        msg = f"--init-wetting does not support parameter sweeps ({phase_name} expansion must yield exactly 1 config)."
        raise click.UsageError(msg)
    return configs[0]


def _build_wetting_init_raw(base_raw: dict[str, Any], wetting_params: dict[str, float]) -> dict[str, Any]:
    init_raw = deepcopy(base_raw)
    init_raw["sim_type"] = "multiphase_wetting"
    init_raw["init_type"] = "multiphase_bubbles"
    init_raw.pop("hysteresis_config", None)
    init_raw.pop("chemical_step_config", None)
    init_raw.pop("gravity_force", None)
    init_raw.pop("gravity_masked_force", None)
    if "bc_config" in init_raw:
        init_raw["bc_config"] = dict(init_raw["bc_config"])
    init_raw["nt"] = _WETTING_INIT_NT
    init_raw["save_interval"] = _WETTING_INIT_NT
    init_raw["output_format"] = "numpy"
    base_name = base_raw.get("simulation_name")
    init_raw["simulation_name"] = f"wetting_init_{base_name}" if base_name else "wetting_init"
    init_raw.setdefault("wetting_config", {}).update(wetting_params)
    return init_raw


def _build_wetting_gravity_raw(
    base_raw: dict[str, Any],
    wetting_params: dict[str, float],
    init_snapshot: str,
) -> dict[str, Any]:
    gravity_raw = deepcopy(base_raw)
    gravity_raw["init_type"] = "init_from_file"
    gravity_raw["init_dir"] = init_snapshot
    gravity_raw.setdefault("wetting_config", {}).update(wetting_params)
    return gravity_raw


def _run_two_phase_wetting_init(
    config_path: str,
    overrides: tuple[str, ...],
    *,
    no_prompt: bool,
    overview: bool,
) -> None:
    """Two-phase wetting initialisation.

    Phase 1 equilibrates the droplet without gravity for ``_WETTING_INIT_NT``
    steps and writes one final snapshot. Phase 2 then runs with gravity,
    initialised from the Phase 1 snapshot.
    """
    from tud_lbm.config.adapter_toml import TomlAdapter

    console.print(f"[cyan]Loading configuration from:[/cyan] {config_path}")
    base_raw = TomlAdapter().load_raw(config_path)
    _apply_overrides(base_raw, overrides)

    wetting_params = _prompt_wetting_params(base_raw, no_prompt=no_prompt)
    init_raw = _build_wetting_init_raw(base_raw, wetting_params)
    init_config = _expand_single_phase(init_raw, "Phase 1")

    console.print(Panel.fit("[bold cyan]Phase 1 - wetting equilibration (no gravity)[/bold cyan]"))
    _display_config_summary(init_config)
    if overview:
        _display_full_overview(init_config)

    if not no_prompt and not Confirm.ask("[bold]Start Phase 1?[/bold]", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    init_data_dir = _run_simulation(init_config)
    init_snapshot = str(Path(init_data_dir) / f"timestep_{_WETTING_INIT_NT}.npz")
    console.print(f"  Init snapshot     : {init_snapshot}")
    console.print()

    gravity_raw = _build_wetting_gravity_raw(base_raw, wetting_params, init_snapshot)
    gravity_config = _expand_single_phase(gravity_raw, "Phase 2")

    console.print(Panel.fit("[bold cyan]Phase 2 - full simulation with gravity[/bold cyan]"))
    _display_config_summary(gravity_config)
    if overview:
        _display_full_overview(gravity_config)

    if not no_prompt and not Confirm.ask("[bold]Start Phase 2?[/bold]", default=True):
        console.print("[yellow]Phase 2 cancelled.[/yellow]")
        return

    _run_simulation(gravity_config)


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


def _confirm_run(sweep_metadata: ArrayParameterSet | None, configs: list[SimulationConfig]) -> str:
    """Return 'yes', 'no', or 'override'."""
    if sweep_metadata is None:
        prompt_text = "[bold]Start simulation?[/bold]"
    else:
        prompt_text = f"[bold]Start parameter sweep ({len(configs)} simulations)?[/bold]"
    choice = Prompt.ask(
        f"{prompt_text} [[green]y[/green]/[red]n[/red]/[cyan]o[/cyan]=override]",
        choices=["y", "n", "o"],
        default="y",
        show_choices=False,
    )
    return {"y": "yes", "n": "no", "o": "override"}[choice]


def _check_sweep_errors(results: list[Any]) -> None:
    """TRY301: raise lives here, outside the try-block in run()."""
    failed = sum(1 for result in results if result.status == "failed")
    if failed > 0:
        msg = f"Parameter sweep completed with {failed} failed simulation(s)."
        raise RuntimeError(msg)


def _validate_run_dir_has_config(run_dir: str) -> Path:
    """TRY301: raise lives here, outside the try-block in animate()."""
    config_path = Path(run_dir) / "config.toml"
    if not config_path.exists():
        msg = f"No config.toml found in {run_dir}. Is this a valid run directory?"
        raise FileNotFoundError(msg)
    return config_path


def _execute_run(
    configs: list[SimulationConfig],
    config: SimulationConfig | None,
    sweep_metadata: ArrayParameterSet | None,
    parameters_list: list[dict[str, Any]] | None,
    max_workers: int | None,
    fail_fast: bool,
    run_compare: bool = False,
) -> None:
    """Dispatch to single-run or parallel-sweep execution."""
    if sweep_metadata is None:
        if config is not None:
            # Single-run path: _run_simulation returns the data directory (string)
            data_dir = _run_simulation(config)
            if run_compare:
                _run_compare_single(Path(data_dir).parent, config)
    else:
        results = _run_parallel_sweep(
            configs,
            parameters_list or [],
            max_workers=max_workers,
            continue_on_error=not fail_fast,
        )
        _check_sweep_errors(results)
        if run_compare:
            _run_compare_sweep(Path(configs[0].results_dir).expanduser())


def _run_compare_single(run_dir: Path, config: SimulationConfig) -> None:
    """Build CSV and comparison plots for a completed single run.

    Uses the in-memory config to avoid re-loading from disk which loses
    expanded/flattened fields.
    """
    from tud_lbm.io.plotting.run_comparison import _COMPARISON_DIR
    from tud_lbm.io.plotting.run_comparison import compare_runs
    from tud_lbm.io.plotting.simulation_csv import build_simulation_csv

    console.print("[dim]Running comparison analysis...[/dim]")
    csv_path = build_simulation_csv(run_dir, config)
    if csv_path is None:
        console.print("[yellow]--compare: CSV export skipped (unsupported sim_type).[/yellow]")
        return
    # compare_runs expects a parent directory that contains run dirs; passing
    # the single run directory will cause it to generate plots for that run.
    compare_runs(run_dir)
    console.print(f"[bold green]Comparison plots saved to:[/bold green] {run_dir / _COMPARISON_DIR}")


def _run_compare_sweep(results_dir: Path) -> None:
    """Build CSVs and comparison plots across all runs in a sweep directory."""
    from tud_lbm.io.plotting.run_comparison import _COMPARISON_DIR
    from tud_lbm.io.plotting.run_comparison import process_parent_dir

    console.print("[dim]Running comparison analysis...[/dim]")
    _n_runs, n_ok = process_parent_dir(results_dir)
    if n_ok == 0:
        console.print("[yellow]--compare: no runs produced CSV data.[/yellow]")
        return
    console.print(f"[bold green]Comparison plots saved to:[/bold green] {results_dir / _COMPARISON_DIR}")


def _print_run_banner() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold blue]TUD-LBM[/bold blue] - Lattice Boltzmann Method Solver",
            subtitle=_CLI_SUBTITLE,
        ),
    )
    console.print()


def _run_with_optional_overrides(
    *,
    raw_config: dict[str, Any] | None,
    configs: list[SimulationConfig],
    config: SimulationConfig | None,
    sweep_metadata: ArrayParameterSet | None,
    parameters_list: list[dict[str, Any]] | None,
    no_prompt: bool,
    overview: bool,
) -> tuple[list[SimulationConfig], SimulationConfig | None, ArrayParameterSet | None, list[dict[str, Any]] | None]:
    if no_prompt:
        return configs, config, sweep_metadata, parameters_list

    while True:
        decision = _confirm_run(sweep_metadata, configs)
        if decision == "no":
            console.print("[yellow]Simulation cancelled.[/yellow]")
            return [], None, None, None
        if decision == "yes":
            return configs, config, sweep_metadata, parameters_list
        if raw_config is None:
            console.print("[yellow]Inline overrides require a config file.[/yellow]")
            continue
        raw_expr = Prompt.ask("[cyan]Enter override[/cyan] [dim](e.g. tau=0.7)[/dim]")
        try:
            _apply_overrides(raw_config, (raw_expr,))
        except (ValueError, TypeError) as exc:
            console.print(f"[red]Invalid override: {exc}[/red]")
            continue
        configs, config, sweep_metadata, parameters_list = _expand_raw_config(raw_config)
        _display_summary(config, sweep_metadata, configs, overview=overview)


def _enable_debug_flags(*, debug_wetting: bool, debug_stability: bool) -> None:
    """Set the module-global debug flags in config_overview before setup/run traces."""
    if not (debug_wetting or debug_stability):
        return

    import tud_lbm.config.config_overview as _flags

    if debug_wetting:
        _flags.DEBUG_FLAG_WETTING = True
        console.print("[dim]Wetting debug logging enabled.[/dim]")
        console.print()

    if debug_stability:
        _flags.DEBUG_FLAG_STABILITY = True
        console.print("[dim]Stability diagnostics enabled (stability_log.csv + NaN guard).[/dim]")
        console.print()


@dataclass(frozen=True)
class RunFlags:
    """Boolean flags for `run`, bundled to keep `_run_impl`'s signature within S107's limit."""

    no_prompt: bool = False
    dry_run: bool = False
    list_operators: bool = False
    list_analysis: bool = False
    fail_fast: bool = False
    overview: bool = False
    debug_wetting: bool = False
    debug_stability: bool = False
    init_wetting: bool = False
    run_compare: bool = False


def _run_impl(
    config_path: str | None,
    overrides: tuple[str, ...],
    max_workers: int | None,
    init_dir: str | None,
    flags: RunFlags,
) -> bool:
    if flags.list_operators:
        _display_simulation_operators()
        return False

    if flags.list_analysis:
        _display_analysis_operators()
        return False

    _enable_debug_flags(debug_wetting=flags.debug_wetting, debug_stability=flags.debug_stability)

    _validate_cli_args(overrides, config_path, init_wetting=flags.init_wetting, init_dir=init_dir)

    if flags.init_wetting:
        if config_path is None:
            msg = "config_path is required for wetting initialisation"
            raise ValueError(msg)
        _run_two_phase_wetting_init(config_path, overrides, no_prompt=flags.no_prompt, overview=flags.overview)
        console.print()
        console.print(
            Panel.fit(
                "[bold green]Wetting initialisation complete![/bold green]",
                title="Success",
            ),
        )
        return False

    if config_path:
        raw_config = _load_raw_config(config_path, overrides, init_dir=init_dir)
        configs, config, sweep_metadata, parameters_list = _expand_raw_config(raw_config)
    else:
        raw_config = None
        configs, config, sweep_metadata, parameters_list = _load_config_interactive()

    _display_summary(config, sweep_metadata, configs, overview=flags.overview)
    if flags.dry_run:
        _print_dry_run_message(sweep_metadata)
        return False

    configs, config, sweep_metadata, parameters_list = _run_with_optional_overrides(
        raw_config=raw_config,
        configs=configs,
        config=config,
        sweep_metadata=sweep_metadata,
        parameters_list=parameters_list,
        no_prompt=flags.no_prompt,
        overview=flags.overview,
    )
    if not configs:
        return False

    _execute_run(configs, config, sweep_metadata, parameters_list, max_workers, flags.fail_fast, flags.run_compare)
    return True


@click.group()
@click.version_option(package_name="tud_lbm")
def cli() -> None:
    """TUD-LBM - Lattice Boltzmann Method Solver."""


@cli.command()
@click.argument("config_path", type=click.Path(exists=True), required=False)
@click.option(
    "--no-prompt",
    is_flag=True,
    help="Skip interactive prompts and use defaults for missing values",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Parse config and display summary without running simulation",
)
@click.option(
    "--list-simulation-operators",
    "list_operators",
    is_flag=True,
    help="List simulation operators (physics, models, lattices) and exit.",
)
@click.option(
    "--list-simulation-analysis",
    "list_analysis",
    is_flag=True,
    help="List analysis operators (plotting, comparison) with CLI usage context and exit.",
)
@click.option(
    "--max-workers",
    type=click.IntRange(min=1),
    default=None,
    help="Number of worker processes for parameter sweeps (default: auto)",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop a parameter sweep on first failed simulation",
)
@click.option(
    "--override",
    "overrides",
    multiple=True,
    help="Override config values using path=value (repeatable), "
    "e.g. --override simulation_type.simulation_name='new name'",
)
@click.option(
    "--overview",
    is_flag=True,
    help="Display the full physical-parameter overview in addition to the compact summary.",
)
@click.option(
    "--debug-wetting",
    is_flag=True,
    help="Enable wetting debug output (sets DEBUG_FLAG_WETTING in config_overview)",
)
@click.option(
    "--debug-stability",
    is_flag=True,
    help=(
        "Enable stability diagnostics: per-save-interval max|u|/max|grad mu|/rho-range/"
        "checkerboard logging to stability_log.csv plus a NaN guard that aborts the run "
        "(sets DEBUG_FLAG_STABILITY in config_overview; not propagated to sweep workers)"
    ),
)
@click.option(
    "--init-wetting",
    is_flag=True,
    help=(
        "Two-phase wetting initialisation: run nt=50000 without gravity to equilibrate "
        "the droplet, then run the full config using that snapshot as the initial condition"
    ),
)
@click.option(
    "--init-dir",
    "init_dir",
    default=None,
    type=click.Path(exists=True),
    help=(
        "Path to .npz snapshot to resume from. "
        "Sets init_type='init_from_file' automatically (overrideable via --override)."
    ),
)
@click.option(
    "--compare",
    "run_compare",
    is_flag=True,
    help="Generate comparison plots after a parameter sweep completes.",
)
def run(**cli_kwargs: object) -> None:
    """Run a TUD-LBM simulation from CONFIG_PATH.

    CONFIG_PATH is an optional path to a configuration file (.toml).
    If omitted, an interactive prompt collects parameters.

    Configuration files can include array parameters (list values) to
    automatically expand into parallel parameter sweeps. The --override
    flag allows replacing or creating config values from the command line
    before expansion.

    Examples:
        # Single simulation
        tud-lbm run example_for_test/config_simple.toml

        # Parameter sweep (if config has array fields)
        tud-lbm run example_for_test/config_parallel.toml

        # Override scalar field
        tud-lbm run config.toml --override tau=0.7

        # Override with quotes (field name may contain underscores)
        tud-lbm run config.toml --override 'simulation_type.simulation_name="new name"'

        # Create nested fields
        tud-lbm run config.toml --override 'gravity_force.force_g=5e-7'

        # Multiple overrides (applied in order)
        tud-lbm run config.toml --override tau=0.7 --override nt=500

        # Override array field to trigger sweep
        tud-lbm run config.toml --override 'tau=[0.6, 0.7, 0.8]' --max-workers 4

        # Dry run with preview (no execution)
        tud-lbm run config.toml --dry-run

        # Stop sweep on first failure
        tud-lbm run config.toml --fail-fast

        # Interactive mode
        tud-lbm run

        # List simulation operators (physics, models, lattices)
        tud-lbm run --list-simulation-operators

        # List analysis operators (plotting, comparison) with CLI usage
        tud-lbm run --list-simulation-analysis

        # Enable wetting debug output
        tud-lbm run config.toml --debug-wetting

        # Enable stability diagnostics (stability_log.csv + NaN guard)
        tud-lbm run config.toml --debug-stability

        # Two-phase wetting init: equilibrate without gravity then run with gravity
        tud-lbm run config.toml --init-wetting

        # Resume from a saved snapshot
        tud-lbm run config.toml --init-dir /path/to/timestep_1000.npz
    """
    _print_run_banner()

    config_path = cast("str | None", cli_kwargs["config_path"])
    max_workers = cast("int | None", cli_kwargs["max_workers"])
    overrides = cast("tuple[str, ...]", cli_kwargs["overrides"])
    init_dir = cast("str | None", cli_kwargs["init_dir"])
    flags = RunFlags(
        no_prompt=cast("bool", cli_kwargs["no_prompt"]),
        dry_run=cast("bool", cli_kwargs["dry_run"]),
        list_operators=cast("bool", cli_kwargs["list_operators"]),
        list_analysis=cast("bool", cli_kwargs["list_analysis"]),
        fail_fast=cast("bool", cli_kwargs["fail_fast"]),
        overview=cast("bool", cli_kwargs["overview"]),
        debug_wetting=cast("bool", cli_kwargs["debug_wetting"]),
        debug_stability=cast("bool", cli_kwargs["debug_stability"]),
        init_wetting=cast("bool", cli_kwargs["init_wetting"]),
        run_compare=cast("bool", cli_kwargs["run_compare"]),
    )

    try:
        completed = _run_impl(config_path, overrides, max_workers, init_dir, flags)
        if completed:
            console.print()
            console.print(
                Panel.fit(
                    "[bold green]Simulation, saving and plotting complete![/bold green]",
                    title="Success",
                ),
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if os.environ.get("TUD_LBM_DEBUG"):
            raise
        sys.exit(1)


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option(
    "--output",
    default=None,
    help="Output file path (.mp4 or .gif). Defaults to plots/animation.mp4 inside RUN_DIR.",
)
@click.option(
    "--fps",
    default=10,
    show_default=True,
    help="Frames per second for the output video.",
)
@click.option(
    "--no-prompt",
    "no_prompt",
    is_flag=True,
    help="Skip interactive field selection and use config defaults.",
)
def animate(run_dir: str, output: str | None, fps: int, no_prompt: bool) -> None:
    """Animate saved snapshots in RUN_DIR."""
    from tud_lbm.config import from_toml
    from tud_lbm.io.plotting import Animator
    from tud_lbm.registry import get_operators

    console.print()
    console.print(
        Panel.fit(
            "[bold blue]TUD-LBM[/bold blue] - Animation",
            subtitle=_CLI_SUBTITLE,
        ),
    )
    console.print()

    try:
        config_path = _validate_run_dir_has_config(run_dir)
        config = from_toml(str(config_path))

        console.print(f"[dim]Run directory : {run_dir}[/dim]")
        console.print(f"[dim]FPS           : {fps}[/dim]")

        default_fields = list(config.animate_fields or config.plot_fields or []) or None

        if no_prompt:
            fields = default_fields
        else:
            all_ops = {**get_operators("plotting"), **get_operators("analysis")}
            fields = _prompt_fields(all_ops, default_fields, "animation fields")

        if fields:
            console.print(f"[dim]Fields        : {', '.join(fields)}[/dim]")
        console.print()

        animator = Animator(config=config, run_dir=run_dir, fps=fps, fields=fields)
        output_path = animator.create(output)

        console.print(f"[bold green]Animation saved to:[/bold green] {output_path}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Animation interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if os.environ.get("TUD_LBM_DEBUG"):
            raise
        sys.exit(1)


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option(
    "--skip",
    default=0,
    show_default=True,
    help="Number of earliest timestep files to skip.",
)
@click.option(
    "--dpi",
    default=150,
    show_default=True,
    help="Resolution in dots per inch for saved figures.",
)
@click.option(
    "--fields",
    default=None,
    help="Comma-separated operator names to activate — skips the interactive prompt.",
)
@click.option(
    "--no-prompt",
    "no_prompt",
    is_flag=True,
    help="Skip interactive field selection and use config defaults.",
)
def visualise(run_dir: str, skip: int, dpi: int, fields: str | None, no_prompt: bool) -> None:
    """Build static figures for saved snapshots in RUN_DIR."""
    from tud_lbm.config import from_toml
    from tud_lbm.io.plotting import FigureBuilder
    from tud_lbm.registry import get_operators

    console.print()
    console.print(
        Panel.fit(
            "[bold blue]TUD-LBM[/bold blue] - Visualisation",
            subtitle=_CLI_SUBTITLE,
        ),
    )
    console.print()

    try:
        config_path = _validate_run_dir_has_config(run_dir)
        config = from_toml(str(config_path))

        console.print(f"[dim]Run directory : {run_dir}[/dim]")

        if fields:
            # Explicit --fields flag: parse and skip prompt.
            field_list: list[str] | None = [f.strip() for f in fields.split(",")]
        elif no_prompt:
            field_list = list(config.plot_fields or []) or None
        else:
            all_ops = {**get_operators("plotting"), **get_operators("analysis")}
            default_fields = list(config.plot_fields or []) or None
            field_list = _prompt_fields(all_ops, default_fields, "visualisation fields")

        if field_list:
            console.print(f"[dim]Fields        : {', '.join(field_list)}[/dim]")
        if skip:
            console.print(f"[dim]Skip          : {skip}[/dim]")
        console.print(f"[dim]DPI           : {dpi}[/dim]")
        console.print()

        builder = FigureBuilder(config=config, run_dir=run_dir, dpi=dpi, fields=field_list)
        _configure_snapshot_fig(builder, field_list)
        saved = builder.build_all(skip=skip)

        if not saved:
            console.print("[yellow]No figures produced. Check that the run directory contains snapshot files.[/yellow]")
        else:
            console.print(f"[bold green]{len(saved)} figure(s) saved to:[/bold green] {builder.plot_dir}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Visualisation interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if os.environ.get("TUD_LBM_DEBUG"):
            raise
        sys.exit(1)


@cli.command()
@click.argument("parent_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--no-prompt",
    "no_prompt",
    is_flag=True,
    help="Skip interactive operator selection; run CSV export only (no per-run analysis plots).",
)
def compare(parent_dir: str, no_prompt: bool) -> None:
    """Build CSV metrics and comparison plots for all runs in PARENT_DIR."""
    from tud_lbm.io.plotting.run_comparison import _COMPARISON_DIR
    from tud_lbm.io.plotting.run_comparison import process_parent_dir
    from tud_lbm.registry import get_operators

    console.print()
    console.print(
        Panel.fit(
            "[bold blue]TUD-LBM[/bold blue] - Comparison Analysis",
            subtitle=_CLI_SUBTITLE,
        ),
    )
    console.print()

    try:
        console.print(f"[dim]Parent directory : {parent_dir}[/dim]")

        if no_prompt:
            fields: list[str] | None = None
        else:
            comparison_ops = get_operators("analysis")
            fields = _prompt_fields(comparison_ops, None, "per-run comparison operators")

        if fields:
            console.print(f"[dim]Operators     : {', '.join(fields)}[/dim]")
        console.print()

        n_runs, n_ok = process_parent_dir(parent_dir, fields=fields)
        if n_runs == 0:
            console.print("[yellow]No simulation run directories found.[/yellow]")
            return
        if n_ok == 0:
            console.print("[yellow]No runs produced CSV data. Check sim_type and snapshot files.[/yellow]")
            return

        out_dir = Path(parent_dir) / _COMPARISON_DIR
        console.print()
        console.print(
            Panel.fit(
                f"[bold green]Comparison analysis complete![/bold green]  {n_ok}/{n_runs} run(s) processed",
                title="Success",
            ),
        )
        console.print(f"[bold green]Plots saved to:[/bold green] {out_dir}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Comparison interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if os.environ.get("TUD_LBM_DEBUG"):
            raise
        sys.exit(1)


@cli.command(name="regime-map")
@click.argument("dirs_txt", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--allowed-root",
    "allowed_roots",
    multiple=True,
    type=click.Path(exists=True, file_okay=False),
    help="Additional directory that a referenced run directory may resolve within, beyond the "
    "default results root (repeatable — e.g. one per HPC mount).",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory for regime_map.png (default: <dirs_txt parent>/regime_map_analysis).",
)
@click.option(
    "--smoothing",
    "smoothing",
    type=click.Choice(["raw", "savgol"]),
    default="raw",
    help="Acceleration-curve smoothing for peak detection: 'raw' (default, unsmoothed) or "
    "'savgol' (Savitzky-Golay filtered, reduces spikiness).",
)
def regime_map(dirs_txt: str, allowed_roots: tuple[str, ...], out_dir: str | None, smoothing: str) -> None:
    """Classify runs listed in DIRS_TXT into pinning/viscous/inertial/unknown and plot Bo_parallel vs Oh."""
    from tud_lbm.io.plotting.regime_map_plot import build_regime_map

    console.print()
    console.print(
        Panel.fit(
            "[bold blue]TUD-LBM[/bold blue] - Regime Map",
            subtitle=_CLI_SUBTITLE,
        ),
    )
    console.print()

    try:
        console.print(f"[dim]Run-dir list : {dirs_txt}[/dim]")
        console.print()

        out_path = build_regime_map(dirs_txt, allowed_roots, out_dir=out_dir, smoothing=cast("Smoothing", smoothing))
        if out_path is None:
            console.print("[yellow]No runs produced a usable classification.[/yellow]")
            sys.exit(1)

        console.print()
        console.print(
            Panel.fit(
                "[bold green]Regime map complete![/bold green]",
                title="Success",
            ),
        )
        console.print(f"[bold green]Plot saved to:[/bold green] {out_path}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Regime map analysis interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if os.environ.get("TUD_LBM_DEBUG"):
            raise
        sys.exit(1)


def _load_single_config(config_toml: str) -> SimulationConfig:
    """Load CONFIG_TOML and expand it to exactly one config; sweeps are rejected."""
    raw_config = _load_raw_config(config_toml, ())
    _configs, config, sweep_metadata, _params = _expand_raw_config(raw_config)
    if sweep_metadata is not None or config is None:
        msg = "analyse does not support parameter sweeps; remove list-valued fields from the config"
        raise click.UsageError(msg)
    return config


def _analyse_surface_tension(config: SimulationConfig, out_dir: Path) -> None:
    """Run the Young-Laplace calibration for *config* and report sigma."""
    if not config.is_multiphase:
        msg = f"surface tension requires a multiphase configuration; got sim_type='{config.sim_type}'"
        raise ValueError(msg)

    from tud_lbm.config.jax_config import configure_jax

    configure_jax()

    from tud_lbm.io.analysis.surface_tension import calibrate_surface_tension
    from tud_lbm.io.analysis.surface_tension.surface_tension import _PLOT_FILENAME

    sigma = calibrate_surface_tension(config, out_dir)

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]Surface tension: σ = {sigma:.6g}[/bold green]",
            title="Success",
        ),
    )
    console.print(f"[bold green]Calibration figure saved to:[/bold green] {out_dir / _PLOT_FILENAME}")


@cli.command()
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--surface-tension",
    "surface_tension",
    is_flag=True,
    help="Measure the lattice surface tension via the Young-Laplace droplet sweep for the configured "
    "EOS (cached results are reused; a cache miss runs the full droplet sweep).",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory for analysis outputs (default: the config file's directory).",
)
def analyse(config_toml: str, surface_tension: bool, out_dir: str | None) -> None:
    """Run standalone analyses for the configuration in CONFIG_TOML.

    Unlike the automatic calibration during `tud-lbm run` (which only
    triggers for EOS without a closed-form sigma), --surface-tension forces
    the Young-Laplace measurement for any supported multiphase EOS.

    Examples:
        # Measure surface tension for the configured EOS
        tud-lbm analyse config.toml --surface-tension

        # Write outputs somewhere other than the config's directory
        tud-lbm analyse config.toml --surface-tension --out-dir results/
    """
    if not surface_tension:
        msg = "select at least one analysis, e.g. --surface-tension"
        raise click.UsageError(msg)

    console.print()
    console.print(
        Panel.fit(
            "[bold blue]TUD-LBM[/bold blue] - Analysis",
            subtitle=_CLI_SUBTITLE,
        ),
    )
    console.print()

    try:
        target_dir = Path(out_dir) if out_dir is not None else Path(config_toml).resolve().parent
        config = _load_single_config(config_toml)
        console.print(f"[dim]Output directory : {target_dir}[/dim]")
        console.print()
        _analyse_surface_tension(config, target_dir)
    except KeyboardInterrupt:
        console.print("\n[yellow]Surface tension analysis interrupted by user.[/yellow]")
        sys.exit(130)
    except click.UsageError:
        raise
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if os.environ.get("TUD_LBM_DEBUG"):
            raise
        sys.exit(1)


if __name__ == "__main__":
    cli()
