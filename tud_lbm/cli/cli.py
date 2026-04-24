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
from pathlib import Path
from typing import Any
import click
import tomllib
from config.array_expansion import ArrayParameterSet
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.prompt import Prompt
from rich.table import Table
from tud_lbm.config import SimulationConfig

console = Console()

_SECTION_ALIAS_MAP = {
    "simulation_type": "",
    "multiphase": "",
    "output": "",
    "boundary_conditions": "bc_config",
    "wetting": "wetting_config",
    "hysteresis": "hysteresis_config",
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


def _normalize_override_path(path: str) -> list[str]:
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
        ValueError: If path is empty or becomes empty after normalization.

    Examples:
        _normalize_override_path('simulation_type.tau')
        # ['tau']

        _normalize_override_path('gravity_force.force_g')
        # ['gravity_force', 'force_g']

        _normalize_override_path('boundary_conditions.top')
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
        path: Dotted-path string (normalized or already valid).
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
    parts = _normalize_override_path(path)

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


def _display_operators() -> None:
    """Display all registered operators grouped by kind in Rich tables."""
    from operators import load_all
    from registry import OPERATOR_REGISTRY
    from registry import get_operator_category
    from registry import get_operators

    load_all()

    categories = sorted(get_operator_category())

    if not categories:
        console.print("[yellow]No operators registered.[/yellow]")
        return

    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]Registered Operators[/bold blue]  "
            f"({len(OPERATOR_REGISTRY)} total across {len(categories)} categories)",
        ),
    )
    console.print()

    for kind in categories:
        ops = get_operators(kind)
        table = Table(
            title=f"[bold magenta]{kind}[/bold magenta]",
            show_header=True,
            header_style="bold cyan",
            title_justify="left",
        )
        table.add_column("Name", style="green", no_wrap=True)
        table.add_column("Target", style="white")
        table.add_column("Metadata", style="dim")

        for name in sorted(ops):
            entry = ops[name]
            target = entry.target
            target_mod = getattr(target, "__module__", type(target).__module__)
            target_name = getattr(
                target,
                "__qualname__",
                getattr(target, "__name__", type(target).__name__),
            )
            target_str = f"{target_mod}.{target_name}"

            meta_str = ""
            if entry.metadata:
                meta_str = ", ".join(f"{k}={v!r}" for k, v in entry.metadata.items())

            table.add_row(name, target_str, meta_str or "—")

        console.print(table)
        console.print()


def _display_config_summary(config: SimulationConfig | None) -> None:
    """Display a summary of the simulation configuration."""
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
    table.add_row("Relaxation Time (τ)", str(config.tau))
    table.add_row("Time Steps", str(config.nt))
    table.add_row("Save Interval", str(config.save_interval))
    table.add_row("Results Directory", config.results_dir)
    if config.save_fields:
        table.add_row("Save Fields", ", ".join(config.save_fields))
    if config.plot_fields:
        table.add_row("Plot Fields", ", ".join(config.plot_fields))

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


def _run_simulation(config: SimulationConfig) -> None:
    """Run the simulation with the given configuration."""
    from config.jax_config import configure_jax

    configure_jax()

    from runner import init_state
    from runner import run
    from setup import build_setup
    from util.io import SimulationIO

    simulation_setup = build_setup(config)
    state = init_state(simulation_setup)

    # Build the IO handler for streaming snapshots to disk.
    io = SimulationIO(
        base_dir=config.results_dir,
        config=config,
        simulation_name=config.simulation_name,
    )

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
        from util.plotting import FigureBuilder

        console.print("[dim]Generating plots...[/dim]")
        builder = FigureBuilder(config, io.run_dir)
        builder.build_all()
        console.print(
            "[bold green]Plotting complete![/bold green]",
        )
    return final_state


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
    from runner.parallel_runner import run_parallel_simulations
    from runner.parallel_runner import save_sweep_log

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


def _validate_cli_args(overrides: tuple[str, ...], config_path: str | None) -> None:
    """TRY301: raise lives here, outside the try-block in main()."""
    if overrides and not config_path:
        msg = "--override requires CONFIG_PATH"
        raise click.UsageError(msg)


def _load_config_from_file(
    config_path: str,
    overrides: tuple[str, ...],
) -> tuple[list[SimulationConfig], SimulationConfig | None, ArrayParameterSet | None, list[dict[str, Any]] | None]:
    """Load and expand a TOML config file; return (configs, config, sweep_metadata, parameters_list)."""
    from config.adapter_toml import TomlAdapter
    from config.array_expansion import enumerate_configs
    from config.array_expansion import expand_config

    console.print(f"[cyan]Loading configuration from:[/cyan] {config_path}")
    raw_config = TomlAdapter().load_raw(config_path)
    _apply_overrides(raw_config, overrides)
    configs, sweep_metadata = expand_config(raw_config)

    if sweep_metadata is None:
        return configs, configs[0], None, None

    parameters_list = [params for _, params, _ in enumerate_configs(raw_config)]
    return configs, None, sweep_metadata, parameters_list


def _load_config_interactive() -> tuple[list[SimulationConfig], SimulationConfig, None, None]:
    """Collect simulation parameters interactively; return the same 4-tuple as _load_config_from_file."""
    # TODO: the interactive mode can be extended further. Plotting is not yet added for example_for_test.
    from config import SimulationConfig

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


def _display_summary(
    config: SimulationConfig | None,
    sweep_metadata: ArrayParameterSet | None,
    configs: list[SimulationConfig],
) -> None:
    """Display either a single-run config summary or a sweep summary."""
    if sweep_metadata is None:
        _display_config_summary(config)
    else:
        _display_sweep_summary(sweep_metadata)
        console.print("[dim]Preview of the first expanded configuration:[/dim]")
        _display_config_summary(configs[0])


def _print_dry_run_message(sweep_metadata: ArrayParameterSet | None) -> None:
    if sweep_metadata is None:
        console.print("[yellow]Dry run mode - simulation not started[/yellow]")
    else:
        console.print("[yellow]Dry run mode - parameter sweep not started[/yellow]")


def _confirm_run(sweep_metadata: ArrayParameterSet | None, configs: list[SimulationConfig]) -> bool:
    if sweep_metadata is None:
        prompt_text = "[bold]Start simulation?[/bold]"
    else:
        prompt_text = f"[bold]Start parameter sweep ({len(configs)} simulations)?[/bold]"
    return Confirm.ask(prompt_text, default=True)


def _check_sweep_errors(results: list[Any]) -> None:
    """TRY301: raise lives here, outside the try-block in main()."""
    failed = sum(1 for result in results if result.status == "failed")
    if failed > 0:
        msg = f"Parameter sweep completed with {failed} failed simulation(s)."
        raise RuntimeError(msg)


def _execute_run(
    configs: list[SimulationConfig],
    config: SimulationConfig | None,
    sweep_metadata: ArrayParameterSet | None,
    parameters_list: list[dict[str, Any]] | None,
    max_workers: int | None,
    fail_fast: bool,
) -> None:
    """Dispatch to single-run or parallel-sweep execution."""
    if sweep_metadata is None:
        _run_simulation(config)
    else:
        results = _run_parallel_sweep(
            configs,
            parameters_list or [],
            max_workers=max_workers,
            continue_on_error=not fail_fast,
        )
        _check_sweep_errors(results)


@click.command()
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
    help="List all registered operators with metadata and exit",
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
@click.version_option(package_name="tud_lbm")
def main(
    config_path: str,
    no_prompt: bool,
    dry_run: bool,
    list_operators: bool,
    max_workers: int | None,
    fail_fast: bool,
    overrides: tuple[str, ...],
) -> None:
    """Run a TUD-LBM simulation.

    CONFIG_PATH is an optional path to a configuration file (.toml).
    If omitted, an interactive prompt collects parameters.

    Configuration files can include array parameters (list values) to
    automatically expand into parallel parameter sweeps. The --override
    flag allows replacing or creating config values from the command line
    before expansion.

    Examples:
        # Single simulation
        tud_lbm example_for_test/config_simple.toml

        # Parameter sweep (if config has array fields)
        tud_lbm example_for_test/config_parallel.toml

        # Override scalar field
        tud_lbm config.toml --override tau=0.7

        # Override with quotes (field name may contain underscores)
        tud_lbm config.toml --override 'simulation_type.simulation_name="new name"'

        # Create nested fields
        tud_lbm config.toml --override 'gravity_force.force_g=5e-7'

        # Multiple overrides (applied in order)
        tud_lbm config.toml --override tau=0.7 --override nt=500

        # Override array field to trigger sweep
        tud_lbm config.toml --override 'tau=[0.6, 0.7, 0.8]' --max-workers 4

        # Dry run with preview (no execution)
        tud_lbm config.toml --dry-run

        # Stop sweep on first failure
        tud_lbm config.toml --fail-fast

        # Interactive mode
        tud_lbm

        # List all registered operators
        tud_lbm --list-simulation-operators
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold blue]TUD-LBM[/bold blue] - Lattice Boltzmann Method Solver",
            subtitle="Delft University of Technology",
        ),
    )
    console.print()

    try:
        if list_operators:
            _display_operators()
            return

        # TRY301: validation raise is inside _validate_cli_args, not this try-block
        _validate_cli_args(overrides, config_path)

        configs, config, sweep_metadata, parameters_list = (
            _load_config_from_file(config_path, overrides) if config_path else _load_config_interactive()
        )

        _display_summary(config, sweep_metadata, configs)

        if dry_run:
            _print_dry_run_message(sweep_metadata)
            return

        if not no_prompt and not _confirm_run(sweep_metadata, configs):
            console.print("[yellow]Simulation cancelled.[/yellow]")
            return

        # TRY301: sweep error raise is inside _check_sweep_errors, not this try-block
        _execute_run(configs, config, sweep_metadata, parameters_list, max_workers, fail_fast)

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


if __name__ == "__main__":
    main()
