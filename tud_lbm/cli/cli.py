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
from copy import deepcopy
from pathlib import Path
from typing import Any
import click
import tomllib
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.prompt import Prompt
from rich.table import Table
from tud_lbm.config import SimulationConfig
from tud_lbm.config.array_expansion import ArrayParameterSet

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
    "analysis": "Analysis plots - computed over snapshot history",
}

# Number of timesteps for the no-gravity wetting equilibration phase.
_WETTING_INIT_NT = 50_000


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
    # Import plotting package for side-effect registration of plotting/analysis operators.
    import tud_lbm.io.plotting as _plotting_mod  # noqa: F401
    from tud_lbm.operators import load_all
    from tud_lbm.registry import OPERATOR_REGISTRY
    from tud_lbm.registry import get_operator_category
    from tud_lbm.registry import get_operators

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

        if kind in _VISUAL_KINDS:
            subtitle = _VISUAL_KINDS[kind]
            table = Table(
                title=f"[bold magenta]{kind}[/bold magenta]  [dim]{subtitle}[/dim]",
                show_header=True,
                header_style="bold cyan",
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
        else:
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
                meta_str = ", ".join(f"{k}={v!r}" for k, v in entry.metadata.items()) if entry.metadata else "—"
                table.add_row(name, target_str, meta_str)

        console.print(table)
        console.print()


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
    from tud_lbm.io.physical_parameters import build_overview

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
    raw_config = TomlAdapter().load_raw(config_path)
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
    init_raw["simulation_name"] = "wetting_init"
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
    initialized from the Phase 1 snapshot.
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
    from tud_lbm.io.plotting.analysis import _COMPARISON_DIR
    from tud_lbm.io.plotting.analysis import build_simulation_csv
    from tud_lbm.io.plotting.analysis import compare_runs

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
    from tud_lbm.io.plotting.analysis import _COMPARISON_DIR
    from tud_lbm.io.plotting.analysis import process_parent_dir

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


def _run_impl(
    config_path: str,
    no_prompt: bool,
    dry_run: bool,
    list_operators: bool,
    max_workers: int | None,
    fail_fast: bool,
    overrides: tuple[str, ...],
    overview: bool,
    debug_wetting: bool,
    init_wetting: bool,
    init_dir: str | None,
    run_compare: bool = False,
) -> bool:
    if list_operators:
        _display_operators()
        return False

    if debug_wetting:
        import tud_lbm.config.config_overview as _flags

        _flags.DEBUG_FLAG = True
        console.print("[dim]Wetting debug logging enabled.[/dim]")
        console.print()

    _validate_cli_args(overrides, config_path, init_wetting=init_wetting, init_dir=init_dir)

    if init_wetting:
        _run_two_phase_wetting_init(config_path, overrides, no_prompt=no_prompt, overview=overview)
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

    _display_summary(config, sweep_metadata, configs, overview=overview)
    if dry_run:
        _print_dry_run_message(sweep_metadata)
        return False

    configs, config, sweep_metadata, parameters_list = _run_with_optional_overrides(
        raw_config=raw_config,
        configs=configs,
        config=config,
        sweep_metadata=sweep_metadata,
        parameters_list=parameters_list,
        no_prompt=no_prompt,
        overview=overview,
    )
    if not configs:
        return False

    _execute_run(configs, config, sweep_metadata, parameters_list, max_workers, fail_fast, run_compare)
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
@click.option(
    "--overview",
    is_flag=True,
    help="Display the full physical-parameter overview in addition to the compact summary.",
)
@click.option(
    "--debug-wetting",
    is_flag=True,
    help="Enable wetting debug output (sets DEBUG_FLAG in config_overview)",
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
def run(
    config_path: str,
    no_prompt: bool,
    dry_run: bool,
    list_operators: bool,
    max_workers: int | None,
    fail_fast: bool,
    overrides: tuple[str, ...],
    overview: bool,
    debug_wetting: bool,
    init_wetting: bool,
    init_dir: str | None,
    run_compare: bool,
) -> None:
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

        # List all registered operators
        tud-lbm run --list-simulation-operators

        # Enable wetting debug output
        tud-lbm run config.toml --debug-wetting

        # Two-phase wetting init: equilibrate without gravity then run with gravity
        tud-lbm run config.toml --init-wetting

        # Resume from a saved snapshot
        tud-lbm run config.toml --init-dir /path/to/timestep_1000.npz
    """
    _print_run_banner()

    try:
        completed = _run_impl(
            config_path,
            no_prompt,
            dry_run,
            list_operators,
            max_workers,
            fail_fast,
            overrides,
            overview,
            debug_wetting,
            init_wetting,
            init_dir,
            run_compare,
        )
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
def animate(run_dir: str, output: str | None, fps: int) -> None:
    """Animate saved snapshots in RUN_DIR."""
    from tud_lbm.config import from_toml
    from tud_lbm.io.plotting import Animator

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
        fields = config.animate_fields or config.plot_fields
        if fields:
            console.print(f"[dim]Fields        : {', '.join(fields)}[/dim]")
        console.print(f"[dim]FPS           : {fps}[/dim]")
        console.print()

        animator = Animator(config=config, run_dir=run_dir, fps=fps)
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
    help="Comma-separated list of plot operator names to activate (overrides config.plot_fields).",
)
def visualise(run_dir: str, skip: int, dpi: int, fields: str | None) -> None:
    """Build static figures for saved snapshots in RUN_DIR."""
    from tud_lbm.config import from_toml
    from tud_lbm.io.plotting import FigureBuilder

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

        field_list = [f.strip() for f in fields.split(",")] if fields else None

        console.print(f"[dim]Run directory : {run_dir}[/dim]")
        active_fields = field_list or config.plot_fields
        if active_fields:
            console.print(f"[dim]Fields        : {', '.join(active_fields)}[/dim]")
        if skip:
            console.print(f"[dim]Skip          : {skip}[/dim]")
        console.print(f"[dim]DPI           : {dpi}[/dim]")
        console.print()

        builder = FigureBuilder(config=config, run_dir=run_dir, dpi=dpi, fields=field_list)
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
def compare(parent_dir: str) -> None:
    """Build CSV metrics and comparison plots for all runs in PARENT_DIR."""
    from tud_lbm.io.plotting.analysis import _COMPARISON_DIR
    from tud_lbm.io.plotting.analysis import process_parent_dir

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
        console.print()

        n_runs, n_ok = process_parent_dir(parent_dir)
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
        console.print("\n[yellow]Analysis interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if os.environ.get("TUD_LBM_DEBUG"):
            raise
        sys.exit(1)


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    add_help_option=False,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def main(args: tuple[str, ...]) -> None:
    """Backward-compatible shim for launchers still targeting ``...:main``.

    Supports both legacy direct run style (``tud-lbm CONFIG.toml``) and
    newer subcommand style (``tud-lbm run ...`` / ``tud-lbm animate ...``).
    """
    forwarded = list(args)

    # New-style help/version and subcommands should use the command group.
    if forwarded and forwarded[0] in {"--help", "-h", "--version", "animate", "visualise", "compare"}:
        cli.main(args=forwarded, standalone_mode=False)
        return

    # Allow stale wrappers to pass through the explicit "run" subcommand token.
    if forwarded and forwarded[0] == "run":
        forwarded = forwarded[1:]

    run.main(args=forwarded, standalone_mode=False)


if __name__ == "__main__":
    cli()
