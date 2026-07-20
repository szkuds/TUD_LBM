"""Running simulations and sweeps on behalf of the ``run`` command."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from rich.prompt import Prompt
from src.cli._console import console
from src.cli._console import success
from src.cli.config_loading import _expand_raw_config
from src.cli.config_loading import _load_config_interactive
from src.cli.config_loading import _load_raw_config
from src.cli.config_loading import _validate_cli_args
from src.cli.display import _display_analysis_operators
from src.cli.display import _display_simulation_operators
from src.cli.display import _display_summary
from src.cli.display import _print_dry_run_message
from src.cli.overrides import _apply_overrides
from src.cli.wetting_init import _run_two_phase_wetting_init

if TYPE_CHECKING:
    from src.config import SimulationConfig
    from src.config.array_expansion import ArrayParameterSet


def _run_simulation(config: SimulationConfig) -> str:
    """Run the simulation with the given configuration.

    Returns:
        The data directory path (``simulation_io.data_dir``) where snapshots were written.
    """
    from src.config.jax_config import configure_jax

    configure_jax()

    from src.pipeline.runner import init_state
    from src.pipeline.runner import run
    from src.pipeline.setup import build_setup
    from src.simulation_io import SimulationIO

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

    from src.simulation_io.analysis.surface_tension import record_surface_tension

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
        from src.simulation_io.plotting import FigureBuilder

        console.print("[dim]Generating plots...[/dim]")
        builder = FigureBuilder(config, io.run_dir)
        builder.build_all()
        console.print(
            "[bold green]Plotting complete![/bold green]",
        )
    return io.data_dir


def _run_parallel_sweep(
    configs: list[SimulationConfig],
    parameters_list: list[dict[str, Any]],
    *,
    max_workers: int | None,
    continue_on_error: bool,
) -> list[Any]:
    """Run a parameter sweep in parallel and save a manifest."""
    from src.pipeline.parallel_runner import run_parallel_simulations
    from src.pipeline.parallel_runner import save_sweep_log

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
    from src.cli.analysis_routing import analyse_run
    from src.simulation_io.plotting.run_comparison import _COMPARISON_DIR
    from src.simulation_io.plotting.run_comparison import compare_runs

    console.print("[dim]Running comparison analysis...[/dim]")
    if analyse_run(run_dir, config) is None:
        console.print("[yellow]--compare: CSV export skipped (unsupported sim_type).[/yellow]")
        return
    # compare_runs expects a parent directory that contains run dirs; passing
    # the single run directory makes it plot that one run via rglob.
    compare_runs(run_dir)
    console.print(f"[bold green]Comparison plots saved to:[/bold green] {run_dir / _COMPARISON_DIR}")


def _run_compare_sweep(results_dir: Path) -> None:
    """Build CSVs and comparison plots across all runs in a sweep directory."""
    from src.cli.analysis_routing import analyse_tree
    from src.simulation_io.plotting.run_comparison import _COMPARISON_DIR

    console.print("[dim]Running comparison analysis...[/dim]")
    _n_runs, n_ok = analyse_tree(results_dir)
    if n_ok == 0:
        console.print("[yellow]--compare: no runs produced CSV data.[/yellow]")
        return
    console.print(f"[bold green]Comparison plots saved to:[/bold green] {results_dir / _COMPARISON_DIR}")


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

    import src.config.config_overview as _flags

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
    continue_run: bool = False


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

    _validate_cli_args(
        overrides,
        config_path,
        init_wetting=flags.init_wetting,
        init_dir=init_dir,
        continue_run=flags.continue_run,
    )

    if flags.init_wetting:
        if config_path is None:
            msg = "config_path is required for wetting initialisation"
            raise ValueError(msg)
        _run_two_phase_wetting_init(config_path, overrides, no_prompt=flags.no_prompt, overview=flags.overview)
        success("Wetting initialisation complete!")
        return False

    if config_path:
        raw_config = _load_raw_config(config_path, overrides, init_dir=init_dir, continue_run=flags.continue_run)
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
