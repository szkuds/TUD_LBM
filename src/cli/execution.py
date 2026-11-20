"""Running a simulation on behalf of the ``run`` command."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
import src.config.config_overview as _flags
from src.cli._console import console
from src.cli._console import success
from src.cli.config_loading import _build_config
from src.cli.config_loading import _load_config_interactive
from src.cli.config_loading import _load_raw_config
from src.cli.config_loading import _validate_cli_args
from src.cli.display import _display_analysis_operators
from src.cli.display import _display_simulation_operators
from src.cli.display import _display_summary
from src.cli.display import _print_dry_run_message
from src.cli.overrides import confirm_or_override
from src.cli.wetting_init import _run_two_phase_wetting_init
from src.config.run_config import COMPARISON_DIRNAME

if TYPE_CHECKING:
    from src.config import SimulationConfig


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

    from src.simulation_io.analysis.coexistence import calibrate_coexistence
    from src.simulation_io.analysis.surface_tension import record_surface_tension

    config = record_surface_tension(config, io.run_dir)
    calibrate_coexistence(config, io.run_dir)

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


def _execute_run(config: SimulationConfig, *, run_compare: bool = False) -> None:
    """Run *config*, then build its CSV and comparison plots when asked."""
    data_dir = _run_simulation(config)
    if run_compare:
        _run_compare_single(Path(data_dir).parent, config)


def _run_compare_single(run_dir: Path, config: SimulationConfig) -> None:
    """Build CSV and comparison plots for a completed single run.

    Uses the in-memory config to avoid re-loading from disk which loses
    expanded/flattened fields.
    """
    from src.cli.analysis_routing import analyse_run
    from src.simulation_io.plotting.run_comparison import compare_runs

    console.print("[dim]Running comparison analysis...[/dim]")
    if analyse_run(run_dir, config) is None:
        console.print("[yellow]--compare: CSV export skipped (no wetting boundary condition).[/yellow]")
        return
    # compare_runs expects a parent directory that contains run dirs; passing
    # the single run directory makes it plot that one run via rglob.
    compare_runs(run_dir)
    console.print(f"[bold green]Comparison plots saved to:[/bold green] {run_dir / COMPARISON_DIRNAME}")


def _run_with_optional_overrides(
    *,
    raw_config: dict[str, Any] | None,
    config: SimulationConfig,
    no_prompt: bool,
    overview: bool,
) -> SimulationConfig | None:
    """Confirm *config*, or rebuild it after an inline override; ``None`` when cancelled."""
    if no_prompt:
        return config

    def rebuild(raw: dict[str, Any]) -> SimulationConfig:
        rebuilt = _build_config(raw)
        _display_summary(rebuilt, overview=overview)
        return rebuilt

    confirmed = confirm_or_override(
        raw_config,
        config,
        prompt=lambda _: "[bold]Start simulation?[/bold]",
        rebuild=rebuild,
    )
    if confirmed is None:
        console.print("[yellow]Simulation cancelled.[/yellow]")
    return confirmed


def _enable_debug_flags(flags: RunFlags) -> None:
    """Set the module-global debug flags in config_overview before setup/run traces."""
    if flags.debug_wetting is not None:
        _flags.DEBUG_FLAG_WETTING = True
        _flags.DEBUG_WETTING_INTERVAL = flags.debug_wetting
        console.print(f"[dim]Wetting debug logging enabled (every {flags.debug_wetting} steps).[/dim]")
        console.print()

    if flags.debug_stability is not None:
        _flags.DEBUG_FLAG_STABILITY = True
        _flags.DEBUG_STABILITY_INTERVAL = flags.debug_stability
        console.print(
            "[dim]Stability diagnostics enabled "
            f"(stability_log.csv + NaN guard, every {flags.debug_stability} steps).[/dim]",
        )
        console.print()


@dataclass(frozen=True)
class RunFlags:
    """Option values for `run`, bundled to keep `_run_impl`'s signature within S107's limit.

    All boolean but the three optional-value flags and the phase-scoped overrides.
    `debug_wetting`, `debug_stability` and `init_wetting` are `None` when the flag
    is absent and otherwise carry its number: the `--debug-wetting [INTERVAL]` and
    `--debug-stability [INTERVAL]` sampling intervals, and the `--init-wetting [NT]`
    Phase 1 length. `override_phase1` and `override_phase2` ride here rather than on
    `_run_impl`'s signature, which is already at PLR0913's five-argument limit.
    """

    no_prompt: bool = False
    dry_run: bool = False
    list_operators: bool = False
    list_analysis: bool = False
    overview: bool = False
    debug_wetting: int | None = None
    debug_stability: int | None = None
    init_wetting: int | None = None
    override_phase1: tuple[str, ...] = ()
    override_phase2: tuple[str, ...] = ()
    run_compare: bool = False
    continue_run: bool = False


def _run_impl(
    config_path: str | None,
    overrides: tuple[str, ...],
    init_dir: str | None,
    flags: RunFlags,
) -> bool:
    if flags.list_operators:
        _display_simulation_operators()
        return False

    if flags.list_analysis:
        _display_analysis_operators()
        return False

    _enable_debug_flags(flags)

    _validate_cli_args(
        overrides,
        config_path,
        init_wetting=flags.init_wetting is not None,
        init_dir=init_dir,
        continue_run=flags.continue_run,
        override_phase1=flags.override_phase1,
        override_phase2=flags.override_phase2,
    )

    if flags.init_wetting is not None:
        if config_path is None:
            msg = "config_path is required for wetting initialisation"
            raise ValueError(msg)
        _run_two_phase_wetting_init(
            config_path,
            overrides,
            no_prompt=flags.no_prompt,
            overview=flags.overview,
            init_nt=flags.init_wetting,
            override_phase1=flags.override_phase1,
            override_phase2=flags.override_phase2,
            dry_run=flags.dry_run,
        )
        if not flags.dry_run:
            success("Wetting initialisation complete!")
        return False

    if config_path:
        raw_config = _load_raw_config(config_path, overrides, init_dir=init_dir, continue_run=flags.continue_run)
        config = _build_config(raw_config)
    else:
        raw_config = None
        config = _load_config_interactive()

    _display_summary(config, overview=flags.overview)
    if flags.dry_run:
        _print_dry_run_message()
        return False

    confirmed = _run_with_optional_overrides(
        raw_config=raw_config,
        config=config,
        no_prompt=flags.no_prompt,
        overview=flags.overview,
    )
    if confirmed is None:
        return False

    _execute_run(confirmed, run_compare=flags.run_compare)
    return True
