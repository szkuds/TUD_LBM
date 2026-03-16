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

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.prompt import Prompt
from rich.table import Table

console = Console()


def _ensure_operators_imported() -> None:
    """Import all operator packages so the global registry is fully populated."""
    import operators.collision  # noqa: F401
    import operators.boundary  # noqa: F401
    import operators.differential  # noqa: F401
    import operators.equilibrium  # noqa: F401
    import operators.streaming  # noqa: F401
    import operators.macroscopic  # noqa: F401
    import operators.force  # noqa: F401
    import operators.initialise.factory  # noqa: F401
    import operators.wetting  # noqa: F401
    import setup.lattice  # noqa: F401
    import util.plotting # noqa: F401


def _display_operators() -> None:
    """Display all registered operators grouped by kind in Rich tables."""
    from registry import OPERATOR_REGISTRY, get_operator_category, get_operators

    _ensure_operators_imported()

    categories = sorted(get_operator_category())

    if not categories:
        console.print("[yellow]No operators registered.[/yellow]")
        return

    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]Registered Operators[/bold blue]  "
            f"({len(OPERATOR_REGISTRY)} total across {len(categories)} categories)",
        )
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
            # Build a human-readable target description
            if isinstance(target, type):
                target_str = f"{target.__module__}.{target.__qualname__}"
            else:
                target_str = f"{target.__module__}.{target.__qualname__}"

            meta_str = ""
            if entry.metadata:
                meta_str = ", ".join(f"{k}={v!r}" for k, v in entry.metadata.items())

            table.add_row(name, target_str, meta_str or "—")

        console.print(table)
        console.print()


def _display_config_summary(config) -> None:
    """Display a summary of the simulation configuration."""
    console.print()

    table = Table(
        title="Simulation Configuration", show_header=True, header_style="bold magenta"
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
        table.add_row("Force", "enabled")

    console.print(table)
    console.print()


def _run_simulation(config):
    """Run the simulation with the given configuration."""
    from config.jax_config import configure_jax

    configure_jax()

    from setup import build_setup
    from runner import run, init_state
    from util.io import SimulationIO

    setup = build_setup(config)
    state = init_state(setup)

    # Build the IO handler for streaming snapshots to disk.
    io = SimulationIO(
        base_dir=config.results_dir,
        config=config.to_dict(),
        simulation_name=config.simulation_name,
    )

    console.print("[bold green]Starting simulation...[/bold green]")
    console.print(f"[dim]Results directory: {io.run_dir}[/dim]")
    console.print()

    final_state, _ = run(
        setup,
        state,
        save_interval=config.save_interval,
        io_handler=io,
        skip_interval=config.skip_interval,
        save_fields=tuple(config.save_fields) if config.save_fields else None,
    )

    console.print("[bold green]Simulation completed![/bold green]")
    return final_state


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
@click.version_option(package_name="tud_lbm")
def main(
    config_path: str, no_prompt: bool, dry_run: bool, list_operators: bool
) -> None:
    """Run a TUD-LBM simulation.

    CONFIG_PATH is an optional path to a configuration file (.toml).
    If omitted, an interactive prompt collects parameters.

    Examples:

        tud_lbm example/config_simple.toml
        tud_lbm example/config_complex.toml --dry-run
        tud_lbm --list-simulation-operators
        tud_lbm                              # interactive mode
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold blue]TUD-LBM[/bold blue] - Lattice Boltzmann Method Solver",
            subtitle="Delft University of Technology",
        )
    )
    console.print()

    try:
        # Handle --list-simulation-operators early
        if list_operators:
            _display_operators()
            return

        if config_path:
            # ── Load config file ─────────────────────────────────────
            from config import from_toml

            console.print(f"[cyan]Loading configuration from:[/cyan] {config_path}")
            config = from_toml(config_path)
        else:
            # ── Interactive mode ─────────────────────────────────────
            from config import SimulationConfig

            console.print(
                "[cyan]Interactive mode - creating default simulation config[/cyan]"
            )

            grid_x = int(Prompt.ask("Grid size X", default="100"))
            grid_y = int(Prompt.ask("Grid size Y", default="100"))
            tau = float(Prompt.ask("Relaxation time (tau)", default="0.6"))
            nt = int(Prompt.ask("Number of timesteps", default="1000"))
            save_interval = int(Prompt.ask("Save interval", default="100"))

            config = SimulationConfig(
                grid_shape=(grid_x, grid_y),
                tau=tau,
                nt=nt,
                save_interval=save_interval,
            )

        # Display configuration summary
        _display_config_summary(config)

        if dry_run:
            console.print("[yellow]Dry run mode - simulation not started[/yellow]")
            return

        # Confirm before running
        if not no_prompt:
            if not Confirm.ask("[bold]Start simulation?[/bold]", default=True):
                console.print("[yellow]Simulation cancelled.[/yellow]")
                return

        # Run the simulation
        _run_simulation(config)

        console.print()
        console.print(
            Panel.fit("[bold green]Simulation complete![/bold green]", title="Success")
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
