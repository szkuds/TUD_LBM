"""Command-line interface for TUD-LBM simulations.

Example Python usage::

    from app_setup import SimulationSetup
    from runner import Run

    setup = SimulationSetup(
        grid_shape=(100, 100),
        tau=0.6,
        nt=10000,
        save_interval=1000,
    )
    sim = Run(setup)
    sim.run()
"""

import os
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.prompt import Prompt
from rich.table import Table

from app_setup import SimulationSetup

console = Console()


def _prompt_missing(setup: SimulationSetup) -> SimulationSetup:
    """Interactively prompt for missing optional configuration values.

    Args:
        setup: The SimulationSetup object

    Returns:
        Updated SimulationSetup (note: dataclass is mutable)
    """
    console.print()
    console.print(Panel.fit(
        "[bold cyan]TUD-LBM Configuration[/bold cyan]",
        subtitle="Interactive Setup"
    ))

    # For now, SimulationSetup has defaults, so prompts are optional
    # This can be expanded when adapter support is added

    return setup


def _display_config_summary(setup: SimulationSetup) -> None:
    """Display a summary of the simulation configuration.

    Args:
        setup: The SimulationSetup object
    """
    console.print()

    table = Table(title="Simulation Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Simulation Type", setup.sim_type)
    table.add_row("Grid Shape", str(setup.grid_shape))
    table.add_row("Lattice Type", setup.lattice_type)
    table.add_row("Relaxation Time (τ)", str(setup.tau))
    table.add_row("Time Steps", str(setup.nt))
    table.add_row("Save Interval", str(setup.save_interval))
    table.add_row("Results Directory", setup.results_dir)
    if setup.save_fields:
        table.add_row("Save Fields", ", ".join(setup.save_fields))

    # Multiphase-specific parameters
    if setup.is_multiphase:
        table.add_row("Kappa", str(setup.kappa))
        table.add_row("Liquid Density", str(setup.rho_l))
        table.add_row("Vapor Density", str(setup.rho_v))
        table.add_row("Interface Width", str(setup.interface_width))

    # Force configuration
    if setup.force_enabled and setup.force_obj:
        force_list = setup.force_obj if isinstance(setup.force_obj, list) else [setup.force_obj]
        force_names = [type(f).__name__ for f in force_list]
        table.add_row("Forces", ", ".join(force_names))

    console.print(table)
    console.print()


def _run_simulation(setup: SimulationSetup) -> object:
    """Run the simulation with the given configuration.

    Args:
        setup: The SimulationSetup object

    Returns:
        The completed simulation instance
    """
    # Import here to avoid circular imports and allow JAX config
    from app_setup import configure_jax
    configure_jax()

    from runner import Run

    console.print("[bold green]Starting simulation...[/bold green]")
    console.print()

    sim = Run(setup)
    sim.run(verbose=True)

    return sim


def _open_paraview(results_dir: str) -> None:
    """Attempt to open ParaView with the results directory.

    Args:
        results_dir: Path to the simulation results directory
    """
    import subprocess
    import shutil

    paraview_path = shutil.which("paraview")
    if paraview_path:
        try:
            console.print(f"[cyan]Opening ParaView with results from: {results_dir}[/cyan]")
            subprocess.Popen([paraview_path, results_dir])
        except Exception as e:
            console.print(f"[yellow]Could not open ParaView: {e}[/yellow]")
    else:
        console.print("[yellow]ParaView not found in PATH. Skipping.[/yellow]")


@click.command()
@click.argument("config_path", type=click.Path(exists=True), required=False)
@click.option(
    "--no-prompt",
    is_flag=True,
    help="Skip interactive prompts and use defaults for missing values"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Parse config and display summary without running simulation"
)
@click.option(
    "--list-simulation_operators",
    is_flag=True,
    help="List all registered simulation_operators and exit"
)
@click.version_option(package_name="tud_lbm")
def main(config_path: str, no_prompt: bool, dry_run: bool, list_simulation_operators: bool) -> None:
    """Run a TUD-LBM simulation.

    CONFIG_PATH is an optional path to a configuration file (.toml).
    If omitted, an interactive prompt collects parameters.

    Examples:

        tud_lbm example/config_simple.toml
        tud_lbm example/config_complex.toml --dry-run
        tud_lbm --list-simulation_operators
        tud_lbm                              # interactive mode
    """
    console.print()
    console.print(Panel.fit(
        "[bold blue]TUD-LBM[/bold blue] - Lattice Boltzmann Method Solver",
        subtitle="Delft University of Technology"
    ))
    console.print()

    # Ensure all simulation_operators are imported so registrations are triggered
    import simulation_operators  # noqa: F401

    if list_simulation_operators:
        from app_setup.registry import OPERATOR_REGISTRY
        table = Table(title="Registered Operators", show_header=True, header_style="bold magenta")
        table.add_column("Kind", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Class", style="yellow")

        for key in sorted(OPERATOR_REGISTRY):
            entry = OPERATOR_REGISTRY[key]
            table.add_row(entry.kind, entry.name, entry.cls.__name__)

        console.print(table)
        return

    try:
        if config_path:
            # ── Load config file via adapter ─────────────────────────
            from app_setup import get_adapter

            console.print(
                f"[cyan]Loading configuration from:[/cyan] {config_path}"
            )
            adapter = get_adapter(config_path)
            setup = adapter.load(config_path)
        else:
            # ── Interactive mode ─────────────────────────────────────
            console.print(
                "[cyan]Interactive mode - creating default simulation config[/cyan]"
            )

            grid_x = int(Prompt.ask("Grid size X", default="100"))
            grid_y = int(Prompt.ask("Grid size Y", default="100"))
            tau = float(Prompt.ask("Relaxation time (tau)", default="0.6"))
            nt = int(Prompt.ask("Number of timesteps", default="1000"))
            save_interval = int(Prompt.ask("Save interval", default="100"))

            setup = SimulationSetup(
                grid_shape=(grid_x, grid_y),
                tau=tau,
                nt=nt,
                save_interval=save_interval,
            )

        # Display configuration summary
        _display_config_summary(setup)

        if dry_run:
            console.print("[yellow]Dry run mode - simulation not started[/yellow]")
            return

        # Confirm before running
        if not no_prompt:
            if not Confirm.ask("[bold]Start simulation?[/bold]", default=True):
                console.print("[yellow]Simulation cancelled.[/yellow]")
                return

        # Run the simulation
        sim = _run_simulation(setup)

        console.print()
        console.print(Panel.fit(
            f"[bold green]Simulation complete![/bold green]\n"
            f"Results saved to: {sim.io_handler.run_dir}",
            title="Success"
        ))

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
