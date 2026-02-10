"""Command-line interface for TUD-LBM simulations.

Usage:
    tud-lbm config.toml
    tud-lbm config.toml --no-prompt
"""

import os
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.prompt import Prompt
from rich.table import Table

from config import (
    BASE_RESULTS_DIR,
    DEFAULT_SAVE_FIELDS,
    AVAILABLE_FIELDS
)

console = Console()



def _prompt_missing(config: dict) -> dict:
    """Interactively prompt for missing optional configuration values.

    Args:
        config: The loaded configuration dictionary

    Returns:
        Updated configuration dictionary with prompted values
    """
    console.print()
    console.print(Panel.fit(
        "[bold cyan]TUD-LBM Configuration[/bold cyan]",
        subtitle="Interactive Setup"
    ))

    # Prompt for results_dir if not set
    if "results_dir" not in config:
        results_dir = Prompt.ask(
            "[yellow]Output directory for results[/yellow]",
            default=BASE_RESULTS_DIR
        )
        config["results_dir"] = os.path.expanduser(results_dir)

    # Prompt for save_fields if not set
    if "save_fields" not in config:
        console.print()
        console.print("[yellow]Which data fields should be saved?[/yellow]")
        console.print(f"  Available: {', '.join(AVAILABLE_FIELDS)}")

        fields_input = Prompt.ask(
            "  Enter comma-separated list",
            default=",".join(DEFAULT_SAVE_FIELDS)
        )
        config["save_fields"] = [f.strip() for f in fields_input.split(",") if f.strip()]

    # Prompt for open_paraview if desired (post-run action)
    if "open_paraview" not in config:
        config["open_paraview"] = Confirm.ask(
            "[yellow]Open results in ParaView after simulation?[/yellow]",
            default=False
        )

    return config


def _display_config_summary(config: dict) -> None:
    """Display a summary of the simulation configuration.

    Args:
        config: The configuration dictionary
    """
    console.print()

    table = Table(title="Simulation Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    # Key parameters to display
    display_keys = [
        ("simulation_type", "Simulation Type"),
        ("grid_shape", "Grid Shape"),
        ("lattice_type", "Lattice Type"),
        ("tau", "Relaxation Time (τ)"),
        ("nt", "Time Steps"),
        ("save_interval", "Save Interval"),
        ("results_dir", "Results Directory"),
        ("save_fields", "Save Fields"),
    ]

    for key, label in display_keys:
        if key in config:
            value = config[key]
            if isinstance(value, (list, tuple)):
                value = ", ".join(str(v) for v in value)
            table.add_row(label, str(value))

    # Multiphase-specific parameters
    if config.get("simulation_type") == "multiphase":
        mp_keys = [
            ("kappa", "Kappa"),
            ("rho_l", "Liquid Density"),
            ("rho_v", "Vapor Density"),
            ("interface_width", "Interface Width"),
        ]
        for key, label in mp_keys:
            if key in config:
                table.add_row(label, str(config[key]))

    # Force configuration
    if config.get("force_enabled"):
        force_obj = config.get("force_obj", [])
        if force_obj:
            force_names = [type(f).__name__ for f in force_obj]
            table.add_row("Forces", ", ".join(force_names))

    console.print(table)
    console.print()


def _run_simulation(config: dict) -> object:
    """Run the simulation with the given configuration.

    Args:
        config: The configuration dictionary

    Returns:
        The completed simulation instance
    """
    # Import here to avoid circular imports and allow JAX config
    from config import configure_jax
    configure_jax()

    from core import Run

    # Extract special options that aren't passed to Run
    open_paraview = config.pop("open_paraview", False)

    console.print("[bold green]Starting simulation...[/bold green]")
    console.print()

    sim = Run(**config)
    sim.run(verbose=True)

    # Post-run actions
    if open_paraview:
        _open_paraview(sim.io_handler.run_dir)

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
@click.argument("config_path", type=click.Path(exists=True))
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
@click.version_option(package_name="tud_lbm")
def main(config_path: str, no_prompt: bool, dry_run: bool) -> None:
    """Run a TUD-LBM simulation from a configuration file.

    CONFIG_PATH is the path to a TOML or JSON configuration file.

    Example usage:
        tud-lbm config.toml
        tud-lbm config.toml --no-prompt
        tud-lbm config.toml --dry-run
    """
    from config import load

    console.print()
    console.print(Panel.fit(
        "[bold blue]TUD-LBM[/bold blue] - Lattice Boltzmann Method Solver",
        subtitle="Delft University of Technology"
    ))
    console.print()

    try:
        # Load configuration
        console.print(f"[cyan]Loading configuration from: {config_path}[/cyan]")
        config = load(config_path)

        # Interactive prompts if enabled
        if not no_prompt:
            config = _prompt_missing(config)
        else:
            # Apply defaults for missing optional values
            if "results_dir" not in config:
                config["results_dir"] = os.path.expanduser(BASE_RESULTS_DIR)
            if "save_fields" not in config:
                config["save_fields"] = DEFAULT_SAVE_FIELDS

        # Display configuration summary
        _display_config_summary(config)

        if dry_run:
            console.print("[yellow]Dry run mode - simulation not started[/yellow]")
            return

        # Confirm before running (if interactive)
        if not no_prompt:
            if not Confirm.ask("[bold]Start simulation?[/bold]", default=True):
                console.print("[yellow]Simulation cancelled.[/yellow]")
                return

        # Run the simulation
        sim = _run_simulation(config)

        console.print()
        console.print(Panel.fit(
            f"[bold green]Simulation complete![/bold green]\n"
            f"Results saved to: {sim.io_handler.run_dir}",
            title="Success"
        ))

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        sys.exit(1)
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
