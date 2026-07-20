"""The ``tud-lbm run`` command."""

from __future__ import annotations
from typing import cast
import click
from tud_lbm.cli._console import cli_command
from tud_lbm.cli._console import success
from tud_lbm.cli.app import cli
from tud_lbm.cli.execution import RunFlags
from tud_lbm.cli.execution import _run_impl


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
@cli_command(title="Lattice Boltzmann Method Solver", interrupt_message="Simulation interrupted by user.")
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

    if _run_impl(config_path, overrides, max_workers, init_dir, flags):
        success("Simulation, saving and plotting complete!")
