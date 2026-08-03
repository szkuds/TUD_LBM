"""The ``tud-lbm compare``, ``regime-map`` and ``analyse`` commands."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast
import click
from src.cli._console import cli_command
from src.cli._console import console
from src.cli._console import success
from src.cli.app import cli
from src.cli.config_loading import _load_single_config
from src.cli.field_select import prompt_fields_marked

if TYPE_CHECKING:
    from src.config import SimulationConfig
    from src.simulation_io.analysis.accelerations import Smoothing


@cli.command()
@click.argument("parent_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--no-prompt",
    "no_prompt",
    is_flag=True,
    help="Skip interactive operator selection; run CSV export only (no per-run analysis plots).",
)
@cli_command(title="Comparison Analysis", interrupt_message="Comparison interrupted by user.")
def compare(parent_dir: str, no_prompt: bool) -> None:
    """Build CSV metrics and comparison plots for all runs in PARENT_DIR."""
    from src.cli.analysis_routing import analyse_tree
    from src.registry import get_operators
    from src.simulation_io.plotting.run_comparison import _COMPARISON_DIR

    console.print(f"[dim]Parent directory : {parent_dir}[/dim]")

    if no_prompt:
        fields: list[str] | None = None
    else:
        comparison_ops = get_operators("analysis")
        fields = prompt_fields_marked(
            comparison_ops,
            None,
            label="per-run comparison operators",
            config_label="the run config",
        )

    if fields:
        console.print(f"[dim]Operators     : {', '.join(fields)}[/dim]")
    console.print()

    n_runs, n_ok = analyse_tree(parent_dir, fields=fields)
    if n_runs == 0:
        console.print("[yellow]No simulation run directories found.[/yellow]")
        return
    if n_ok == 0:
        console.print("[yellow]No runs produced CSV data. Check sim_type and snapshot files.[/yellow]")
        return

    out_dir = Path(parent_dir) / _COMPARISON_DIR
    success(f"Comparison analysis complete!  {n_ok}/{n_runs} run(s) processed")
    console.print(f"[bold green]Plots saved to:[/bold green] {out_dir}")


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
@cli_command(title="Regime Map", interrupt_message="Regime map analysis interrupted by user.")
def regime_map(dirs_txt: str, allowed_roots: tuple[str, ...], out_dir: str | None, smoothing: str) -> None:
    """Classify runs listed in DIRS_TXT into pinning/viscous/inertial/unknown and plot Bo_parallel vs Oh."""
    from src.simulation_io.plotting.regime_map_plot import build_regime_map

    console.print(f"[dim]Run-dir list : {dirs_txt}[/dim]")
    console.print()

    out_path = build_regime_map(dirs_txt, allowed_roots, out_dir=out_dir, smoothing=cast("Smoothing", smoothing))
    if out_path is None:
        console.print("[yellow]No runs produced a usable classification.[/yellow]")
        sys.exit(1)

    success("Regime map complete!")
    console.print(f"[bold green]Plot saved to:[/bold green] {out_path}")


def _analyse_surface_tension(config: SimulationConfig, out_dir: Path) -> None:
    """Run the Young-Laplace calibration for *config* and report sigma."""
    if not config.is_multiphase:
        msg = f"surface tension requires a multiphase configuration; got sim_type='{config.sim_type}'"
        raise ValueError(msg)

    from src.config.jax_config import configure_jax

    configure_jax()

    from src.simulation_io.analysis.surface_tension import calibrate_surface_tension
    from src.simulation_io.analysis.surface_tension import surface_tension_dir

    sigma = calibrate_surface_tension(config, out_dir)

    success(f"Surface tension: σ = {sigma:.6g}")
    console.print(f"[bold green]Calibration outputs saved to:[/bold green] {surface_tension_dir(out_dir)}")


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
@cli_command(title="Analysis", interrupt_message="Surface tension analysis interrupted by user.")
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

    target_dir = Path(out_dir) if out_dir is not None else Path(config_toml).resolve().parent
    config = _load_single_config(config_toml)
    console.print(f"[dim]Output directory : {target_dir}[/dim]")
    console.print()
    _analyse_surface_tension(config, target_dir)
