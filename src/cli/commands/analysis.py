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
from src.config.run_config import COMPARISON_DIRNAME

# Imported for its registration side effect: both choice lists below are read
# off the ``dimensionless`` registry at import time, and click evaluates them
# when the commands are defined.
from src.simulation_io.analysis.physical_parameters import dimensionless_keys
from src.simulation_io.plotting.run_labels import LABEL_PARAM_CHOICES

if TYPE_CHECKING:
    from src.config import SimulationConfig
    from src.simulation_io.analysis.accelerations import Smoothing
    from src.simulation_io.plotting.regime_map_plot import AxisScale

_AXIS_CHOICES = dimensionless_keys()
_SCALE_CHOICES = ["linear", "log"]


@cli.command()
@click.argument("parent_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--no-prompt",
    "no_prompt",
    is_flag=True,
    help="Skip interactive operator selection; run CSV export only (no per-run analysis plots).",
)
@click.option(
    "--label-param",
    "label_params",
    multiple=True,
    type=click.Choice(LABEL_PARAM_CHOICES),
    help="Dimensionless number(s) to put in the legend labels, repeatable and rendered in the order "
    "given (default: whichever of Bo/Bo_parallel, Oh and Re differ across the runs). Use 'name' for "
    "the run name, i.e. the label used before this option existed.",
)
@cli_command(title="Comparison Analysis", interrupt_message="Comparison interrupted by user.")
def compare(parent_dir: str, no_prompt: bool, label_params: tuple[str, ...]) -> None:
    """Build CSV metrics and comparison plots for all runs in PARENT_DIR."""
    from src.cli.analysis_routing import analyse_tree
    from src.registry import get_operators

    console.print(f"[dim]Parent directory : {parent_dir}[/dim]")
    if label_params:
        console.print(f"[dim]Legend labels    : {', '.join(label_params)}[/dim]")

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

    n_runs, n_ok = analyse_tree(parent_dir, fields=fields, label_keys=list(label_params) or None)
    if n_runs == 0:
        console.print("[yellow]No simulation run directories found.[/yellow]")
        return
    if n_ok == 0:
        console.print("[yellow]No runs produced CSV data. Check sim_type and snapshot files.[/yellow]")
        return

    out_dir = Path(parent_dir) / COMPARISON_DIRNAME
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
    help="Output directory for the figure (default: <dirs_txt parent>/regime_map_analysis).",
)
@click.option(
    "--smoothing",
    "smoothing",
    type=click.Choice(["raw", "savgol"]),
    default="raw",
    help="Acceleration-curve smoothing for peak detection: 'raw' (default, unsmoothed) or "
    "'savgol' (Savitzky-Golay filtered, reduces spikiness).",
)
@click.option(
    "--x",
    "x_key",
    type=click.Choice(_AXIS_CHOICES),
    default=None,
    help="Dimensionless number on the x axis (default: bo_parallel).",
)
@click.option(
    "--y",
    "y_key",
    type=click.Choice(_AXIS_CHOICES),
    default=None,
    help="Dimensionless number on the y axis (default: oh).",
)
@click.option(
    "--xscale",
    "xscale",
    type=click.Choice(_SCALE_CHOICES),
    default="linear",
    help="Scale of the x axis. Use 'log' for a quantity spanning decades, e.g. La.",
)
@click.option(
    "--yscale",
    "yscale",
    type=click.Choice(_SCALE_CHOICES),
    default="linear",
    help="Scale of the y axis.",
)
@cli_command(title="Regime Map", interrupt_message="Regime map analysis interrupted by user.")
def regime_map(
    dirs_txt: str,
    allowed_roots: tuple[str, ...],
    out_dir: str | None,
    smoothing: str,
    x_key: str | None,
    y_key: str | None,
    xscale: str,
    yscale: str,
) -> None:
    """Classify runs listed in DIRS_TXT and plot any pair of dimensionless numbers.

    Regimes are pinning / dissipative / capillary / steady / unknown.
    """
    # Imported lazily so the CLI does not pull in the plotting stack to define
    # its options, and so tests can patch ``build_regime_map`` at this seam. The
    # axis defaults come from the same module for the same reason.
    from src.simulation_io.plotting.regime_map_plot import DEFAULT_X_KEY
    from src.simulation_io.plotting.regime_map_plot import DEFAULT_Y_KEY
    from src.simulation_io.plotting.regime_map_plot import build_regime_map

    x_key = x_key or DEFAULT_X_KEY
    y_key = y_key or DEFAULT_Y_KEY
    console.print(f"[dim]Run-dir list : {dirs_txt}[/dim]")
    console.print(f"[dim]Axes         : {x_key} ({xscale}) vs {y_key} ({yscale})[/dim]")
    console.print()

    out_path = build_regime_map(
        dirs_txt,
        allowed_roots,
        out_dir=out_dir,
        smoothing=cast("Smoothing", smoothing),
        x_key=x_key,
        y_key=y_key,
        xscale=cast("AxisScale", xscale),
        yscale=cast("AxisScale", yscale),
    )
    if out_path is None:
        console.print("[yellow]No runs produced a usable classification for these axes.[/yellow]")
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


def _analyse_length_scale(config: SimulationConfig, config_toml: str, out_dir: Path) -> None:
    """Plot the region behind the Bo/Oh length scale and refresh the parameter overview."""
    from src.simulation_io.analysis.physical_parameters import write_length_scale_figure
    from src.simulation_io.analysis.physical_parameters import write_physical_parameters

    # A run directory holds its own config.toml, so pointing the command at one
    # finds that run's snapshots; --out-dir only redirects where output lands.
    run_dir = Path(config_toml).resolve().parent

    figure_path = write_length_scale_figure(config, out_dir, run_dir=run_dir)
    if figure_path is None:
        console.print("[yellow]No droplet region could be resolved; no figure written.[/yellow]")
    else:
        success("Length-scale diagnostic written")
        console.print(f"[bold green]Plot saved to:[/bold green] {figure_path}")

    # physical_parameters.txt is otherwise only written at run start, so a
    # finished run would keep the Bo computed before this analysis existed.
    overview_path = out_dir / "physical_parameters.txt"
    write_physical_parameters(config, overview_path)
    console.print(f"[bold green]Overview refreshed:[/bold green] {overview_path}")


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
    "--length-scale",
    "length_scale",
    is_flag=True,
    help="Plot the droplet region behind the Bo/Oh length scale against the region measured from the "
    "run's snapshots, and refresh physical_parameters.txt with the measured densities.",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory for analysis outputs (default: the config file's directory).",
)
@cli_command(title="Analysis", interrupt_message="Analysis interrupted by user.")
def analyse(config_toml: str, surface_tension: bool, length_scale: bool, out_dir: str | None) -> None:
    """Run standalone analyses for the configuration in CONFIG_TOML.

    Unlike the automatic calibration during `tud-lbm run` (which only
    triggers for EOS without a closed-form sigma), --surface-tension forces
    the Young-Laplace measurement for any supported multiphase EOS.

    Examples:
        # Measure surface tension for the configured EOS
        tud-lbm analyse config.toml --surface-tension

        # Check the droplet region the Bond number is built from
        tud-lbm analyse run_dir/config.toml --length-scale

        # Write outputs somewhere other than the config's directory
        tud-lbm analyse config.toml --surface-tension --out-dir results/
    """
    if not (surface_tension or length_scale):
        msg = "select at least one analysis: --surface-tension or --length-scale"
        raise click.UsageError(msg)

    target_dir = Path(out_dir) if out_dir is not None else Path(config_toml).resolve().parent
    config = _load_single_config(config_toml)
    console.print(f"[dim]Output directory : {target_dir}[/dim]")
    console.print()
    if surface_tension:
        _analyse_surface_tension(config, target_dir)
    if length_scale:
        _analyse_length_scale(config, config_toml, target_dir)
