"""The ``tud-lbm compare`` and ``tud-lbm analyse`` groups.

``compare runs``       overlay the ``simulation_data.csv`` series of several runs;
``compare regime-map`` classify several runs and plot any pair of dimensionless numbers;
``analyse ...``        standalone physics diagnostics of one configuration.

Both ``compare`` subcommands name their runs the same way: a directory, whose run
directories are found at any depth, or a ``.txt`` file listing run directories.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import click
from src.cli._console import cli_command
from src.cli._console import console
from src.cli._console import success
from src.cli.app import cli
from src.cli.config_loading import _load_single_config
from src.cli.config_loading import _validate_run_dir_has_config
from src.config.analysis_options import AxisScale
from src.config.analysis_options import Smoothing
from src.config.multiphase import is_multiphase
from src.config.run_config import COMPARISON_DIRNAME
from src.config.run_config import PHYSICAL_PARAMETERS_FILENAME
from src.config.run_config import REGIME_MAP_DIRNAME

# Imported for its registration side effect: both choice lists below are read
# off the ``dimensionless`` registry at import time, and click evaluates them
# when the commands are defined.
from src.simulation_io.analysis.physical_parameters import dimensionless_keys
from src.simulation_io.analysis.run_labels import LABEL_PARAM_CHOICES

if TYPE_CHECKING:
    from src.cli.analysis_routing import RunSet
    from src.config import SimulationConfig

_AXIS_CHOICES = dimensionless_keys()
_SMOOTHING_CHOICES = [choice.value for choice in Smoothing]
_SCALE_CHOICES = [choice.value for choice in AxisScale]

_run_set_argument = click.argument("target", metavar="TARGET", type=click.Path(exists=True))
_allowed_root_option = click.option(
    "--allowed-root",
    "allowed_roots",
    multiple=True,
    type=click.Path(exists=True, file_okay=False),
    help="With a .txt list: an additional directory that a listed run directory may resolve within, beyond "
    "the default results root (repeatable, e.g. one per HPC mount).",
)


_comparison_out_dir_option = click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help=f"Output directory (default: TARGET/{COMPARISON_DIRNAME}, or beside a .txt list).",
)
_regime_map_out_dir_option = click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help=f"Output directory (default: TARGET/{REGIME_MAP_DIRNAME}, or beside a .txt list).",
)


def _resolve_runs(target: str, allowed_roots: tuple[str, ...]) -> RunSet:
    """Resolve TARGET into its runs, printing what was found."""
    from src.cli.analysis_routing import resolve_run_set

    run_set = resolve_run_set(target, allowed_roots)
    console.print(f"[dim]Runs          : {len(run_set.run_dirs)} from {target}[/dim]")
    return run_set


@cli.group()
def compare() -> None:
    """Build figures over several runs, given as a directory or a .txt list of run directories."""


@compare.command("runs")
@_run_set_argument
@_allowed_root_option
@_comparison_out_dir_option
@click.option(
    "--label-param",
    "label_params",
    multiple=True,
    type=click.Choice(LABEL_PARAM_CHOICES),
    help="Dimensionless number(s) to put in the legend labels, repeatable and rendered in the order "
    "given (default: whichever of Bo/Bo_parallel, Oh and Re differ across the runs). Use 'name' for "
    "the run name.",
)
@cli_command(title="Comparison", interrupt_message="Comparison interrupted by user.")
def compare_runs_command(
    target: str,
    allowed_roots: tuple[str, ...],
    out_dir: str | None,
    label_params: tuple[str, ...],
) -> None:
    """Export each run's simulation_data.csv and overlay the runs of TARGET.

    Per-run figures are `tud-lbm visualise analysis RUN_DIR`.
    """
    from src.cli.analysis_routing import analyse_runs

    run_set = _resolve_runs(target, allowed_roots)
    if label_params:
        console.print(f"[dim]Legend labels : {', '.join(label_params)}[/dim]")
    console.print()

    if not run_set.run_dirs:
        console.print("[yellow]No simulation run directories found.[/yellow]")
        return
    destination = Path(out_dir) if out_dir is not None else run_set.out_root / COMPARISON_DIRNAME
    n_ok = analyse_runs(run_set, destination, label_keys=list(label_params) or None)
    if n_ok == 0:
        console.print("[yellow]No runs produced CSV data. Check the [wetting] section and snapshot files.[/yellow]")
        return

    success(f"Comparison complete!  {n_ok}/{len(run_set.run_dirs)} run(s) processed")
    console.print(f"[bold green]Plots saved to:[/bold green] {destination}")


@compare.command("regime-map")
@_run_set_argument
@_allowed_root_option
@_regime_map_out_dir_option
@click.option(
    "--smoothing",
    "smoothing",
    type=click.Choice(_SMOOTHING_CHOICES),
    default=Smoothing.RAW.value,
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
    default=AxisScale.LINEAR.value,
    help="Scale of the x axis. Use 'log' for a quantity spanning decades, e.g. La.",
)
@click.option(
    "--yscale",
    "yscale",
    type=click.Choice(_SCALE_CHOICES),
    default=AxisScale.LINEAR.value,
    help="Scale of the y axis.",
)
@cli_command(title="Regime Map", interrupt_message="Regime map analysis interrupted by user.")
def regime_map(
    target: str,
    allowed_roots: tuple[str, ...],
    out_dir: str | None,
    smoothing: str,
    x_key: str | None,
    y_key: str | None,
    xscale: str,
    yscale: str,
) -> None:
    """Classify the runs of TARGET and plot any pair of dimensionless numbers.

    Regimes are pinning / dissipative / capillary / steady / unknown.
    """
    # Imported lazily so the CLI does not pull in the plotting stack to define
    # its options, and so tests can patch ``build_regime_map`` at this seam. The
    # axis defaults come from the same module for the same reason.
    from src.simulation_io.plotting.regime_map_plot import DEFAULT_X_KEY
    from src.simulation_io.plotting.regime_map_plot import DEFAULT_Y_KEY
    from src.simulation_io.plotting.regime_map_plot import build_regime_map

    run_set = _resolve_runs(target, allowed_roots)
    x_key = x_key or DEFAULT_X_KEY
    y_key = y_key or DEFAULT_Y_KEY
    console.print(f"[dim]Axes          : {x_key} ({xscale}) vs {y_key} ({yscale})[/dim]")
    console.print()

    out_path = build_regime_map(
        run_set.run_dirs,
        Path(out_dir) if out_dir is not None else run_set.out_root / REGIME_MAP_DIRNAME,
        smoothing=Smoothing(smoothing),
        x_key=x_key,
        y_key=y_key,
        xscale=AxisScale(xscale),
        yscale=AxisScale(yscale),
    )
    if out_path is None:
        console.print("[yellow]No runs produced a usable classification for these axes.[/yellow]")
        sys.exit(1)

    success("Regime map complete!")
    console.print(f"[bold green]Plot saved to:[/bold green] {out_path}")


def _analyse_surface_tension(config: SimulationConfig, out_dir: Path) -> None:
    """Write the Young-Laplace calibration artefacts for *config* and report sigma."""
    if not is_multiphase(config):
        msg = "surface tension requires a multiphase configuration; got no [multiphase] section"
        raise ValueError(msg)

    from src.config.jax_config import configure_jax

    configure_jax()

    from src.simulation_io.analysis.surface_tension import calibrate_surface_tension
    from src.simulation_io.analysis.surface_tension import surface_tension_dir

    sigma = calibrate_surface_tension(config, out_dir)
    if sigma is None:
        return

    success(f"Surface tension: σ = {sigma:.6g}")
    console.print(f"[bold green]Calibration outputs saved to:[/bold green] {surface_tension_dir(out_dir)}")


def _analyse_coexistence(config: SimulationConfig, out_dir: Path) -> None:
    """Write the Maxwell-construction artefacts for *config* and report the coexistence pair."""
    if not is_multiphase(config):
        msg = "coexistence requires a multiphase configuration; got no [multiphase] section"
        raise ValueError(msg)

    from src.config.jax_config import configure_jax

    configure_jax()

    from src.simulation_io.analysis.coexistence import calibrate_coexistence
    from src.simulation_io.analysis.coexistence import coexistence_dir

    pair = calibrate_coexistence(config, out_dir)
    if pair is None:
        eos = config.multiphase["eos"] if config.multiphase else None
        console.print(
            f"[yellow]EOS {eos!r} has no coexistence solver: its rho_l/rho_v are its own parameters.[/yellow]"
        )
        return

    rho_v, rho_l = pair
    success(f"Maxwell coexistence: rho_v = {rho_v:.6g}, rho_l = {rho_l:.6g}")
    console.print(f"[bold green]Coexistence outputs saved to:[/bold green] {coexistence_dir(out_dir)}")


def _analyse_length_scale(config: SimulationConfig, run_dir: Path, out_dir: Path) -> None:
    """Plot the inclusion area behind the Bo/Oh length scale and refresh the parameter overview."""
    from src.simulation_io.analysis.physical_parameters import write_length_scale_figure
    from src.simulation_io.analysis.physical_parameters import write_physical_parameters

    figure_path = write_length_scale_figure(config, out_dir, run_dir=run_dir)
    if figure_path is None:
        console.print("[yellow]No inclusion area could be resolved; no figure written.[/yellow]")
    else:
        success("Length-scale diagnostic written")
        console.print(f"[bold green]Plot saved to:[/bold green] {figure_path}")

    # physical_parameters.txt is otherwise only written at run start, so a
    # finished run would keep the Bo computed before this analysis existed.
    overview_path = out_dir / PHYSICAL_PARAMETERS_FILENAME
    write_physical_parameters(config, overview_path)
    console.print(f"[bold green]Overview refreshed:[/bold green] {overview_path}")


_config_argument = click.argument("config_toml", type=click.Path(exists=True, dir_okay=False))
_analyse_out_dir_option = click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory for analysis outputs (default: the directory holding the config).",
)


def _analysis_target(config_toml: str | Path, out_dir: str | None) -> tuple[SimulationConfig, Path]:
    """Load *config_toml* and resolve the output directory, printing it."""
    target_dir = Path(out_dir) if out_dir is not None else Path(config_toml).resolve().parent
    config = _load_single_config(str(config_toml))
    console.print(f"[dim]Output directory : {target_dir}[/dim]")
    console.print()
    return config, target_dir


@cli.group()
def analyse() -> None:
    """Run standalone physics diagnostics of one configuration or run."""


@analyse.command("surface-tension")
@_config_argument
@_analyse_out_dir_option
@cli_command(title="Analysis - surface tension", interrupt_message="Analysis interrupted by user.")
def analyse_surface_tension(config_toml: str, out_dir: str | None) -> None:
    """Report the measured lattice surface tension of the fluid in CONFIG_TOML.

    Writes the Young-Laplace calibration figures. This reads the calibration
    cache for any multiphase EOS, including one with a closed-form sigma that was
    measured for verification. A fluid not yet measured prints how to stage the
    droplet sweep (`tud-lbm calibration stage`).
    """
    config, target_dir = _analysis_target(config_toml, out_dir)
    _analyse_surface_tension(config, target_dir)


@analyse.command("coexistence")
@_config_argument
@_analyse_out_dir_option
@cli_command(title="Analysis - coexistence", interrupt_message="Analysis interrupted by user.")
def analyse_coexistence(config_toml: str, out_dir: str | None) -> None:
    """Compare the Maxwell coexistence densities of the EOS in CONFIG_TOML with the configured ones.

    Writes the equal-area figure. An EOS without a coexistence solver prints
    that its densities are parameters.
    """
    config, target_dir = _analysis_target(config_toml, out_dir)
    _analyse_coexistence(config, target_dir)


@analyse.command("length-scale")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@_analyse_out_dir_option
@cli_command(title="Analysis - length scale", interrupt_message="Analysis interrupted by user.")
def analyse_length_scale(run_dir: str, out_dir: str | None) -> None:
    """Plot the inclusion area behind the Bo/Oh length scale of RUN_DIR.

    Compares it with the area measured from the run's snapshots, and refreshes
    physical_parameters.txt with the measured densities.
    """
    config, target_dir = _analysis_target(_validate_run_dir_has_config(run_dir), out_dir)
    _analyse_length_scale(config, Path(run_dir).resolve(), target_dir)
