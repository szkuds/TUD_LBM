"""The ``tud-lbm calibration`` commands: the Young-Laplace sweep as ordinary runs.

``stage`` writes the sweep's run configs, ``collect`` fits the finished runs
into the surface-tension cache, and ``refresh`` rewrites the parameter overview
of runs whose fluid has since been calibrated. The runs in between go through
``tud-lbm run`` like any other config — in practice on DelftBlue, through
``scripts/db_pipeline.sh`` and ``scripts/db_download.sh``.

Everything lands under ``$TUD_LBM_DATA_DIR/surface_tension/``, one folder per
fluid: its staged configs, its sweep runs and its fitted result.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import click
from src.cli._console import cli_command
from src.cli._console import console
from src.cli._console import success
from src.cli.app import cli
from src.cli.config_loading import _expand_raw_config
from src.cli.config_loading import _load_raw_config
from src.config.run_config import CONFIG_FILENAME

if TYPE_CHECKING:
    from src.config import SimulationConfig

#: Run directories sit at ``<results_dir>/<date>/<time>_<name>``, so a results
#: root reaches them two levels down.
_RUN_DIR_GLOBS = ("*", "*/*")


@cli.group()
def calibration() -> None:
    """Stage, collect and apply surface-tension calibrations."""


def _stageable(config: SimulationConfig, *, closed_form: bool) -> bool:
    """Whether *config*'s fluid still needs a sweep staged for it."""
    from src.registry import get_operator_names
    from src.simulation_io.analysis.surface_tension import is_calibrated
    from src.simulation_io.analysis.surface_tension import is_sweep_config
    from src.simulation_io.analysis.surface_tension import needs_calibration

    if not config.is_multiphase or config.eos not in get_operator_names("pressure"):
        return False
    # A sweep run is the measurement of its fluid, not a run that needs one:
    # staging for it would duplicate the very sweep it belongs to.
    if is_sweep_config(config):
        return False
    if not (closed_form or needs_calibration(config)):
        return False
    return not is_calibrated(config)


@calibration.command()
@click.argument("config_tomls", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--manifest",
    "manifest",
    type=click.Path(dir_okay=False),
    default=None,
    help="Also write the staged config paths to this file, one per line (read by scripts/db_pipeline.sh).",
)
@click.option(
    "--closed-form",
    "closed_form",
    is_flag=True,
    help="Also stage a fluid whose EOS has a closed-form sigma, to verify that form numerically.",
)
@cli_command(title="Calibration: stage", interrupt_message="Staging interrupted by user.")
def stage(config_tomls: tuple[str, ...], manifest: str | None, closed_form: bool) -> None:
    """Write the droplet-sweep run configs for the fluids of CONFIG_TOMLS.

    One config per sweep radius is written for every distinct fluid that has no
    cached sigma yet, into that fluid's folder under the surface_tension results
    directory: <fluid>/configs/R<radius>.toml. Configs sharing a fluid share one
    sweep, a parameter sweep contributes every fluid it expands to, and a config
    that is itself a sweep run stages nothing. The written files are ordinary
    run configs.

    Examples:
        # Stage the sweep for one config
        tud-lbm calibration stage config.toml

        # Stage every uncalibrated fluid of a batch
        tud-lbm calibration stage batch/*.toml
    """
    from src.config.adapter_toml import TomlAdapter
    from src.simulation_io.analysis.surface_tension import calibration_configs
    from src.simulation_io.analysis.surface_tension import sweep_config_path

    adapter = TomlAdapter()
    staged: dict[Path, SimulationConfig] = {}
    for config_toml in config_tomls:
        configs, *_ = _expand_raw_config(_load_raw_config(config_toml, ()))
        for config in configs:
            if not _stageable(config, closed_form=closed_form):
                continue
            for sweep_config in calibration_configs(config):
                path = sweep_config_path(sweep_config)
                # The folder name is a readable label, not the fluid's identity.
                if staged.setdefault(path, sweep_config) != sweep_config:
                    msg = (
                        f"two different fluids share the folder {path.parent.parent.name!r} "
                        "(same EOS, kappa and densities, other parameters differ); stage them one at a time"
                    )
                    raise ValueError(msg)

    for path, sweep_config in staged.items():
        adapter.save(sweep_config, str(path))
    if manifest is not None:
        Path(manifest).write_text("".join(f"{path}\n" for path in staged), encoding="utf-8")

    if not staged:
        console.print("[dim]Every fluid is already calibrated (or needs no calibration); nothing staged.[/dim]")
        return
    folders = sorted({path.parent for path in staged})
    for folder in folders:
        # Not through rich: a long path must stay on one line to be copyable.
        click.echo(str(folder))
    success(f"Staged {len(staged)} sweep config(s) for {len(folders)} fluid(s)")


@calibration.command()
@click.argument("root", required=False, type=click.Path(exists=True, file_okay=False))
@cli_command(title="Calibration: collect", interrupt_message="Collection interrupted by user.")
def collect(root: str | None) -> None:
    """Fit sigma from the finished sweep runs under ROOT.

    ROOT defaults to the surface_tension results directory, where every fluid
    has its own folder. Each fluid whose sweep is complete is fitted and stored
    in the surface-tension cache, and its fit figure and data are written into
    its folder; an incomplete sweep is reported and left for a later call, and a
    fluid already in the cache is left as it is (delete its cache entry to
    re-fit).

    Examples:
        tud-lbm calibration collect
        tud-lbm calibration collect ~/TUD_LBM_data/surface_tension
    """
    from src.config.adapter_toml import TomlAdapter
    from src.simulation_io.analysis.surface_tension import SURFACE_TENSION_ROOT
    from src.simulation_io.analysis.surface_tension import collect_calibration
    from src.simulation_io.analysis.surface_tension import find_sweep_runs
    from src.simulation_io.analysis.surface_tension import is_calibrated

    groups = find_sweep_runs(root if root is not None else SURFACE_TENSION_ROOT)
    if not groups:
        console.print("[dim]No sweep runs found.[/dim]")
        return
    adapter = TomlAdapter()
    collected = 0
    for fluid_dir, run_dirs in groups.items():
        if is_calibrated(adapter.load(str(run_dirs[-1] / CONFIG_FILENAME))):
            console.print(f"[dim]{fluid_dir.name}: already in the cache.[/dim]")
            continue
        console.print(f"[bold]{fluid_dir.name}[/bold] ({len(run_dirs)} run(s))")
        if collect_calibration(run_dirs, out_dir=fluid_dir) is not None:
            collected += 1
    if collected:
        success(f"Collected {collected} calibration(s) — commit the surface-tension cache to share them")


def _run_dirs(path: Path) -> list[Path]:
    """*path* itself when it is a run directory, else the run directories beneath it."""
    if (path / CONFIG_FILENAME).is_file():
        return [path]
    return sorted(
        candidate
        for pattern in _RUN_DIR_GLOBS
        for candidate in path.glob(pattern)
        if (candidate / CONFIG_FILENAME).is_file()
    )


@calibration.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False))
@cli_command(title="Calibration: refresh", interrupt_message="Refresh interrupted by user.")
def refresh(paths: tuple[str, ...]) -> None:
    """Rewrite the parameter overview of the runs under PATHS with the measured sigma.

    Each of PATHS is a run directory, a dated directory of runs, or a results
    directory. A run whose fluid is calibrated gets its physical_parameters.txt
    and surface_tension/ figures rewritten; its config.toml is left alone, since
    sigma is resolved from the cache, not stored in the config.

    Examples:
        tud-lbm calibration refresh ~/TUD_LBM_data/bubble/2026-09-23
    """
    from src.config.adapter_toml import TomlAdapter
    from src.simulation_io.analysis.surface_tension import is_calibrated
    from src.simulation_io.analysis.surface_tension import is_sweep_config
    from src.simulation_io.analysis.surface_tension import needs_calibration
    from src.simulation_io.analysis.surface_tension import record_surface_tension

    adapter = TomlAdapter()
    refreshed = 0
    for run_dir in dict.fromkeys(run_dir for path in paths for run_dir in _run_dirs(Path(path))):
        config = adapter.load(str(run_dir / CONFIG_FILENAME))
        if is_sweep_config(config) or not needs_calibration(config) or not is_calibrated(config):
            continue
        console.print(f"[bold]{run_dir}[/bold]")
        record_surface_tension(config, run_dir)
        refreshed += 1
    success(f"Refreshed {refreshed} run(s)")
