"""The ``tud-lbm sweep`` commands: a parameter sweep as staged run configs.

``stage`` writes one completed TOML per combination of the ``--set`` axes. A
sweep is never run in process: the staged files go through ``tud-lbm run`` or
``scripts/db_pipeline.sh`` like any other config.
"""

from __future__ import annotations
from pathlib import Path
import click
from rich.table import Table
from src.cli._console import cli_command
from src.cli._console import console
from src.cli._console import success
from src.cli.app import cli
from src.cli.config_loading import _load_raw_config
from src.cli.sweep_staging import parse_axes
from src.cli.sweep_staging import stage_configs
from src.cli.sweep_staging import staged_path
from src.cli.sweep_staging import write_staged


@cli.group()
def sweep() -> None:
    """Stage parameter sweeps as one run config per combination."""


@sweep.command()
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--set",
    "sets",
    multiple=True,
    required=True,
    help="A swept parameter as PATH=[v1, v2, ...] (repeatable; --override path syntax). "
    "The combinations are the Cartesian product of all axes.",
)
@click.option(
    "--name",
    "template",
    default=None,
    help="Run-name template over the last key of each axis, e.g. 'var_inc_inc_{inclination_angle_deg}'. "
    "Default: <simulation_name>_<key>_<value> per axis.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory for the staged TOMLs (default: <config stem>_sweep/ beside CONFIG_TOML).",
)
@click.option("--override", "overrides", multiple=True, help="Override a config value before the axes (repeatable).")
@click.option("--force", is_flag=True, help="Overwrite staged TOMLs that already exist.")
@click.option("--dry-run", is_flag=True, help="List the combinations without writing any file.")
@cli_command(title="Parameter sweep staging", interrupt_message="Staging interrupted by user.")
def stage(
    *,
    config_toml: str,
    sets: tuple[str, ...],
    template: str | None,
    out_dir: str | None,
    overrides: tuple[str, ...],
    force: bool,
    dry_run: bool,
) -> None:
    r"""Write one completed run config per combination of the --set axes.

    Each file is <simulation_name>.toml, validated and with every default filled
    in, like a run directory's own config.toml.

    Examples:
        tud-lbm sweep stage cfg.toml --set 'gravity_force.inclination_angle_deg=[20, 40, 60]'

        tud-lbm sweep stage cfg.toml --set 'tau=[0.85, 0.99]' --set 'wetting.phi_left=[1.0, 1.1]' \
            --name 'var_tau_{tau}_phi_{phi_left}' --out configs/var_tau/
    """
    axes = parse_axes(sets)
    staged = stage_configs(_load_raw_config(config_toml, overrides), axes, template)
    target = Path(out_dir) if out_dir else Path(config_toml).with_name(f"{Path(config_toml).stem}_sweep")

    table = Table(title=f"Parameter sweep ({len(staged)} runs)", header_style="bold magenta")
    table.add_column("Run", style="cyan", no_wrap=True)
    for axis in axes:
        table.add_column(axis.path, style="green")
    for entry in staged:
        table.add_row(str(entry.config.simulation_name), *(str(entry.values[axis.path]) for axis in axes))
    console.print(table)

    if dry_run:
        for entry in staged:
            console.print(f"  [dim]{staged_path(target, entry.config)}[/dim]")
        console.print("[yellow]Dry run mode - nothing written[/yellow]")
        return
    paths = write_staged(staged, target, force=force)
    success(f"Staged {len(paths)} run configs in {target}")
