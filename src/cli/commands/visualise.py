"""The ``tud-lbm animate`` and ``tud-lbm visualise`` commands."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import click
import numpy as np
from src.cli._console import cli_command
from src.cli._console import console
from src.cli.app import cli
from src.cli.config_loading import _validate_run_dir_has_config
from src.cli.field_select import _configure_snapshot_fig
from src.cli.field_select import prompt_fields_marked

if TYPE_CHECKING:
    from src.config import SimulationConfig
    from src.simulation_io.plotting import FigureBuilder

#: Registry kinds behind each ``visualise`` form.
_FIELD_KINDS = ("plotting",)
_ANALYSIS_KINDS = ("analysis",)
_BOTH_KINDS = ("plotting", "analysis")

_USER_INTERRUPT = "Visualisation interrupted by user."
_RUN_CONFIG_LABEL = "this run's config.toml"
_MIN_GRID_DIMENSIONS = 2
_SINGLE_SNAPSHOT_USAGE = "--single requires PATH to point to an existing .npz snapshot file."
_SINGLE_SNAPSHOT_FIELD_USAGE = "--single requires a snapshot containing a two-dimensional 'rho' or 'u' field."
_RUN_DIRECTORY_USAGE = "PATH must point to an existing run directory unless --single is provided."


def _load_run_config(run_dir: str | Path) -> SimulationConfig:
    """Load the ``config.toml`` stored inside a run directory."""
    from src.config import from_toml

    return from_toml(str(_validate_run_dir_has_config(str(run_dir))))


def _resolve_single_snapshot(path_arg: str | Path) -> tuple[Path, SimulationConfig]:
    """Resolve a standalone snapshot and infer the configuration needed to plot it."""
    snapshot_path = Path(path_arg)
    _validate_single_snapshot_path(snapshot_path)

    with np.load(snapshot_path) as raw:
        fields = set(raw.files)
        field = _single_snapshot_field(raw, fields)

    _validate_single_snapshot_field(field)
    nx, ny = (int(dim) for dim in field.shape[:_MIN_GRID_DIMENSIONS])

    from src.config import SimulationConfig

    plot_fields = [name for name, key in (("density", "rho"), ("velocity", "u")) if key in fields]
    return snapshot_path, SimulationConfig(grid_shape=(nx, ny), plot_fields=plot_fields)


def _validate_single_snapshot_path(snapshot_path: Path) -> None:
    """Ensure ``--single`` receives one existing ``.npz`` snapshot file."""
    if snapshot_path.suffix.lower() == ".npz" and snapshot_path.is_file():
        return
    raise click.UsageError(_SINGLE_SNAPSHOT_USAGE)


def _single_snapshot_field(raw: np.lib.npyio.NpzFile, fields: set[str]) -> np.ndarray:
    """Return the field used to infer the standalone snapshot geometry."""
    for field_name in ("rho", "u"):
        if field_name in fields:
            return raw[field_name]
    raise click.UsageError(_SINGLE_SNAPSHOT_FIELD_USAGE)


def _validate_single_snapshot_field(field: np.ndarray) -> None:
    """Ensure the standalone snapshot exposes at least two grid dimensions."""
    if field.ndim >= _MIN_GRID_DIMENSIONS:
        return
    raise click.UsageError(_SINGLE_SNAPSHOT_FIELD_USAGE)


def _operators_for(kinds: tuple[str, ...]) -> dict:
    """Merge the registered operators of every kind in *kinds*."""
    from src.registry import get_operators

    merged: dict = {}
    for kind in kinds:
        merged.update(get_operators(kind))
    return merged


def _restrict(names: list[str] | None, available: dict) -> list[str] | None:
    """Keep only *names* that exist in *available*.

    This is what makes ``visualise RUN_DIR analysis`` honour a config whose
    ``plot_fields`` mixes both kinds: the field entries are dropped rather than
    silently rendered.
    """
    if names is None:
        return None
    return [name for name in names if name in available] or None


@dataclass(frozen=True)
class VisualiseContext:
    """Options bound on the ``visualise`` group, shared with its subcommands."""

    run_dir: Path
    snapshot_path: Path | None
    config: SimulationConfig | None
    skip: int
    dpi: int
    fields: str | None
    no_prompt: bool


def _build_figures(ctx: VisualiseContext, kinds: tuple[str, ...]) -> None:
    """Build the figures of the given *kinds* for one run directory."""
    from src.simulation_io.plotting import FigureBuilder

    config = ctx.config or _load_run_config(ctx.run_dir)
    available = _operators_for(kinds)

    field_list = _select_visualise_fields(ctx, config, available)
    _print_visualise_summary(ctx, field_list)

    # An empty selection would make FigureBuilder fall back to its own default,
    # which spans both kinds — so name the kind's operators explicitly.
    builder = FigureBuilder(
        config=config,
        run_dir=ctx.run_dir,
        dpi=ctx.dpi,
        fields=field_list or sorted(available),
    )
    _configure_snapshot_fig(builder, field_list)
    saved = _build_requested_figures(builder, ctx)

    if not saved:
        console.print("[yellow]No figures produced. Check that the run directory contains snapshot files.[/yellow]")
    else:
        output_dir = _figure_output_dir(builder, ctx, saved)
        console.print(f"[bold green]{len(saved)} figure(s) saved to:[/bold green] {output_dir}")


def _select_visualise_fields(ctx: VisualiseContext, config: SimulationConfig, available: dict) -> list[str] | None:
    """Resolve which operators ``visualise`` should render."""
    configured = list(config.plot_fields or [])
    if ctx.fields:
        return _restrict([f.strip() for f in ctx.fields.split(",")], available)
    if ctx.no_prompt or ctx.snapshot_path is not None:
        return _restrict(configured, available)
    return prompt_fields_marked(
        available,
        _restrict(configured, available),
        configured=configured,
        label="visualisation fields",
        config_label=_RUN_CONFIG_LABEL,
    )


def _print_visualise_summary(ctx: VisualiseContext, field_list: list[str] | None) -> None:
    """Print the effective inputs for a ``visualise`` invocation."""
    console.print(f"[dim]Run directory : {ctx.run_dir}[/dim]")
    if ctx.snapshot_path is not None:
        console.print(f"[dim]Snapshot      : {ctx.snapshot_path}[/dim]")
    if field_list:
        console.print(f"[dim]Fields        : {', '.join(field_list)}[/dim]")
    if ctx.skip and ctx.snapshot_path is None:
        console.print(f"[dim]Skip          : {ctx.skip}[/dim]")
    console.print(f"[dim]DPI           : {ctx.dpi}[/dim]")
    console.print()


def _build_requested_figures(builder: FigureBuilder, ctx: VisualiseContext) -> list[Path]:
    """Render either one standalone snapshot or every snapshot in a run directory."""
    if ctx.snapshot_path is None:
        return builder.build_all(skip=ctx.skip)
    single = builder.build_single(ctx.snapshot_path)
    return [single] if single is not None else []


def _figure_output_dir(builder: FigureBuilder, ctx: VisualiseContext, saved: list[Path]) -> Path:
    """Return the directory reported after figures are produced."""
    if ctx.snapshot_path is None:
        return builder.plot_dir
    return saved[0].parent


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--output",
    default=None,
    help="Output file path (.mp4 or .gif). Defaults to plots/animation.mp4 inside RUN_DIR.",
)
@click.option(
    "--fps",
    default=10,
    show_default=True,
    help="Frames per second for the output video.",
)
@click.option(
    "--fields",
    default=None,
    help="Comma-separated operator names to activate — skips the interactive prompt.",
)
@click.option(
    "--no-prompt",
    "no_prompt",
    is_flag=True,
    help="Skip interactive field selection and use config defaults.",
)
@cli_command(title="Animation", interrupt_message="Animation interrupted by user.")
def animate(run_dir: str, output: str | None, fps: int, fields: str | None, no_prompt: bool) -> None:
    """Animate saved snapshots in RUN_DIR.

    A frame is one composite figure holding both field and analysis panels, so
    unlike ``visualise`` there is nothing to split into subcommands.
    """
    from src.simulation_io.plotting import Animator

    config = _load_run_config(run_dir)
    available = _operators_for(_BOTH_KINDS)

    console.print(f"[dim]Run directory : {run_dir}[/dim]")
    console.print(f"[dim]FPS           : {fps}[/dim]")

    # animate falls back to plot_fields, unlike visualise which only reads it.
    configured = list(config.animate_fields or config.plot_fields or [])
    if fields:
        selected = [f.strip() for f in fields.split(",")] or None
    elif no_prompt:
        selected = configured or None
    else:
        selected = prompt_fields_marked(
            available,
            configured or None,
            configured=configured,
            label="animation fields",
            config_label=_RUN_CONFIG_LABEL,
        )

    if selected:
        console.print(f"[dim]Fields        : {', '.join(selected)}[/dim]")
    console.print()

    animator = Animator(config=config, run_dir=run_dir, fps=fps, fields=selected)
    output_path = animator.create(output)

    console.print(f"[bold green]Animation saved to:[/bold green] {output_path}")


# Groups disable interspersed args by default, which would reject the natural
# ``visualise RUN_DIR --no-prompt`` ordering. The subcommands take no options of
# their own, so nothing is ambiguous.
@cli.group(invoke_without_command=True, context_settings={"allow_interspersed_args": True})
@click.argument("path_arg", metavar="PATH", type=click.Path(exists=True))
@click.option(
    "--skip",
    default=0,
    show_default=True,
    help="Number of earliest timestep files to skip.",
)
@click.option(
    "--dpi",
    default=150,
    show_default=True,
    help="Resolution in dots per inch for saved figures.",
)
@click.option(
    "--fields",
    default=None,
    help="Comma-separated operator names to activate — skips the interactive prompt.",
)
@click.option(
    "--no-prompt",
    "no_prompt",
    is_flag=True,
    help="Skip interactive field selection and use config defaults.",
)
@click.option(
    "--single",
    is_flag=True,
    help="Treat PATH as one .npz snapshot and render it beside the source file.",
)
@click.pass_context
def visualise(
    ctx: click.Context,
    path_arg: str,
    skip: int,
    dpi: int,
    fields: str | None,
    no_prompt: bool,
    single: bool,
) -> None:
    """Build static figures for saved snapshots in RUN_DIR or for one ``.npz`` state.

    With no subcommand this builds both per-timestep field snapshots and
    snapshot-history analysis figures. ``--single`` instead renders exactly one
    figure beside the given ``.npz`` snapshot, without requiring ``config.toml``.
    """
    ctx.obj = _visualise_context(
        path_arg=path_arg,
        skip=skip,
        dpi=dpi,
        fields=fields,
        no_prompt=no_prompt,
        single=single,
    )
    if ctx.invoked_subcommand is None:
        _visualise_both(ctx.obj)


def _visualise_context(
    path_arg: str,
    skip: int,
    dpi: int,
    fields: str | None,
    no_prompt: bool,
    single: bool,
) -> VisualiseContext:
    """Build the shared context for the ``visualise`` group and subcommands."""
    if single:
        snapshot_path, config = _resolve_single_snapshot(path_arg)
        run_dir = snapshot_path.parent
    else:
        run_dir = Path(path_arg)
        if not run_dir.is_dir():
            raise click.UsageError(_RUN_DIRECTORY_USAGE)
        snapshot_path, config = None, None

    return VisualiseContext(
        run_dir=run_dir,
        snapshot_path=snapshot_path,
        config=config,
        skip=skip,
        dpi=dpi,
        fields=fields,
        no_prompt=no_prompt,
    )


@visualise.command("fields")
@click.pass_obj
def visualise_fields(obj: VisualiseContext) -> None:
    """Build only per-timestep field snapshots (density, velocity, force)."""
    _visualise_fields_only(obj)


@visualise.command("analysis")
@click.pass_obj
def visualise_analysis(obj: VisualiseContext) -> None:
    """Build only snapshot-history analysis figures."""
    _visualise_analysis_only(obj)


# click invokes subcommands *after* the group callback returns, so a decorator
# on the group callback would never see a subcommand's exceptions. Decorate the
# work functions instead.
_visualise_both = cli_command(title="Visualisation", interrupt_message=_USER_INTERRUPT)(
    lambda obj: _build_figures(obj, _BOTH_KINDS)
)
_visualise_fields_only = cli_command(title="Visualisation - fields", interrupt_message=_USER_INTERRUPT)(
    lambda obj: _build_figures(obj, _FIELD_KINDS)
)
_visualise_analysis_only = cli_command(title="Visualisation - analysis", interrupt_message=_USER_INTERRUPT)(
    lambda obj: _build_figures(obj, _ANALYSIS_KINDS)
)
