"""The ``tud-lbm animate`` and ``tud-lbm visualise`` commands."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import click
from tud_lbm.cli._console import cli_command
from tud_lbm.cli._console import console
from tud_lbm.cli.app import cli
from tud_lbm.cli.config_loading import _validate_run_dir_has_config
from tud_lbm.cli.field_select import _configure_snapshot_fig
from tud_lbm.cli.field_select import prompt_fields_marked

if TYPE_CHECKING:
    from tud_lbm.config import SimulationConfig

#: Registry kinds behind each ``visualise`` form.
_FIELD_KINDS = ("plotting",)
_ANALYSIS_KINDS = ("analysis",)
_BOTH_KINDS = ("plotting", "analysis")

_RUN_CONFIG_LABEL = "this run's config.toml"


def _load_run_config(run_dir: str) -> SimulationConfig:
    """Load the ``config.toml`` stored inside a run directory."""
    from tud_lbm.config import from_toml

    return from_toml(str(_validate_run_dir_has_config(run_dir)))


def _operators_for(kinds: tuple[str, ...]) -> dict:
    """Merge the registered operators of every kind in *kinds*."""
    from tud_lbm.registry import get_operators

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

    run_dir: str
    skip: int
    dpi: int
    fields: str | None
    no_prompt: bool


def _build_figures(ctx: VisualiseContext, kinds: tuple[str, ...]) -> None:
    """Build the figures of the given *kinds* for one run directory."""
    from tud_lbm.io.plotting import FigureBuilder

    config = _load_run_config(ctx.run_dir)
    available = _operators_for(kinds)

    console.print(f"[dim]Run directory : {ctx.run_dir}[/dim]")

    configured = list(config.plot_fields or [])
    if ctx.fields:
        field_list = _restrict([f.strip() for f in ctx.fields.split(",")], available)
    elif ctx.no_prompt:
        field_list = _restrict(configured, available)
    else:
        field_list = prompt_fields_marked(
            available,
            _restrict(configured, available),
            configured=configured,
            label="visualisation fields",
            config_label=_RUN_CONFIG_LABEL,
        )

    if field_list:
        console.print(f"[dim]Fields        : {', '.join(field_list)}[/dim]")
    if ctx.skip:
        console.print(f"[dim]Skip          : {ctx.skip}[/dim]")
    console.print(f"[dim]DPI           : {ctx.dpi}[/dim]")
    console.print()

    # An empty selection would make FigureBuilder fall back to its own default,
    # which spans both kinds — so name the kind's operators explicitly.
    builder = FigureBuilder(
        config=config,
        run_dir=ctx.run_dir,
        dpi=ctx.dpi,
        fields=field_list or sorted(available),
    )
    _configure_snapshot_fig(builder, field_list)
    saved = builder.build_all(skip=ctx.skip)

    if not saved:
        console.print("[yellow]No figures produced. Check that the run directory contains snapshot files.[/yellow]")
    else:
        console.print(f"[bold green]{len(saved)} figure(s) saved to:[/bold green] {builder.plot_dir}")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
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
    from tud_lbm.io.plotting import Animator

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
@click.argument("run_dir", type=click.Path(exists=True))
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
@click.pass_context
def visualise(
    ctx: click.Context,
    run_dir: str,
    skip: int,
    dpi: int,
    fields: str | None,
    no_prompt: bool,
) -> None:
    """Build static figures for saved snapshots in RUN_DIR.

    With no subcommand this builds both per-timestep field snapshots and
    snapshot-history analysis figures.
    """
    ctx.obj = VisualiseContext(run_dir=run_dir, skip=skip, dpi=dpi, fields=fields, no_prompt=no_prompt)
    if ctx.invoked_subcommand is None:
        _visualise_both(ctx.obj)


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
_visualise_both = cli_command(title="Visualisation", interrupt_message="Visualisation interrupted by user.")(
    lambda obj: _build_figures(obj, _BOTH_KINDS)
)
_visualise_fields_only = cli_command(
    title="Visualisation - fields", interrupt_message="Visualisation interrupted by user."
)(lambda obj: _build_figures(obj, _FIELD_KINDS))
_visualise_analysis_only = cli_command(
    title="Visualisation - analysis", interrupt_message="Visualisation interrupted by user."
)(lambda obj: _build_figures(obj, _ANALYSIS_KINDS))
