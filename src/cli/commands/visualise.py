"""The ``tud-lbm visualise`` group: the figures of one run.

``visualise fields``   per-timestep field panels of a run, of one ``.npz`` snapshot, or an animation;
``visualise analysis`` snapshot-history figures and ``simulation_data.csv`` of a run;
``visualise setup``    the paper setup schematic of a run.

Figures over several runs are ``tud-lbm compare``.
"""

from __future__ import annotations
from dataclasses import replace
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
from src.cli.field_select import prompt_overlays
from src.config.run_config import DATA_DIRNAME

if TYPE_CHECKING:
    from src.config import SimulationConfig
    from src.simulation_io.plotting import FigureBuilder

#: Registry kinds behind each ``visualise`` product.
_FIELD_KINDS = ("plotting",)
_ANALYSIS_KINDS = ("analysis",)
#: An animation frame is one composite figure, so it may hold both kinds.
_ANIMATION_KINDS = ("plotting", "analysis")

_USER_INTERRUPT = "Visualisation interrupted by user."
_RUN_CONFIG_LABEL = "this run's config.toml"
_MIN_GRID_DIMENSIONS = 2
_DEFAULT_FPS = 10
_SNAPSHOT_SUFFIX = ".npz"
_SNAPSHOT_FIELD_USAGE = "A snapshot must contain a two-dimensional 'rho' or 'u' field."
_TARGET_USAGE = "TARGET must be a run directory or a .npz snapshot file."
_ANIMATE_SNAPSHOT_USAGE = "--animate needs a run directory, not a single snapshot."
_ANIMATE_ONLY_USAGE = "--output and --fps apply only with --animate."
_ANIMATE_SKIP_USAGE = "--skip does not apply with --animate."


def _load_run_config(run_dir: str | Path) -> SimulationConfig:
    """Load the ``config.toml`` stored inside a run directory."""
    from src.config import from_toml

    return from_toml(str(_validate_run_dir_has_config(str(run_dir))))


def _is_snapshot(path: Path) -> bool:
    """Whether *path* is one saved ``.npz`` snapshot rather than a run directory."""
    return path.is_file() and path.suffix.lower() == _SNAPSHOT_SUFFIX


def _snapshot_config(snapshot_path: Path) -> SimulationConfig:
    """Infer the configuration needed to plot a standalone snapshot.

    The snapshot alone gives the grid and which of density and velocity it
    holds; no ``config.toml`` is needed.
    """
    with np.load(snapshot_path) as raw:
        fields = set(raw.files)
        field = _snapshot_geometry_field(raw, fields)

    if field.ndim < _MIN_GRID_DIMENSIONS:
        raise click.UsageError(_SNAPSHOT_FIELD_USAGE)
    nx, ny = (int(dim) for dim in field.shape[:_MIN_GRID_DIMENSIONS])

    from src.config import SimulationConfig

    plot_fields = [name for name, key in (("density", "rho"), ("velocity", "u")) if key in fields]
    return SimulationConfig(grid_shape=(nx, ny), plot_fields=plot_fields)


def _snapshot_geometry_field(raw: np.lib.npyio.NpzFile, fields: set[str]) -> np.ndarray:
    """Return the field used to infer the standalone snapshot geometry."""
    for field_name in ("rho", "u"):
        if field_name in fields:
            return raw[field_name]
    raise click.UsageError(_SNAPSHOT_FIELD_USAGE)


def _run_dir_for_snapshot(snapshot_path: Path) -> Path:
    """Return the directory whose ``plots/`` tree a standalone snapshot writes into.

    Snapshots saved by a run live in ``<run_dir>/data/``, so the figure belongs
    in ``<run_dir>/plots/`` — not in a second ``plots/`` tree nested under
    ``data/``. A snapshot sitting anywhere else is its own run directory.
    """
    parent = snapshot_path.parent
    if parent.name == DATA_DIRNAME:
        return parent.parent
    return parent


def _comma_list(value: str | None) -> list[str] | None:
    """Split a comma-separated CLI option into stripped names, or ``None`` when unset."""
    if not value:
        return None
    return [name.strip() for name in value.split(",") if name.strip()] or None


def _with_interface_levels(config: SimulationConfig, interface_levels: list[str] | None) -> SimulationConfig:
    """Apply a ``--interface-levels`` override, so operators read one source."""
    if interface_levels is None:
        return config
    return replace(config, interface_levels=interface_levels)


def _operators_for(kinds: tuple[str, ...]) -> dict:
    """Merge the selectable operators of every kind in *kinds*.

    Overlay-only operators are left out: they are chosen through the overlay
    question or ``--overlay``, never as a field.
    """
    from src.registry import get_operators

    merged: dict = {}
    for kind in kinds:
        merged.update(
            {
                name: entry
                for name, entry in get_operators(kind).items()
                if not getattr(entry.target, "overlay_only", False)
            }
        )
    return merged


def _restrict(names: list[str] | None, available: dict) -> list[str] | None:
    """Keep only *names* that exist in *available*.

    This is what makes ``visualise analysis`` honour a config whose
    ``plot_fields`` mixes both kinds: the field entries are dropped rather than
    silently rendered.
    """
    if names is None:
        return None
    return [name for name in names if name in available] or None


def _select_fields(
    *,
    explicit: str | None,
    interactive: bool,
    available: dict,
    configured: list[str],
    label: str,
) -> list[str] | None:
    """Resolve the operators to render: ``--fields``, else the prompt, else the config.

    Every source is restricted to *available*, so one kind never renders the
    operators of another.
    """
    if explicit:
        return _restrict(_comma_list(explicit), available)
    if not interactive:
        return _restrict(configured, available)
    return prompt_fields_marked(
        available,
        _restrict(configured, available),
        configured=configured,
        label=label,
        config_label=_RUN_CONFIG_LABEL,
    )


def _select_overlays(
    *,
    explicit: list[str] | None,
    interactive: bool,
    field_list: list[str] | None,
    config: SimulationConfig,
) -> list[str] | None:
    """Resolve the overlays for one invocation, asking only when it makes sense.

    ``--overlay`` always wins. Otherwise the opt-out question is asked only in
    the interactive flow, and only when a field panel could carry an overlay.
    ``None`` lets ``FigureBuilder`` fall back to the config's ``overlay_fields``.
    """
    if explicit is not None:
        return explicit
    if not interactive:
        return None
    from src.registry import get_operators

    plotting_ops = get_operators("plotting")
    if field_list is not None and not any(name in plotting_ops for name in field_list):
        return None
    return prompt_overlays(plotting_ops, config)


def _print_selection(
    target: Path,
    field_list: list[str] | None,
    overlays: list[str] | None = None,
    interface_levels: list[str] | None = None,
) -> None:
    """Print the effective inputs of one invocation."""
    console.print(f"[dim]Target        : {target}[/dim]")
    if field_list:
        console.print(f"[dim]Fields        : {', '.join(field_list)}[/dim]")
    if overlays is not None:
        console.print(f"[dim]Overlays      : {', '.join(overlays) or 'none'}[/dim]")
    if interface_levels:
        console.print(f"[dim]Interface lvls: {', '.join(interface_levels)}[/dim]")


def _report_saved(saved: list[Path], output_dir: Path) -> None:
    """Report the figures written, or that there were none."""
    if not saved:
        console.print("[yellow]No figures produced. Check that the run directory contains snapshot files.[/yellow]")
        return
    console.print(f"[bold green]{len(saved)} figure(s) saved to:[/bold green] {output_dir}")


_fields_option = click.option(
    "--fields",
    default=None,
    help="Comma-separated operator names to render; skips the interactive prompt.",
)
_no_prompt_option = click.option(
    "--no-prompt",
    "no_prompt",
    is_flag=True,
    help="Skip the interactive questions and use the run's config.toml.",
)
_dpi_option = click.option(
    "--dpi",
    default=150,
    show_default=True,
    help="Resolution in dots per inch for saved figures.",
)
_run_dir_argument = click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))


@cli.group()
def visualise() -> None:
    """Build the figures of one run: field panels, analysis history or the setup schematic.

    Figures over several runs are ``tud-lbm compare``.
    """


@visualise.command("fields")
@click.argument("target", metavar="TARGET", type=click.Path(exists=True))
@_fields_option
@_no_prompt_option
@_dpi_option
@click.option(
    "--skip",
    default=0,
    show_default=True,
    help="Number of earliest snapshots to skip.",
)
@click.option(
    "--overlay",
    "overlay",
    default=None,
    help="Comma-separated overlay operators drawn on top of every field panel (e.g. 'interface,contact_angle').",
)
@click.option(
    "--interface-levels",
    "interface_levels",
    default=None,
    help=(
        "Comma-separated interface markers for the interface contour and the contact-angle "
        "wetting band: 'config', 'measured'. Defaults to both."
    ),
)
@click.option(
    "--animate",
    is_flag=True,
    help="Write one mp4/gif of every snapshot instead of one figure per snapshot.",
)
@click.option(
    "--output",
    default=None,
    help="With --animate: output file (.mp4 or .gif). Defaults to plots/animation.mp4 inside TARGET.",
)
@click.option(
    "--fps",
    type=int,
    default=None,
    help=f"With --animate: frames per second.  [default: {_DEFAULT_FPS}]",
)
@cli_command(title="Visualisation - fields", interrupt_message=_USER_INTERRUPT)
def visualise_fields(
    target: str,
    fields: str | None,
    no_prompt: bool,
    dpi: int,
    skip: int,
    overlay: str | None,
    interface_levels: str | None,
    animate: bool,
    output: str | None,
    fps: int | None,
) -> None:
    """Build per-timestep field panels for TARGET, a run directory or one .npz snapshot.

    A snapshot is drawn without a config.toml, into the plots/ tree of the run
    that holds it. --animate encodes every snapshot of a run into one video; its
    frames may also hold analysis panels.
    """
    path = Path(target)
    snapshot = _is_snapshot(path)
    if not snapshot and not path.is_dir():
        raise click.UsageError(_TARGET_USAGE)
    if animate:
        _check_animate_options(snapshot=snapshot, skip=skip)
    elif output is not None or fps is not None:
        raise click.UsageError(_ANIMATE_ONLY_USAGE)

    levels = _comma_list(interface_levels)
    if snapshot:
        _build_snapshot_figure(path, dpi=dpi, fields=fields, overlay=overlay, interface_levels=levels)
        return

    config = _with_interface_levels(_load_run_config(path), levels)
    interactive = not fields and not no_prompt
    if animate:
        # An animation falls back to plot_fields when animate_fields is unset.
        configured = list(config.animate_fields or config.plot_fields or [])
        available = _operators_for(_ANIMATION_KINDS)
        label = "animation fields"
    else:
        configured = list(config.plot_fields or [])
        available = _operators_for(_FIELD_KINDS)
        label = "field panels"
    field_list = _select_fields(
        explicit=fields, interactive=interactive, available=available, configured=configured, label=label
    )
    overlays = _select_overlays(
        explicit=_comma_list(overlay), interactive=interactive, field_list=field_list, config=config
    )
    _print_selection(path, field_list, overlays, levels)

    if animate:
        _write_animation(config, path, dpi=dpi, fps=fps, output=output, field_list=field_list, overlays=overlays)
    else:
        _write_field_figures(config, path, dpi=dpi, skip=skip, field_list=field_list, overlays=overlays)


def _write_field_figures(
    config: SimulationConfig,
    run_dir: Path,
    *,
    dpi: int,
    skip: int,
    field_list: list[str] | None,
    overlays: list[str] | None,
) -> None:
    """Render one field figure per snapshot of *run_dir*."""
    if skip:
        console.print(f"[dim]Skip          : {skip}[/dim]")
    console.print()
    builder = _figure_builder(
        config, run_dir, dpi=dpi, field_list=field_list, available=_operators_for(_FIELD_KINDS), overlays=overlays
    )
    _report_saved(builder.build_all(skip=skip), builder.plot_dir)


def _check_animate_options(*, snapshot: bool, skip: int) -> None:
    """Reject the ``fields`` options that have no meaning for an animation."""
    if snapshot:
        raise click.UsageError(_ANIMATE_SNAPSHOT_USAGE)
    if skip:
        raise click.UsageError(_ANIMATE_SKIP_USAGE)


def _figure_builder(
    config: SimulationConfig,
    run_dir: Path,
    *,
    dpi: int,
    field_list: list[str] | None,
    available: dict,
    overlays: list[str] | None = None,
) -> FigureBuilder:
    """A ``FigureBuilder`` for *field_list*, never wider than *available*.

    An empty selection would make ``FigureBuilder`` fall back to its own default,
    which spans both kinds, so the kind's operators are named explicitly.
    """
    from src.simulation_io.plotting import FigureBuilder

    return FigureBuilder(
        config=config,
        run_dir=run_dir,
        dpi=dpi,
        fields=field_list or sorted(available),
        overlays=overlays,
    )


def _build_snapshot_figure(
    snapshot_path: Path,
    *,
    dpi: int,
    fields: str | None,
    overlay: str | None,
    interface_levels: list[str] | None,
) -> None:
    """Render one figure for a standalone ``.npz`` snapshot."""
    config = _with_interface_levels(_snapshot_config(snapshot_path), interface_levels)
    available = _operators_for(_FIELD_KINDS)
    field_list = _select_fields(
        explicit=fields,
        interactive=False,
        available=available,
        configured=list(config.plot_fields or []),
        label="field panels",
    )
    overlays = _comma_list(overlay)
    _print_selection(snapshot_path, field_list, overlays, interface_levels)
    console.print()

    builder = _figure_builder(
        config,
        _run_dir_for_snapshot(snapshot_path),
        dpi=dpi,
        field_list=field_list,
        available=available,
        overlays=overlays,
    )
    single = builder.build_single(snapshot_path)
    _report_saved([single] if single is not None else [], single.parent if single is not None else builder.plot_dir)


def _write_animation(
    config: SimulationConfig,
    run_dir: Path,
    *,
    dpi: int,
    fps: int | None,
    output: str | None,
    field_list: list[str] | None,
    overlays: list[str] | None,
) -> None:
    """Encode every snapshot of *run_dir* into one video."""
    from src.simulation_io.plotting import Animator

    fps = fps if fps is not None else _DEFAULT_FPS
    console.print(f"[dim]FPS           : {fps}[/dim]")
    console.print()
    animator = Animator(config=config, run_dir=run_dir, fps=fps, dpi=dpi, fields=field_list, overlays=overlays)
    output_path = animator.create(output)
    console.print(f"[bold green]Animation saved to:[/bold green] {output_path}")


@visualise.command("analysis")
@_run_dir_argument
@_fields_option
@_no_prompt_option
@_dpi_option
@cli_command(title="Visualisation - analysis", interrupt_message=_USER_INTERRUPT)
def visualise_analysis(run_dir: str, fields: str | None, no_prompt: bool, dpi: int) -> None:
    """Build the snapshot-history figures and simulation_data.csv of RUN_DIR."""
    path = Path(run_dir)
    config = _load_run_config(path)
    available = _operators_for(_ANALYSIS_KINDS)
    field_list = _select_fields(
        explicit=fields,
        interactive=not fields and not no_prompt,
        available=available,
        configured=list(config.plot_fields or []),
        label="analysis figures",
    )
    _print_selection(path, field_list)
    console.print()

    from src.cli.analysis_routing import analyse_run

    # The CSV comes first: the figures then read the droplet series it cached.
    csv_path = analyse_run(path, config)
    builder = _figure_builder(config, path, dpi=dpi, field_list=field_list, available=available)
    _configure_snapshot_fig(builder, field_list)
    _report_saved(builder.build_all(), builder.plot_dir)
    if csv_path is not None:
        console.print(f"[bold green]Metrics saved to:[/bold green] {csv_path}")


@visualise.command("setup")
@_run_dir_argument
@click.option(
    "--timestep",
    "timesteps",
    type=int,
    multiple=True,
    help="Snapshot timestep to draw; repeat for several. Defaults to the hysteresis snapshot and the last one.",
)
@click.option(
    "--angle-tolerance",
    default=1.0,
    show_default=True,
    help="Degrees within which the pre-step advancing/receding angles count as reached.",
)
@click.option(
    "--draw-angle",
    default=15.0,
    show_default=True,
    help="Schematic slope angle on the page, in degrees (not the run's inclination).",
)
@click.option(
    "--output",
    default=None,
    help="Output file; the suffix sets the format. Defaults to plots/analysis/setup.pdf inside RUN_DIR.",
)
@cli_command(title="Visualisation - setup", interrupt_message=_USER_INTERRUPT)
def visualise_setup(
    run_dir: str,
    timesteps: tuple[int, ...],
    angle_tolerance: float,
    draw_angle: float,
    output: str | None,
) -> None:
    """Draw the setup schematic of RUN_DIR: interface, stepped edge, boundaries and incline."""
    config = _load_run_config(run_dir)
    from src.simulation_io.plotting.setup_figure import select_setup_timesteps
    from src.simulation_io.plotting.setup_figure import snapshot_files
    from src.simulation_io.plotting.setup_figure import write_setup_figure

    available = list(snapshot_files(Path(run_dir)))
    if unknown := [step for step in timesteps if step not in available]:
        span = f"{available[0]}..{available[-1]}" if available else "none"
        msg = f"No snapshot for --timestep {unknown}; available: {span}."
        raise click.UsageError(msg)

    console.print(f"[dim]Run directory : {run_dir}[/dim]")
    if timesteps:
        chosen = timesteps
    else:
        choice = select_setup_timesteps(config, Path(run_dir), angle_tolerance)
        chosen = choice.timesteps
        console.print(f"[dim]First droplet : {choice.reason}[/dim]")
    console.print(f"[dim]Timesteps     : {', '.join(str(step) for step in chosen) or 'none'}[/dim]")
    console.print(f"[dim]Slope angle   : {draw_angle:g} deg (schematic)[/dim]")
    console.print()

    path = write_setup_figure(config, run_dir, timesteps=chosen, draw_angle_deg=draw_angle, out_path=output)
    if path is None:
        console.print("[yellow]No figure produced: the run needs snapshots and two phase densities.[/yellow]")
    else:
        console.print(f"[bold green]Setup figure saved to:[/bold green] {path}")
