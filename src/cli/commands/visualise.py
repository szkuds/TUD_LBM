"""The ``tud-lbm visualise`` group: the per-run figures, of one run or of several.

``visualise fields``   per-timestep field panels of a run, of one ``.npz`` snapshot, or an animation;
``visualise analysis`` snapshot-history figures and ``simulation_data.csv`` of a run;
``visualise setup``    the paper setup schematic of a run.

Each command takes one or more TARGETs. A TARGET is a run directory, a directory
that holds run directories at any depth, or a ``.txt`` list of run directories
(:func:`_expand_targets`). Every run gets its own figures in its own ``plots/``
tree, with one field selection for all of them. A run that fails is skipped and
the others continue; the command then exits 1 and names it (:class:`_RunBatch`). Figures that overlay several
runs are ``tud-lbm compare``.
"""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
import click
import numpy as np
from src.cli._console import cli_command
from src.cli._console import console
from src.cli._console import debug_mode
from src.cli.app import cli
from src.cli.config_loading import _validate_run_dir_has_config
from src.cli.field_select import _configure_snapshot_fig
from src.cli.field_select import prompt_fields_marked
from src.cli.field_select import prompt_overlays
from src.config.run_config import CONFIG_FILENAME
from src.config.run_config import DATA_DIRNAME

if TYPE_CHECKING:
    from collections.abc import Callable
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
_TARGET_USAGE = "TARGET must be a run directory, a directory of runs, a .txt run list or a .npz snapshot file."
_RUN_TARGET_USAGE = "TARGET must be a run directory, a directory of runs or a .txt run list."
_ANIMATE_SNAPSHOT_USAGE = "--animate needs a run directory, not a single snapshot."
_ANIMATE_ONLY_USAGE = "--output and --fps apply only with --animate."
_ANIMATE_SKIP_USAGE = "--skip does not apply with --animate."
_SINGLE_OUTPUT_USAGE = "--output names one file, so it needs exactly one run."


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


def _expand_targets(
    targets: tuple[str, ...],
    allowed_roots: tuple[str, ...] = (),
    *,
    snapshots: bool = False,
) -> list[Path]:
    """Expand the TARGET arguments into run directories (and snapshots), in order.

    - a directory with a ``config.toml`` is one run;
    - a directory without one gives every run directory beneath it, or itself
      when there is none, so the missing ``config.toml`` is reported for it;
    - a ``.txt`` file gives the run directories it lists;
    - a ``.npz`` file is one snapshot, only when *snapshots* is true.

    A path named twice is kept once.

    Raises:
        click.UsageError: A target is none of the above, or the targets hold no run.
    """
    from src.cli.analysis_routing import RUN_LIST_SUFFIX
    from src.cli.analysis_routing import find_run_dirs
    from src.cli.analysis_routing import parse_run_dir_list

    expanded: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            has_config = (path / CONFIG_FILENAME).is_file()
            expanded.extend([path] if has_config else (find_run_dirs(path) or [path]))
        elif path.is_file() and path.suffix.lower() == RUN_LIST_SUFFIX:
            expanded.extend(parse_run_dir_list(path, allowed_roots))
        elif snapshots and _is_snapshot(path):
            expanded.append(path)
        else:
            msg = f"{path}: {_TARGET_USAGE if snapshots else _RUN_TARGET_USAGE}"
            raise click.UsageError(msg)
    if not expanded:
        msg = f"No run directories in {', '.join(targets)}."
        raise click.UsageError(msg)
    return list(dict.fromkeys(expanded))


class _RunBatch:
    """Runs several targets one after another, and skips a run that fails.

    With one run, or with ``TUD_LBM_DEBUG`` set, an error propagates as it
    always did, so a single run keeps its exit code and a debug run its
    traceback. With several, a failed run is reported and the next one starts;
    the command then exits 1 and names every failed run.
    """

    def __init__(self, run_dirs: list[Path]) -> None:
        self.run_dirs = run_dirs
        self.failed: dict[Path, str] = {}
        self._skip = len(run_dirs) > 1 and not debug_mode()

    def attempt(self, run_dir: Path, work: Callable[[Path], None]) -> bool:
        """Do *work* for *run_dir*; whether it succeeded."""
        if run_dir in self.failed:
            return False
        if not self._skip:
            work(run_dir)
            return True
        try:
            work(run_dir)
        except Exception as exc:  # noqa: BLE001
            # One broken run must not stop the plots of the others.
            self.failed[run_dir] = str(exc)
            console.print(f"[bold red]Skipped[/bold red] {run_dir}: {exc}")
            return False
        return True

    def first_config(self, load: Callable[[Path], SimulationConfig]) -> SimulationConfig:
        """The config of the first run that loads, which marks the selection prompt."""
        loaded: list[SimulationConfig] = []
        for run_dir in self.run_dirs:
            if self.attempt(run_dir, lambda path: loaded.append(load(path))):
                return loaded[0]
        self.finish()
        raise AssertionError  # finish() raises: every run failed.

    def each(self, work: Callable[[Path], None]) -> None:
        """Do *work* for every run not failed yet, then report the failures."""
        for run_dir in self.run_dirs:
            self.attempt(run_dir, work)
        self.finish()

    def finish(self) -> None:
        """Raise one error that names every failed run, when there is one."""
        if not self.failed:
            return
        names = "\n".join(f"  {run_dir}: {reason}" for run_dir, reason in self.failed.items())
        msg = f"{len(self.failed)} of {len(self.run_dirs)} runs failed and were skipped:\n{names}"
        raise RuntimeError(msg)


def _print_run_count(paths: list[Path]) -> None:
    """Print how many runs one invocation works on, when there is more than one."""
    if len(paths) > 1:
        console.print(f"[dim]Runs          : {len(paths)}[/dim]")


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
_targets_argument = click.argument(
    "targets", metavar="TARGET...", nargs=-1, required=True, type=click.Path(exists=True)
)
_allowed_root_option = click.option(
    "--allowed-root",
    "allowed_roots",
    multiple=True,
    type=click.Path(exists=True, file_okay=False),
    help="With a .txt list: an additional directory that a listed run directory may resolve within, beyond "
    "the default results root (repeatable).",
)


@cli.group()
def visualise() -> None:
    """Build the figures of each run: field panels, analysis history or the setup schematic.

    Every command takes one or more TARGETs: a run directory, a directory of
    runs or a .txt run list. Figures that overlay several runs are ``tud-lbm compare``.
    """


@visualise.command("fields")
@_targets_argument
@_allowed_root_option
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
    help="With --animate and one run: output file (.mp4 or .gif). Defaults to plots/animation.mp4 inside the run.",
)
@click.option(
    "--fps",
    type=int,
    default=None,
    help=f"With --animate: frames per second.  [default: {_DEFAULT_FPS}]",
)
@cli_command(title="Visualisation - fields", interrupt_message=_USER_INTERRUPT)
def visualise_fields(
    targets: tuple[str, ...],
    allowed_roots: tuple[str, ...],
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
    """Build per-timestep field panels for each TARGET.

    A TARGET is a run directory, a directory of runs, a .txt run list or one
    .npz snapshot. A snapshot is drawn without a config.toml, into the plots/
    tree of the run that holds it. --animate encodes every snapshot of a run
    into one video; its frames may also hold analysis panels. The fields are
    selected once and used for every run.
    """
    paths = _expand_targets(targets, allowed_roots, snapshots=True)
    snapshot_paths = [path for path in paths if _is_snapshot(path)]
    run_dirs = [path for path in paths if not _is_snapshot(path)]
    if animate:
        _check_animate_options(snapshot=bool(snapshot_paths), skip=skip)
        if output is not None and len(run_dirs) > 1:
            raise click.UsageError(_SINGLE_OUTPUT_USAGE)
    elif output is not None or fps is not None:
        raise click.UsageError(_ANIMATE_ONLY_USAGE)

    levels = _comma_list(interface_levels)
    for snapshot_path in snapshot_paths:
        _build_snapshot_figure(snapshot_path, dpi=dpi, fields=fields, overlay=overlay, interface_levels=levels)
    if not run_dirs:
        return

    # The first run's config marks the prompt; the selection then holds for every run.
    _print_run_count(run_dirs)
    batch = _RunBatch(run_dirs)
    first = _with_interface_levels(batch.first_config(_load_run_config), levels)
    interactive = not fields and not no_prompt
    if animate:
        # An animation falls back to plot_fields when animate_fields is unset.
        configured = list(first.animate_fields or first.plot_fields or [])
        available = _operators_for(_ANIMATION_KINDS)
        label = "animation fields"
    else:
        configured = list(first.plot_fields or [])
        available = _operators_for(_FIELD_KINDS)
        label = "field panels"
    field_list = _select_fields(
        explicit=fields, interactive=interactive, available=available, configured=configured, label=label
    )
    overlays = _select_overlays(
        explicit=_comma_list(overlay), interactive=interactive, field_list=field_list, config=first
    )

    def _work(run_dir: Path) -> None:
        config = _with_interface_levels(_load_run_config(run_dir), levels)
        _print_selection(run_dir, field_list, overlays, levels)
        if animate:
            _write_animation(config, run_dir, dpi=dpi, fps=fps, output=output, field_list=field_list, overlays=overlays)
        else:
            _write_field_figures(config, run_dir, dpi=dpi, skip=skip, field_list=field_list, overlays=overlays)

    batch.each(_work)


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
@_targets_argument
@_allowed_root_option
@_fields_option
@_no_prompt_option
@_dpi_option
@cli_command(title="Visualisation - analysis", interrupt_message=_USER_INTERRUPT)
def visualise_analysis(
    targets: tuple[str, ...], allowed_roots: tuple[str, ...], fields: str | None, no_prompt: bool, dpi: int
) -> None:
    """Build the snapshot-history figures and simulation_data.csv of each TARGET.

    A TARGET is a run directory, a directory of runs or a .txt run list. The
    figures are selected once and used for every run.
    """
    run_dirs = _expand_targets(targets, allowed_roots)
    _print_run_count(run_dirs)
    batch = _RunBatch(run_dirs)
    first = batch.first_config(_load_run_config)
    available = _operators_for(_ANALYSIS_KINDS)
    field_list = _select_fields(
        explicit=fields,
        interactive=not fields and not no_prompt,
        available=available,
        configured=list(first.plot_fields or []),
        label="analysis figures",
    )
    batch.each(
        lambda run_dir: _write_analysis(
            _load_run_config(run_dir), run_dir, dpi=dpi, field_list=field_list, available=available
        )
    )


def _write_analysis(
    config: SimulationConfig,
    run_dir: Path,
    *,
    dpi: int,
    field_list: list[str] | None,
    available: dict,
) -> None:
    """Write the ``simulation_data.csv`` and the analysis figures of one run."""
    from src.cli.analysis_routing import analyse_run

    _print_selection(run_dir, field_list)
    console.print()
    # The CSV comes first: the figures then read the droplet series it cached.
    csv_path = analyse_run(run_dir, config)
    builder = _figure_builder(config, run_dir, dpi=dpi, field_list=field_list, available=available)
    _configure_snapshot_fig(builder, field_list)
    _report_saved(builder.build_all(), builder.plot_dir)
    if csv_path is not None:
        console.print(f"[bold green]Metrics saved to:[/bold green] {csv_path}")


@visualise.command("setup")
@_targets_argument
@_allowed_root_option
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
    help="With one run: output file; the suffix sets the format. Defaults to plots/analysis/setup.pdf inside the run.",
)
@cli_command(title="Visualisation - setup", interrupt_message=_USER_INTERRUPT)
def visualise_setup(
    targets: tuple[str, ...],
    allowed_roots: tuple[str, ...],
    timesteps: tuple[int, ...],
    angle_tolerance: float,
    draw_angle: float,
    output: str | None,
) -> None:
    """Draw the setup schematic of each TARGET: interface, stepped edge, boundaries and incline.

    A TARGET is a run directory, a directory of runs or a .txt run list.
    """
    run_dirs = _expand_targets(targets, allowed_roots)
    if output is not None and len(run_dirs) > 1:
        raise click.UsageError(_SINGLE_OUTPUT_USAGE)
    _print_run_count(run_dirs)
    _RunBatch(run_dirs).each(
        lambda run_dir: _write_setup(
            run_dir, timesteps=timesteps, angle_tolerance=angle_tolerance, draw_angle=draw_angle, output=output
        )
    )


def _write_setup(
    run_dir: Path,
    *,
    timesteps: tuple[int, ...],
    angle_tolerance: float,
    draw_angle: float,
    output: str | None,
) -> None:
    """Draw the setup schematic of one run."""
    config = _load_run_config(run_dir)
    from src.simulation_io.plotting.setup_figure import select_setup_timesteps
    from src.simulation_io.plotting.setup_figure import snapshot_files
    from src.simulation_io.plotting.setup_figure import write_setup_figure

    available = list(snapshot_files(run_dir))
    if unknown := [step for step in timesteps if step not in available]:
        span = f"{available[0]}..{available[-1]}" if available else "none"
        msg = f"{run_dir}: no snapshot for --timestep {unknown}; available: {span}."
        raise click.UsageError(msg)

    console.print(f"[dim]Run directory : {run_dir}[/dim]")
    if timesteps:
        chosen = timesteps
    else:
        choice = select_setup_timesteps(config, run_dir, angle_tolerance)
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
