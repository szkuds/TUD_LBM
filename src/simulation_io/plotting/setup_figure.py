"""Setup figure: the simulated domain drawn as an inclined, chemically stepped slope.

One schematic, built from a run directory, that shows what a paper's setup
figure has to: the interface of the inclusion at ``rho_mean`` (solid red), the
wall split into its two surfaces, the boundary condition on every edge as an
abbreviation with a key, the contact angles, gravity and the inclination.

Two snapshots are drawn on the same slope by default. The second is the last
one. The first is the first snapshot at which the contact angles have reached
the pre-step hysteresis window — the front line at its advancing angle and the
rear line at its receding angle — read from ``simulation_data.csv``, so the
droplet shows the hysteresis rather than its symmetric initial shape.

Everything is built in the domain frame (the cell-centre coordinates of
``imshow(rho.T, origin="lower")``, which :func:`interface_lines` returns) and
rotated as a whole. The slope is drawn at a *schematic* angle, not the run's
own: a 201x51 domain at 60 degrees is unreadable. The rotation maps the run's
gravity direction for that schematic angle to straight down the page.

Public module: it registers no operator, so the auto-discovery scan of this
package must not pick it up. Matplotlib is imported function-locally.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from src.config.run_config import ANALYSIS_DIRNAME
from src.config.run_config import DATA_DIRNAME
from src.config.run_config import PLOTS_DIRNAME
from src.config.run_config import SETUP_FIGURE_FILENAME
from src.config.run_config import SIMULATION_CSV_FILENAME
from src.config.run_config import SNAPSHOT_GLOB
from src.simulation_io.analysis.droplet_metrics import extract_rho_2d
from src.simulation_io.analysis.droplet_metrics import inclination_angle_deg
from src.simulation_io.analysis.droplet_metrics import parse_timestep
from src.simulation_io.analysis.interface_contour import config_rho_mean
from src.simulation_io.analysis.interface_contour import interface_lines
from src.simulation_io.analysis.wetting_overlay import SIDES
from src.simulation_io.analysis.wetting_overlay import WALL_NORMAL
from src.simulation_io.analysis.wetting_overlay import angle_glyph
from src.simulation_io.analysis.wetting_overlay import contact_angles
from src.simulation_io.analysis.wetting_overlay import to_physical
from src.simulation_io.analysis.wetting_overlay import wetting_edge
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

if TYPE_CHECKING:
    from collections.abc import Sequence
    import matplotlib.axes
    from src.config import SimulationConfig
    from src.simulation_io.analysis.wetting_overlay import Side

#: Abbreviation and key text per boundary-condition name. A name missing here is
#: labelled with the name itself, so a new BC is never mislabelled.
BC_ABBREVIATIONS: dict[str, tuple[str, str]] = {
    "periodic": ("PBC", "periodic"),
    "symmetry": ("FS", "free-slip (symmetry)"),
    "bounce-back": ("NS", "no-slip"),
    "wetting": ("NS+W", "no-slip + wetting"),
}

#: In-plane edges with their outward normal in the domain frame.
_EDGE_NORMALS: dict[str, tuple[float, float]] = {
    "bottom": (0.0, -1.0),
    "top": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
}

_FIGURE_WIDTH = 8.0
_ALPHA_ARC_POINTS = 24


@dataclass(frozen=True)
class SnapshotChoice:
    """The timesteps a setup figure draws, and why."""

    timesteps: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class _Wall:
    """The wetting wall in its own frame: tangential extent, step and surfaces."""

    edge: str
    length: int
    step: float | None
    #: Whether the surface before the step is the less wetting one.
    pre_is_hydrophobic: bool


# --- Snapshot selection ------------------------------------------------------


def snapshot_files(run_dir: Path) -> dict[int, Path]:
    """The run's ``.npz`` snapshots keyed by timestep, in ascending order."""
    found = {
        step: path
        for path in (Path(run_dir) / DATA_DIRNAME).glob(SNAPSHOT_GLOB)
        if (step := parse_timestep(path.stem)) is not None
    }
    return dict(sorted(found.items()))


def _front_side(config: SimulationConfig) -> Side:
    """The downhill contact line: gravity drives the liquid along ``+sin(angle)``."""
    return "left" if inclination_angle_deg(config) < 0.0 else "right"


def _hysteresis_timestep(config: SimulationConfig, run_dir: Path, tolerance: float) -> tuple[int, str] | None:
    """First CSV row whose angles have reached the pre-step hysteresis window.

    Both bounds are required first — front at the advancing angle *and* rear at
    the receding one — because the advancing bound alone is met while the rear
    line is still relaxing, which does not show the window. Falls back to the
    advancing bound alone. ``None`` when the run has no chemical step, no CSV,
    or no row qualifies.
    """
    step = config.chemical_step_wall
    chem = config.chemical_step_config
    csv_path = Path(run_dir) / SIMULATION_CSV_FILENAME
    if step is None or chem is None or not csv_path.is_file():
        return None
    import pandas as pd

    frame = pd.read_csv(csv_path, encoding="utf-8")
    front = _front_side(config)
    rear: Side = "left" if front == "right" else "right"
    on_pre = frame[f"cll_{front}"] < step.step_x
    advancing = frame[f"ca_{front}"] >= float(chem["ca_advancing_pre_step"]) - tolerance
    receding = frame[f"ca_{rear}"] <= float(chem["ca_receding_pre_step"]) + tolerance
    for mask, reason in (
        (on_pre & advancing & receding, "advancing and receding angles reached"),
        (on_pre & advancing, "advancing angle reached (receding angle never is)"),
    ):
        if mask.any():
            return int(frame.loc[mask, "iteration"].iloc[0]), reason
    return None


def select_setup_timesteps(config: SimulationConfig, run_dir: Path, angle_tolerance_deg: float = 1.0) -> SnapshotChoice:
    """The default pair of snapshots: the hysteresis snapshot and the last one.

    A wall with one surface has no before and after, so only the last snapshot
    is drawn. Otherwise the first falls back to the run's first snapshot when the hysteresis rule
    cannot be applied. :attr:`SnapshotChoice.reason` says which happened.
    """
    steps = list(snapshot_files(run_dir))
    if not steps:
        return SnapshotChoice((), "no snapshots")
    if config.chemical_step_wall is None:
        return SnapshotChoice((steps[-1],), "last snapshot only (the wall has one surface)")
    found = _hysteresis_timestep(config, run_dir, angle_tolerance_deg)
    if found is not None and found[0] in steps:
        first, reason = found
    else:
        first, reason = steps[0], "first snapshot (no hysteresis snapshot could be read from the CSV)"
    return SnapshotChoice(tuple(dict.fromkeys((first, steps[-1]))), reason)


# --- Geometry ----------------------------------------------------------------


def page_rotation_deg(config: SimulationConfig, draw_angle_deg: float) -> float:
    """Rotation of the domain frame that draws the slope at *draw_angle_deg*.

    Gravity acts on the liquid along ``(sin a, -cos a)`` in the domain frame, so
    rotating by ``-a`` puts it straight down the page. The schematic angle takes
    the sign of the run's own, and a run without an inclination is not rotated.
    """
    inclination = inclination_angle_deg(config) if _has_gravity(config) else 0.0
    if abs(inclination) <= 0.0:
        return 0.0
    return -math.copysign(draw_angle_deg, inclination)


def rotate(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate ``(k, 2)`` points anticlockwise about the origin."""
    angle = math.radians(angle_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    return np.asarray(points, dtype=float) @ np.array([[cos, sin], [-sin, cos]])


def _has_gravity(config: SimulationConfig) -> bool:
    return any(name.startswith("gravity") for name in config.active_forces)


def _shape(config: SimulationConfig) -> tuple[int, int]:
    return int(config.grid_shape[0]), int(config.grid_shape[1])


def _wall(config: SimulationConfig) -> _Wall | None:
    """The wetting wall of *config*, or ``None`` for a run without one."""
    edge = wetting_edge(config)
    if edge is None:
        return None
    nx, ny = _shape(config)
    length = nx if edge in ("bottom", "top") else ny
    step = config.chemical_step_wall
    chem = config.chemical_step_config
    if step is None or chem is None:
        return _Wall(edge, length, None, pre_is_hydrophobic=False)
    return _Wall(
        edge,
        length,
        step.step_x,
        pre_is_hydrophobic=float(chem["ca_advancing_pre_step"]) > float(chem["ca_advancing_post_step"]),
    )


def _wall_points(
    wall: _Wall, tangential: Sequence[float], depth: Sequence[float], shape: tuple[int, int]
) -> np.ndarray:
    """Domain-frame points at *tangential* along the wall, *depth* into the solid."""
    x, y = to_physical(tangential, [WALL_NORMAL - d for d in depth], wall.edge, shape)
    return np.column_stack((x, y))


def _wall_segments(wall: _Wall) -> list[tuple[float, float, str]]:
    """``(start, stop, surface)`` per wall surface; surface is ``pre``, ``post`` or ``plain``."""
    start, stop = -0.5, wall.length - 0.5
    if wall.step is None:
        return [(start, stop, "plain")]
    return [(start, wall.step, "pre"), (wall.step, stop, "post")]


def _surface_color(wall: _Wall, surface: str) -> str:
    if surface == "plain":
        return DEFAULT_STYLE.setup_wall_colors["plain"]
    hydrophobic = (surface == "pre") == wall.pre_is_hydrophobic
    return DEFAULT_STYLE.setup_wall_colors["hydrophobic" if hydrophobic else "hydrophilic"]


def edge_labels(config: SimulationConfig) -> dict[str, str]:
    """Abbreviation per in-plane edge; a stepped wetting wall is labelled per surface when drawn."""
    return {
        entry.edge: BC_ABBREVIATIONS.get(entry.name, (entry.name, entry.name))[0]
        for entry in config.boundary_edges
        if entry.edge in _EDGE_NORMALS
    }


def key_text(config: SimulationConfig) -> str:
    """The key under the figure: each abbreviation in use, and the surfaces' angle windows."""
    names = dict.fromkeys(entry.name for entry in config.boundary_edges if entry.edge in _EDGE_NORMALS)
    entries = [
        f"{abbr}: {text}" for abbr, text in (BC_ABBREVIATIONS[name] for name in names if name in BC_ABBREVIATIONS)
    ]
    lines = ["    ".join(entries)]
    chem = config.chemical_step_config
    if chem is not None and config.chemical_step_wall is not None:
        lines.append(
            "    ".join(
                rf"$W_{index}$: $\theta_A={float(chem[f'ca_advancing_{surface}_step']):g}^\circ$, "
                rf"$\theta_R={float(chem[f'ca_receding_{surface}_step']):g}^\circ$"
                for index, surface in ((1, "pre"), (2, "post"))
            )
        )
    return "\n".join(line for line in lines if line)


def _angle_symbols(config: SimulationConfig, centre: float, wall: _Wall) -> dict[Side, str]:
    """Label per contact line: rear/front under gravity, else the surface it stands on."""
    if _has_gravity(config) and abs(inclination_angle_deg(config)) > 0.0:
        front = _front_side(config)
        return {side: r"$\theta_A$" if side == front else r"$\theta_R$" for side in SIDES}
    if wall.step is None:
        return dict.fromkeys(SIDES, r"$\theta$")
    return dict.fromkeys(SIDES, r"$\theta_1$" if centre < wall.step else r"$\theta_2$")


# --- Drawing -----------------------------------------------------------------


class _Canvas:
    """An axes that takes domain-frame geometry and draws it rotated."""

    def __init__(self, ax: matplotlib.axes.Axes, rotation_deg: float) -> None:
        self.ax = ax
        self.rotation_deg = rotation_deg
        self._extent: list[np.ndarray] = []

    def page(self, points: np.ndarray) -> np.ndarray:
        """Domain-frame points in page coordinates, recorded for the axis limits."""
        rotated = rotate(np.atleast_2d(points), self.rotation_deg)
        self._extent.append(rotated)
        return rotated

    def line(
        self, points: np.ndarray, *, color: str, linewidth: float, linestyle: str = "-", gid: str | None = None
    ) -> None:
        xy = self.page(points)
        self.ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=linewidth, linestyle=linestyle, gid=gid)

    def text(self, point: Sequence[float], label: str, *, ha: str = "center", va: str = "center") -> None:
        x, y = self.page(np.asarray(point, dtype=float))[0]
        self.ax.text(
            x,
            y,
            label,
            ha=ha,
            va=va,
            fontsize=DEFAULT_STYLE.setup_fontsize,
            color=DEFAULT_STYLE.setup_annotation_color,
        )

    def finish(self, margin: float) -> None:
        """Fit the limits to everything drawn through :meth:`page`."""
        extent = np.vstack(self._extent)
        (x_lo, y_lo), (x_hi, y_hi) = extent.min(axis=0), extent.max(axis=0)
        self.ax.set_xlim(x_lo - margin, x_hi + margin)
        self.ax.set_ylim(y_lo - margin, y_hi + margin)
        self.ax.set_aspect("equal")
        self.ax.axis("off")


def _draw_domain(canvas: _Canvas, shape: tuple[int, int]) -> None:
    nx, ny = shape
    corners = np.array([[-0.5, -0.5], [nx - 0.5, -0.5], [nx - 0.5, ny - 0.5], [-0.5, ny - 0.5], [-0.5, -0.5]])
    canvas.line(
        corners,
        color=DEFAULT_STYLE.setup_annotation_color,
        linewidth=DEFAULT_STYLE.setup_outline_linewidth,
        linestyle=DEFAULT_STYLE.setup_outline_linestyle,
        gid="domain",
    )


def _draw_wall(canvas: _Canvas, wall: _Wall, shape: tuple[int, int], thickness: float) -> None:
    """The solid, as one filled band per surface on the solid side of the wall."""
    from matplotlib.patches import Polygon

    for start, stop, surface in _wall_segments(wall):
        corners = _wall_points(wall, [start, stop, stop, start], [0.0, 0.0, thickness, thickness], shape)
        canvas.ax.add_patch(
            Polygon(canvas.page(corners), closed=True, color=_surface_color(wall, surface), linewidth=0, gid=surface)
        )


def _draw_edge_labels(
    canvas: _Canvas, config: SimulationConfig, wall: _Wall | None, shape: tuple[int, int], pad: float, thickness: float
) -> None:
    nx, ny = shape
    centre = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0])
    half = np.array([nx / 2.0, ny / 2.0])
    for edge, label in edge_labels(config).items():
        normal = np.array(_EDGE_NORMALS[edge])
        ha = {"left": "right", "right": "left"}.get(edge, "center")
        va = {"bottom": "top", "top": "bottom"}.get(edge, "center")
        if wall is not None and edge == wall.edge:
            segments = _wall_segments(wall)
            for index, (start, stop, _) in enumerate(segments, start=1):
                text = label if len(segments) == 1 else rf"{label}$_{index}$"
                anchor = _wall_points(wall, [(start + stop) / 2.0], [thickness + pad], shape)[0]
                canvas.text(anchor, text, ha=ha, va=va)
            continue
        canvas.text(centre + normal * (half + pad), label, ha=ha, va=va)


def _draw_droplet(
    canvas: _Canvas,
    config: SimulationConfig,
    data: dict[str, np.ndarray],
    wall: _Wall | None,
    rho_mean: float,
) -> np.ndarray | None:
    """One snapshot's interface and contact-angle glyphs; returns the interface centroid."""
    rho_2d = extract_rho_2d(data["rho"])
    lines = interface_lines(rho_2d, rho_mean)
    for line in lines:
        canvas.line(
            line,
            color=DEFAULT_STYLE.setup_interface_color,
            linewidth=DEFAULT_STYLE.setup_interface_linewidth,
            gid="interface",
        )
    angles = contact_angles(data, rho_2d, config)
    if wall is not None and angles is not None:
        shape = (rho_2d.shape[0], rho_2d.shape[1])
        length = DEFAULT_STYLE.contact_angle_length_fraction * min(shape)
        symbols = _angle_symbols(config, 0.5 * (angles.cll_left + angles.cll_right), wall)
        for side in SIDES:
            glyph = angle_glyph(angles, side, shape, length)
            for polyline in (glyph.tangent, glyph.arc):
                canvas.line(
                    polyline,
                    color=DEFAULT_STYLE.setup_annotation_color,
                    linewidth=DEFAULT_STYLE.setup_annotation_linewidth,
                )
            canvas.text(glyph.label_xy, symbols[side])
    return np.vstack(lines).mean(axis=0) if lines else None


def _draw_gravity(canvas: _Canvas, centroid: np.ndarray, length: float) -> None:
    """``F_g`` from the inclusion's centre, straight down the page."""
    tail = canvas.page(centroid)[0]
    head = tail - np.array([0.0, length])
    canvas._extent.append(head[None, :])  # noqa: SLF001 - the arrow tip sets the limits too
    canvas.ax.annotate(
        "",
        xy=tuple(head),
        xytext=tuple(tail),
        arrowprops={
            "arrowstyle": "-|>",
            "color": DEFAULT_STYLE.setup_annotation_color,
            "linewidth": DEFAULT_STYLE.setup_annotation_linewidth,
        },
        annotation_clip=False,
    ).set_gid("gravity")
    canvas.ax.text(
        head[0] - 0.1 * length,
        head[1] + 0.5 * length,
        r"$F_g$",
        ha="right",
        va="center",
        fontsize=DEFAULT_STYLE.setup_fontsize,
        color=DEFAULT_STYLE.setup_annotation_color,
    )


def _draw_inclination(canvas: _Canvas, wall: _Wall, shape: tuple[int, int], thickness: float) -> None:
    """Horizontal reference from the slope's downhill corner, with the ``alpha`` arc.

    The slope is whichever wall-parallel edge is lowest on the page: the solid's
    outer face for a wall underneath the fluid, the opposite edge for a wall on top.
    """
    across = shape[1] if wall.edge in ("bottom", "top") else shape[0]
    faces = [
        canvas.page(_wall_points(wall, [-0.5, wall.length - 0.5], [depth, depth], shape))
        for depth in (thickness, -float(across))
    ]
    ends = min(faces, key=lambda face: float(face[:, 1].min()))
    low, high = (ends[0], ends[1]) if ends[0, 1] <= ends[1, 1] else (ends[1], ends[0])
    uphill = high - low
    span = float(np.hypot(*uphill))
    direction = math.copysign(1.0, uphill[0])
    reference = np.array([low, low + np.array([direction * span * math.cos(math.radians(canvas.rotation_deg)), 0.0])])
    canvas._extent.append(reference)  # noqa: SLF001 - already in page coordinates
    canvas.ax.plot(
        reference[:, 0],
        reference[:, 1],
        color=DEFAULT_STYLE.setup_annotation_color,
        linewidth=DEFAULT_STYLE.setup_outline_linewidth,
        gid="horizontal",
    )
    base = 0.0 if direction > 0 else math.pi
    sweep = np.linspace(base, math.atan2(uphill[1], uphill[0]), _ALPHA_ARC_POINTS)
    radius = 0.12 * span
    canvas.ax.plot(
        low[0] + radius * np.cos(sweep),
        low[1] + radius * np.sin(sweep),
        color=DEFAULT_STYLE.setup_annotation_color,
        linewidth=DEFAULT_STYLE.setup_annotation_linewidth,
        gid="alpha",
    )
    middle = 0.5 * (sweep[0] + sweep[-1])
    canvas.ax.text(
        low[0] + 1.3 * radius * math.cos(middle),
        low[1] + 1.3 * radius * math.sin(middle),
        r"$\alpha$",
        ha="center",
        va="center",
        fontsize=DEFAULT_STYLE.setup_fontsize,
        color=DEFAULT_STYLE.setup_annotation_color,
    )


def write_setup_figure(
    config: SimulationConfig,
    run_dir: str | Path,
    *,
    timesteps: Sequence[int] | None = None,
    angle_tolerance_deg: float = 1.0,
    draw_angle_deg: float = 15.0,
    out_path: str | Path | None = None,
) -> Path | None:
    """Write the setup figure of the run in *run_dir*.

    Args:
        config: The run's configuration.
        run_dir: Run directory holding ``data/timestep_*.npz``.
        timesteps: Snapshots to draw. Defaults to :func:`select_setup_timesteps`.
        angle_tolerance_deg: Tolerance of the default snapshot rule.
        draw_angle_deg: Schematic slope angle on the page.
        out_path: Destination; its suffix sets the format. Defaults to
            ``plots/analysis/setup.pdf`` inside *run_dir*.

    Returns:
        The figure path, or ``None`` when the run has no snapshot to draw or no
        ``rho_mean`` to contour at.

    Raises:
        ValueError: If a requested timestep has no snapshot.
    """
    run_dir = Path(run_dir)
    files = snapshot_files(run_dir)
    rho_mean = config_rho_mean(config)
    chosen = (
        tuple(timesteps)
        if timesteps is not None
        else select_setup_timesteps(config, run_dir, angle_tolerance_deg).timesteps
    )
    if not chosen or rho_mean is None:
        return None
    if missing := [step for step in chosen if step not in files]:
        msg = f"No snapshot for timestep(s) {missing}; available: {min(files)}..{max(files)}"
        raise ValueError(msg)

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from src.simulation_io.plotting._analysis_common import load_snapshot

    shape = _shape(config)
    wall = _wall(config)
    thickness = DEFAULT_STYLE.setup_wall_thickness_fraction * min(shape)
    pad = 0.5 * thickness
    fig, ax = plt.subplots(figsize=(_FIGURE_WIDTH, _FIGURE_WIDTH * 0.6))
    canvas = _Canvas(ax, page_rotation_deg(config, draw_angle_deg))

    _draw_domain(canvas, shape)
    if wall is not None:
        _draw_wall(canvas, wall, shape, thickness)
    _draw_edge_labels(canvas, config, wall, shape, pad, thickness)
    centroids = [_draw_droplet(canvas, config, load_snapshot(files[step]), wall, rho_mean) for step in chosen]
    if _has_gravity(config) and centroids[0] is not None:
        _draw_gravity(canvas, centroids[0], 0.4 * min(shape))
    if wall is not None and abs(canvas.rotation_deg) > 0.0:
        _draw_inclination(canvas, wall, shape, thickness)
    canvas.finish(margin=4.0 * pad)
    ax.text(
        0.5,
        -0.02,
        key_text(config),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=DEFAULT_STYLE.setup_key_fontsize,
        linespacing=1.6,
        gid="key",
    )

    dest = (
        Path(out_path) if out_path is not None else run_dir / PLOTS_DIRNAME / ANALYSIS_DIRNAME / SETUP_FIGURE_FILENAME
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=DEFAULT_STYLE.dpi, bbox_inches="tight")
    plt.close(fig)
    return dest
