"""Composite figure: ca_theta_vs_x with density/velocity snapshots at chosen timesteps."""

from __future__ import annotations
import string
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from tud_lbm.io.plotting._analysis_common import _CONTACT_ANGLE_Y_LABEL
from tud_lbm.io.plotting._analysis_common import _parse_timestep
from tud_lbm.io.plotting._analysis_common import _set_empty_state
from tud_lbm.io.plotting.base import AnalysisPlot
from tud_lbm.io.plotting.ca_theta_plot import _CA_THETA_TITLE
from tud_lbm.io.plotting.ca_theta_plot import _LABEL_X_AVG_NORM
from tud_lbm.io.plotting.ca_theta_plot import DualAxisStyle
from tud_lbm.io.plotting.ca_theta_plot import _compute_ca_theta_arrays
from tud_lbm.io.plotting.ca_theta_plot import _draw_dual_axis_on_ax
from tud_lbm.io.plotting.density import DensityPlotOperator
from tud_lbm.io.plotting.figure_config import DEFAULT_STYLE
from tud_lbm.io.plotting.velocity import VelocityPlotOperator
from tud_lbm.registry import comparison_operator

if TYPE_CHECKING:
    from pathlib import Path
    import matplotlib.axes
    import matplotlib.figure
    import matplotlib.gridspec
    from tud_lbm.config import SimulationConfig

_MAX_SNAPSHOTS = len(string.ascii_uppercase)
_MAX_SNAPSHOT_COLS = 5

# imshow(aspect="equal") shrinks each panel to the colorbar+title-adjusted axes
# width before _strip_panel_chrome clears them; bias the predicted image height
# upward (effective width down) so the GridSpec cell never undershoots the
# rendered image and the two panels overlap.
_PANEL_WIDTH_SHRINK = 0.85
# Gap (inches) reserved between the dual-axis plot and the snapshot grid below
# it — just enough for its x-tick labels and x-axis label to render into.
_DUAL_AXIS_LABEL_GAP_IN = 1.2
# Gap (inches) between stacked rows of snapshot blocks when nrows > 1.
_ROW_GAP_IN = 0.3


def _strip_panel_chrome(ax: matplotlib.axes.Axes) -> None:
    """Remove a snapshot panel's colorbar, axis labels, ticks, and title."""
    for im in ax.images:
        if im.colorbar is not None:
            im.colorbar.remove()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")


def _render_snapshot_block(
    fig: matplotlib.figure.Figure,
    snapshot_gs: matplotlib.gridspec.GridSpecBase,
    row_block: int,
    col: int,
    actual_ts: int,
    fp: Path | None,
    density_op: DensityPlotOperator,
    velocity_op: VelocityPlotOperator,
) -> tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]:
    """Render one density/velocity panel pair stacked with no gap between them."""
    block_gs = snapshot_gs[row_block, col].subgridspec(2, 1, hspace=0.0)
    ax_rho = fig.add_subplot(block_gs[0])
    ax_u = fig.add_subplot(block_gs[1])
    # imshow's equal-aspect box would otherwise centre each image with padding,
    # opening a gap at the shared edge even at hspace=0; anchor both to that edge.
    ax_rho.set_anchor("S")
    ax_u.set_anchor("N")

    if fp is not None:
        with np.load(fp) as raw:
            snap = {key: np.asarray(raw[key]) for key in raw.files}
        if density_op.is_available(snap):
            density_op(ax_rho, snap, actual_ts)
        if velocity_op.is_available(snap):
            velocity_op(ax_u, snap, actual_ts)

    for panel_ax in (ax_rho, ax_u):
        _strip_panel_chrome(panel_ax)
    return ax_rho, ax_u


def _label_block(
    fig: matplotlib.figure.Figure, letter: str, ax_rho: matplotlib.axes.Axes, ax_u: matplotlib.axes.Axes
) -> None:
    """Place the block's letter to the left of the stacked panels, vertically centred."""
    pos_rho = ax_rho.get_position()
    pos_u = ax_u.get_position()
    y_center = (pos_rho.y1 + pos_u.y0) / 2
    fig.text(
        pos_rho.x0 - 0.01,
        y_center,
        letter,
        transform=fig.transFigure,
        ha="right",
        va="center",
        fontweight="bold",
        fontsize=2 * DEFAULT_STYLE.title_fontsize,
    )


def _move_legend_left_center(ax: matplotlib.axes.Axes) -> None:
    """Reposition the dual-axis legend to sit left-of-axes, vertically centred."""
    legend = ax.get_legend()
    if legend is None:
        return
    handles = [h for h in legend.legend_handles if h is not None]
    labels = [text.get_text() for text in legend.get_texts()]
    fontsize = legend.get_texts()[0].get_fontsize() if legend.get_texts() else DEFAULT_STYLE.legend_fontsize
    legend.remove()
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(0.02, 0.5), fontsize=fontsize)


@comparison_operator(name="snapshot_fig")
class SnapshotOverviewPlot(AnalysisPlot):
    """ca_theta_vs_x plot with lettered density/velocity snapshots alongside it."""

    name = "snapshot_fig"
    required_keys = ("ca_left", "ca_right", "cll_left", "cll_right")
    is_multi_panel = True

    def __init__(self, config: SimulationConfig | None = None) -> None:
        """Initialize with optional config; timesteps are set by the CLI prompt."""
        super().__init__(config)
        self.timesteps: list[int] = []

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:  # noqa: ARG002
        """Unused: this operator builds its figure via ``render_figure`` instead."""
        return {}

    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:  # noqa: ARG002
        """Render a placeholder; this operator is only meaningful via render_figure()."""
        _set_empty_state(
            ax,
            title="Snapshot overview",
            ylabel=_CONTACT_ANGLE_Y_LABEL,
            required_keys=("only supported via `tud-lbm visualise --fields snapshot_fig`",),
        )

    def render_figure(self, files: list[Path]) -> matplotlib.figure.Figure:
        """Build the ca_theta_vs_x plot with lettered density/velocity snapshots."""
        if self.config is None or not self.timesteps:
            fig, ax = plt.subplots(figsize=DEFAULT_STYLE.analysis_figsize)
            _set_empty_state(ax, title=_CA_THETA_TITLE, ylabel=_CONTACT_ANGLE_Y_LABEL, required_keys=("timesteps",))
            return fig

        if len(self.timesteps) > _MAX_SNAPSHOTS:
            msg = f"snapshot_fig supports at most {_MAX_SNAPSHOTS} timesteps (A-Z)."
            raise ValueError(msg)

        data = _compute_ca_theta_arrays(files, self.config, compute_x_pos=True)
        ts_arr = data["timesteps"]
        if ts_arr.size == 0:
            fig, ax = plt.subplots(figsize=DEFAULT_STYLE.analysis_figsize)
            _set_empty_state(ax, title=_CA_THETA_TITLE, ylabel=_CONTACT_ANGLE_Y_LABEL)
            return fig

        file_by_ts = {_parse_timestep(fp.stem): fp for fp in files if _parse_timestep(fp.stem) is not None}

        # Nearest-match each requested timestep to an available one, dedupe, sort ascending
        # so the right-hand snapshot column reads top-to-bottom in chronological order.
        matched: dict[int, float] = {}
        for requested in self.timesteps:
            idx = int(np.argmin(np.abs(ts_arr - requested)))
            matched[int(ts_arr[idx])] = float(data["x_pos"][idx])
        ordered = sorted(matched.items())

        n = len(ordered)
        ncols = min(n, _MAX_SNAPSHOT_COLS)
        nrows = -(-n // _MAX_SNAPSHOT_COLS)  # ceil division

        nx, ny = self.config.grid_shape[0], self.config.grid_shape[1]
        col_width = DEFAULT_STYLE.panel_figsize[0]
        image_height = col_width * (ny / nx) * _PANEL_WIDTH_SHRINK
        block_height = 2 * image_height
        snapshot_height = nrows * block_height + (nrows - 1) * _ROW_GAP_IN
        dual_axis_height = DEFAULT_STYLE.dual_axis_figsize[1]
        fig_width = max(DEFAULT_STYLE.dual_axis_figsize[0], col_width * ncols)
        fig_height = dual_axis_height + _DUAL_AXIS_LABEL_GAP_IN + snapshot_height

        fig = plt.figure(figsize=(fig_width, fig_height))
        outer_gs = fig.add_gridspec(
            3,
            1,
            height_ratios=[dual_axis_height, _DUAL_AXIS_LABEL_GAP_IN, snapshot_height],
            hspace=0.0,
        )
        ax_main = fig.add_subplot(outer_gs[0])
        _draw_dual_axis_on_ax(
            ax_main,
            data["x_pos"],
            data["ca_trailing"],
            data["ca_leading"],
            data["theta_trailing"],
            data["theta_leading"],
            x_label=_LABEL_X_AVG_NORM,
            style=DualAxisStyle(
                legend_fontsize=DEFAULT_STYLE.panel_legend_fontsize * 2,
                legend_outside=True,
                axis_label_fontsize=2 * DEFAULT_STYLE.axis_label_fontsize,
                tick_label_fontsize=2 * DEFAULT_STYLE.tick_label_fontsize,
            ),
        )
        _move_legend_left_center(ax_main)
        snapshot_gs = outer_gs[2].subgridspec(nrows, ncols, hspace=0.15, wspace=0.15)

        density_op = DensityPlotOperator(self.config)
        velocity_op = VelocityPlotOperator(self.config)
        line_trans = ax_main.get_xaxis_transform()  # x in data coords, y in axes coords

        blocks: list[tuple[str, matplotlib.axes.Axes, matplotlib.axes.Axes]] = []
        for idx, (letter, (actual_ts, x_val)) in enumerate(zip(string.ascii_uppercase, ordered, strict=False)):
            row_block, col = divmod(idx, _MAX_SNAPSHOT_COLS)
            ax_main.axvline(x_val, color="grey", linestyle="--", linewidth=1)
            ax_main.text(x_val, 1.02, letter, transform=line_trans, ha="center", fontweight="bold")
            ax_rho, ax_u = _render_snapshot_block(
                fig,
                snapshot_gs,
                row_block,
                col,
                actual_ts,
                file_by_ts.get(actual_ts),
                density_op,
                velocity_op,
            )
            ax_rho.set_label(letter)
            blocks.append((letter, ax_rho, ax_u))
        fig.subplots_adjust(top=0.93, bottom=0.02, left=0.1, right=0.95)
        for letter, ax_rho, ax_u in blocks:
            _label_block(fig, letter, ax_rho, ax_u)
        return fig
