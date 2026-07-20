"""Contact-line speed history analysis plots."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from tud_lbm.io.plotting._analysis_common import _CONTACT_LINE_SPEED_Y_LABEL
from tud_lbm.io.plotting._analysis_common import _X_LABEL_TIMESTEP
from tud_lbm.io.plotting._analysis_common import _BaseAnalysisPlot
from tud_lbm.io.plotting._analysis_common import _load_timesteps
from tud_lbm.io.plotting._analysis_common import _set_empty_state
from tud_lbm.io.plotting.base import AnalysisPlot
from tud_lbm.io.plotting.figure_config import DEFAULT_STYLE
from tud_lbm.registry import analysis_operator

if TYPE_CHECKING:
    from pathlib import Path
    import matplotlib.axes

_CONTACT_LINE_SPEEDS_TITLE = "Contact-line speeds vs timestep"


class _ContactLineSpeedBase(_BaseAnalysisPlot):
    cl_key: str

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        iters, snapshots = _load_timesteps(files, (self.cl_key,))
        cl = np.asarray([float(np.asarray(s[self.cl_key]).squeeze()) for s in snapshots], dtype=float)
        if len(cl) == 0:
            return {"iters": iters, "values": cl}
        if len(cl) == 1:
            return {"iters": iters, "values": np.asarray([0.0], dtype=float)}
        d_iter = np.diff(iters).astype(float)
        d_iter[d_iter == 0] = np.nan
        speeds = np.diff(cl) / d_iter
        vals = np.concatenate(([0.0], speeds))
        vals = np.nan_to_num(vals, nan=0.0)
        return {"iters": iters, "values": vals}


@analysis_operator(name="contact_line_speed_left")
class ContactLineSpeedLeftPlot(_ContactLineSpeedBase):
    """Plot left contact-line speed over time."""

    name = "contact_line_speed_left"
    title = "Left contact-line speed vs timestep"
    ylabel = "d(cll_left)/dt"
    color = DEFAULT_STYLE.colors["contact_line_speed_left"]
    cl_key = "cll_left"
    required_keys = ("cll_left",)


@analysis_operator(name="contact_line_speed_right")
class ContactLineSpeedRightPlot(_ContactLineSpeedBase):
    """Plot right contact-line speed over time."""

    name = "contact_line_speed_right"
    title = "Right contact-line speed vs timestep"
    ylabel = "d(cll_right)/dt"
    color = DEFAULT_STYLE.colors["contact_line_speed_right"]
    cl_key = "cll_right"
    required_keys = ("cll_right",)


@analysis_operator(name="contact_line_speeds_pair")
class ContactLineSpeedsPairPlot(AnalysisPlot):
    """Render paired left/right contact-line speed history."""

    name = "contact_line_speeds_pair"
    required_keys = ("cll_left", "cll_right")

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute left/right contact-line speed arrays for all snapshots."""
        left = ContactLineSpeedLeftPlot().compute(files)
        right = ContactLineSpeedRightPlot().compute(files)
        iters = left["iters"] if len(left["iters"]) >= len(right["iters"]) else right["iters"]
        return {"iters": iters, "left": left["values"], "right": right["values"]}

    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:
        """Draw the paired contact-line speed scatter plot."""
        ax.clear()
        iters = precomputed["iters"]
        left = precomputed["left"]
        right = precomputed["right"]
        if len(iters) == 0:
            _set_empty_state(
                ax,
                title=_CONTACT_LINE_SPEEDS_TITLE,
                ylabel=_CONTACT_LINE_SPEED_Y_LABEL,
                required_keys=self.required_keys,
            )
            return
        ax.scatter(
            iters[: len(left)],
            left,
            s=DEFAULT_STYLE.scatter_marker_size,
            color=DEFAULT_STYLE.colors["contact_line_speed_left"],
            alpha=DEFAULT_STYLE.scatter_alpha,
            edgecolors="none",
            label="Left",
        )
        ax.scatter(
            iters[: len(right)],
            right,
            s=DEFAULT_STYLE.scatter_marker_size,
            color=DEFAULT_STYLE.colors["contact_line_speed_right"],
            alpha=DEFAULT_STYLE.scatter_alpha,
            edgecolors="none",
            label="Right",
        )
        ax.set_title(_CONTACT_LINE_SPEEDS_TITLE)
        ax.set_xlabel(_X_LABEL_TIMESTEP)
        ax.set_ylabel(_CONTACT_LINE_SPEED_Y_LABEL)
        ax.grid(False)
        ax.legend(loc="best", fontsize=DEFAULT_STYLE.pair_legend_fontsize)
