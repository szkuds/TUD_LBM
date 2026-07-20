"""Contact-line speed history analysis plots."""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.registry import analysis_operator
from src.simulation_io.plotting._analysis_common import _CONTACT_LINE_SPEED_Y_LABEL
from src.simulation_io.plotting._analysis_common import _X_LABEL_TIMESTEP
from src.simulation_io.plotting._analysis_common import _BaseAnalysisPlot
from src.simulation_io.plotting._analysis_common import _droplet_series
from src.simulation_io.plotting._analysis_common import _empty_series_arrays
from src.simulation_io.plotting._analysis_common import _set_empty_state
from src.simulation_io.plotting.base import AnalysisPlot
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

if TYPE_CHECKING:
    from pathlib import Path
    import matplotlib.axes
    import numpy as np

_CONTACT_LINE_SPEEDS_TITLE = "Contact-line speeds vs timestep"


class _ContactLineSpeedBase(_BaseAnalysisPlot):
    """Contact-line speed for one side, differentiated over actual iteration gaps."""

    #: Attribute on :class:`DropletSeries` holding this side's speed.
    speed_attr: str

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Contact-line speed per snapshot, from the shared series."""
        series = _droplet_series(self.config, files)
        if series is None:
            return _empty_series_arrays("iters", "values")
        return {"iters": series.iteration, "values": getattr(series, self.speed_attr)}


@analysis_operator(name="contact_line_speed_left")
class ContactLineSpeedLeftPlot(_ContactLineSpeedBase):
    """Plot left contact-line speed over time."""

    name = "contact_line_speed_left"
    title = "Left contact-line speed vs timestep"
    ylabel = "d(cll_left)/dt"
    color = DEFAULT_STYLE.colors["contact_line_speed_left"]
    speed_attr = "v_left"
    required_keys = ("cll_left",)


@analysis_operator(name="contact_line_speed_right")
class ContactLineSpeedRightPlot(_ContactLineSpeedBase):
    """Plot right contact-line speed over time."""

    name = "contact_line_speed_right"
    title = "Right contact-line speed vs timestep"
    ylabel = "d(cll_right)/dt"
    color = DEFAULT_STYLE.colors["contact_line_speed_right"]
    speed_attr = "v_right"
    required_keys = ("cll_right",)


@analysis_operator(name="contact_line_speeds_pair")
class ContactLineSpeedsPairPlot(AnalysisPlot):
    """Render paired left/right contact-line speed history."""

    name = "contact_line_speeds_pair"
    required_keys = ("cll_left", "cll_right")

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Left and right contact-line speeds per snapshot, from the shared series."""
        series = _droplet_series(self.config, files)
        if series is None:
            return _empty_series_arrays("iters", "left", "right")
        return {"iters": series.iteration, "left": series.v_left, "right": series.v_right}

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
