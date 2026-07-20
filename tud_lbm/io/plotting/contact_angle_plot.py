"""Contact-angle history analysis plots."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tud_lbm.io.plotting._analysis_common import _CONTACT_ANGLE_Y_LABEL
from tud_lbm.io.plotting._analysis_common import _X_LABEL_TIMESTEP
from tud_lbm.io.plotting._analysis_common import _BaseAnalysisPlot
from tud_lbm.io.plotting._analysis_common import _droplet_series
from tud_lbm.io.plotting._analysis_common import _empty_series_arrays
from tud_lbm.io.plotting._analysis_common import _set_empty_state
from tud_lbm.io.plotting.base import AnalysisPlot
from tud_lbm.io.plotting.figure_config import DEFAULT_STYLE
from tud_lbm.registry import analysis_operator

if TYPE_CHECKING:
    from pathlib import Path
    import matplotlib.axes
    import numpy as np

_CONTACT_ANGLES_TITLE = "Contact angles vs timestep"


@analysis_operator(name="contact_angle_left")
class ContactAngleLeftPlot(_BaseAnalysisPlot):
    """Plot left contact angle over time."""

    name = "contact_angle_left"
    title = "Left contact angle vs timestep"
    ylabel = _CONTACT_ANGLE_Y_LABEL
    color = DEFAULT_STYLE.colors["contact_angle_left"]
    required_keys = ("ca_left",)

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Left contact angle per snapshot, derived from the density field."""
        series = _droplet_series(self.config, files)
        if series is None:
            return _empty_series_arrays("iters", "values")
        return {"iters": series.iteration, "values": series.theta_left}


@analysis_operator(name="contact_angle_right")
class ContactAngleRightPlot(_BaseAnalysisPlot):
    """Plot right contact angle over time."""

    name = "contact_angle_right"
    title = "Right contact angle vs timestep"
    ylabel = _CONTACT_ANGLE_Y_LABEL
    color = DEFAULT_STYLE.colors["contact_angle_right"]
    required_keys = ("ca_right",)

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Right contact angle per snapshot, derived from the density field."""
        series = _droplet_series(self.config, files)
        if series is None:
            return _empty_series_arrays("iters", "values")
        return {"iters": series.iteration, "values": series.theta_right}


@analysis_operator(name="contact_angles_pair")
class ContactAnglesPairPlot(AnalysisPlot):
    """Render paired left/right contact-angle history."""

    name = "contact_angles_pair"
    required_keys = ("ca_left", "ca_right")

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Left and right contact angles per snapshot, from the shared series."""
        series = _droplet_series(self.config, files)
        if series is None:
            return _empty_series_arrays("iters", "left", "right")
        return {"iters": series.iteration, "left": series.theta_left, "right": series.theta_right}

    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:
        """Draw the paired contact-angle scatter plot."""
        ax.clear()
        iters = precomputed["iters"]
        if len(iters) == 0:
            _set_empty_state(
                ax,
                title=_CONTACT_ANGLES_TITLE,
                ylabel=_CONTACT_ANGLE_Y_LABEL,
                required_keys=self.required_keys,
            )
            return
        ax.scatter(
            iters,
            precomputed["left"],
            s=DEFAULT_STYLE.scatter_marker_size,
            color=DEFAULT_STYLE.colors["contact_angle_left"],
            alpha=DEFAULT_STYLE.scatter_alpha,
            edgecolors="none",
            label="Left",
        )
        ax.scatter(
            iters,
            precomputed["right"],
            s=DEFAULT_STYLE.scatter_marker_size,
            color=DEFAULT_STYLE.colors["contact_angle_right"],
            alpha=DEFAULT_STYLE.scatter_alpha,
            edgecolors="none",
            label="Right",
        )
        ax.set_title(_CONTACT_ANGLES_TITLE)
        ax.set_xlabel(_X_LABEL_TIMESTEP)
        ax.set_ylabel(_CONTACT_ANGLE_Y_LABEL)
        ax.grid(False)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncols=2,
            fontsize=DEFAULT_STYLE.panel_legend_fontsize,
        )
