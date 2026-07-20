"""Contact-angle history analysis plots."""

from __future__ import annotations
from src.registry import analysis_operator
from src.simulation_io.plotting._analysis_common import _CONTACT_ANGLE_Y_LABEL
from src.simulation_io.plotting._analysis_common import _PairSeriesPlot
from src.simulation_io.plotting._analysis_common import _SeriesAttrPlot
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

_CONTACT_ANGLES_TITLE = "Contact angles vs timestep"


@analysis_operator(name="contact_angle_left")
class ContactAngleLeftPlot(_SeriesAttrPlot):
    """Plot left contact angle over time."""

    name = "contact_angle_left"
    title = "Left contact angle vs timestep"
    ylabel = _CONTACT_ANGLE_Y_LABEL
    color = DEFAULT_STYLE.colors["contact_angle_left"]
    series_attr = "theta_left"
    required_keys = ("ca_left",)


@analysis_operator(name="contact_angle_right")
class ContactAngleRightPlot(_SeriesAttrPlot):
    """Plot right contact angle over time."""

    name = "contact_angle_right"
    title = "Right contact angle vs timestep"
    ylabel = _CONTACT_ANGLE_Y_LABEL
    color = DEFAULT_STYLE.colors["contact_angle_right"]
    series_attr = "theta_right"
    required_keys = ("ca_right",)


@analysis_operator(name="contact_angles_pair")
class ContactAnglesPairPlot(_PairSeriesPlot):
    """Render paired left/right contact-angle history."""

    name = "contact_angles_pair"
    title = _CONTACT_ANGLES_TITLE
    ylabel = _CONTACT_ANGLE_Y_LABEL
    left_attr = "theta_left"
    right_attr = "theta_right"
    left_color = DEFAULT_STYLE.colors["contact_angle_left"]
    right_color = DEFAULT_STYLE.colors["contact_angle_right"]
    legend_kwargs = {  # noqa: RUF012
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.02),
        "ncols": 2,
        "fontsize": DEFAULT_STYLE.panel_legend_fontsize,
    }
    required_keys = ("ca_left", "ca_right")
