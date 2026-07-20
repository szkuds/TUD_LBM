"""Contact-line speed history analysis plots."""

from __future__ import annotations
from src.registry import analysis_operator
from src.simulation_io.plotting._analysis_common import _CONTACT_LINE_SPEED_Y_LABEL
from src.simulation_io.plotting._analysis_common import _PairSeriesPlot
from src.simulation_io.plotting._analysis_common import _SeriesAttrPlot
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

_CONTACT_LINE_SPEEDS_TITLE = "Contact-line speeds vs timestep"


@analysis_operator(name="contact_line_speed_left")
class ContactLineSpeedLeftPlot(_SeriesAttrPlot):
    """Plot left contact-line speed over time."""

    name = "contact_line_speed_left"
    title = "Left contact-line speed vs timestep"
    ylabel = "d(cll_left)/dt"
    color = DEFAULT_STYLE.colors["contact_line_speed_left"]
    series_attr = "v_left"
    required_keys = ("cll_left",)


@analysis_operator(name="contact_line_speed_right")
class ContactLineSpeedRightPlot(_SeriesAttrPlot):
    """Plot right contact-line speed over time."""

    name = "contact_line_speed_right"
    title = "Right contact-line speed vs timestep"
    ylabel = "d(cll_right)/dt"
    color = DEFAULT_STYLE.colors["contact_line_speed_right"]
    series_attr = "v_right"
    required_keys = ("cll_right",)


@analysis_operator(name="contact_line_speeds_pair")
class ContactLineSpeedsPairPlot(_PairSeriesPlot):
    """Render paired left/right contact-line speed history."""

    name = "contact_line_speeds_pair"
    title = _CONTACT_LINE_SPEEDS_TITLE
    ylabel = _CONTACT_LINE_SPEED_Y_LABEL
    left_attr = "v_left"
    right_attr = "v_right"
    left_color = DEFAULT_STYLE.colors["contact_line_speed_left"]
    right_color = DEFAULT_STYLE.colors["contact_line_speed_right"]
    legend_kwargs = {"loc": "best", "fontsize": DEFAULT_STYLE.pair_legend_fontsize}  # noqa: RUF012
    required_keys = ("cll_left", "cll_right")
