"""The named edges, contact lines and option values, pinned to what they are written as.

Each value is part of something already on disk or in the solver: a contact
line's value is the suffix of saved keys (``ca_left``), an edge's value is a
``[boundary_conditions]`` key, and an option value is what the CLI accepts.
"""

from __future__ import annotations
import pytest
from src.config.analysis_options import AxisScale
from src.config.analysis_options import Smoothing
from src.config.boundary import BC_APPLICATION_ORDER
from src.config.boundary import IN_PLANE_EDGES
from src.config.boundary import OUT_OF_PLANE_EDGES
from src.config.boundary import Edge
from src.config.wetting import ContactLine
from src.operators.boundary._faces import face_axis
from src.operators.boundary._faces import face_cells
from src.operators.wetting._params import WettingState
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE


def test_contact_line_values_are_the_saved_key_suffixes():
    assert [line.value for line in ContactLine] == ["left", "right"]
    for line in ContactLine:
        for prefix in ("phi", "d_rho", "ca", "cll"):
            assert f"{prefix}_{line}" in WettingState._fields


def test_contact_line_other_is_the_opposite_line():
    assert ContactLine.LEFT.other is ContactLine.RIGHT
    assert ContactLine.RIGHT.other is ContactLine.LEFT


def test_contact_line_style_tables_name_every_line():
    """``figure_config`` must stay stdlib-only, so it keys on the values; this keeps them in step."""
    assert set(DEFAULT_STYLE.wetting_band_linestyles) == set(ContactLine)
    assert set(DEFAULT_STYLE.wetting_region_colors) == set(ContactLine)


def test_edges_are_declared_in_application_order():
    assert tuple(Edge) == BC_APPLICATION_ORDER
    assert [edge.value for edge in BC_APPLICATION_ORDER] == ["bottom", "top", "left", "right", "front", "back"]
    assert set(IN_PLANE_EDGES) | set(OUT_OF_PLANE_EDGES) == set(Edge)


@pytest.mark.parametrize(
    ("edge", "axis", "far"),
    [
        (Edge.BOTTOM, 1, False),
        (Edge.TOP, 1, True),
        (Edge.LEFT, 0, False),
        (Edge.RIGHT, 0, True),
        (Edge.BACK, 2, False),
        (Edge.FRONT, 2, True),
    ],
)
def test_edge_orientation(edge: Edge, axis: int, far: bool):
    assert edge.normal_axis == axis
    assert edge.at_far_end is far
    assert face_axis(edge) == axis
    assert face_cells(edge)[axis] == (-1 if far else 0)


def test_tangential_ends_bound_the_wall():
    assert Edge.BOTTOM.tangential_ends == (Edge.LEFT, Edge.RIGHT)
    assert Edge.TOP.tangential_ends == (Edge.LEFT, Edge.RIGHT)
    assert Edge.LEFT.tangential_ends == (Edge.BOTTOM, Edge.TOP)
    assert Edge.RIGHT.tangential_ends == (Edge.BOTTOM, Edge.TOP)


def test_option_values_are_the_cli_spellings():
    assert [choice.value for choice in Smoothing] == ["raw", "savgol"]
    assert [choice.value for choice in AxisScale] == ["linear", "log"]
