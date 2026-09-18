"""The ``contact_angle`` overlay: wetting band cells, contact-angle glyphs and plumbing."""

from __future__ import annotations
import math
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from src.config import SimulationConfig
from src.operators.differential._pad_utils import _apply_stencil_padding
from src.operators.differential._pad_utils import determine_pad_modes
from src.operators.wetting._apply_edge import _apply_wetting_edge
from src.operators.wetting._contact_angle import compute_contact_angle
from src.operators.wetting._wetting_modification import anchor_window_half_width
from src.registry import get_operators
from src.simulation_io.analysis.wetting_overlay import WALL_NORMAL
from src.simulation_io.analysis.wetting_overlay import angle_glyph
from src.simulation_io.analysis.wetting_overlay import contact_angles
from src.simulation_io.analysis.wetting_overlay import to_physical
from src.simulation_io.analysis.wetting_overlay import wall_bands
from src.simulation_io.plotting import FigureBuilder

_NX, _NY = 60, 30
_RHO_L, _RHO_V = 1.0, 0.1
_RADIUS = 15.0
#: Circle centre below the wall row, so the drop meets the wall at theta < 90.
_CENTRE_DEPTH = 5.0
_EDGES = ("bottom", "top", "left", "right")


def _config(edge: str | None = "bottom", shape: tuple[int, int] = (_NX, _NY), **overrides: object) -> SimulationConfig:
    params: dict[str, object] = {
        "sim_type": "multiphase",
        "grid_shape": shape,
        "eos": "double-well",
        "kappa": 0.02,
        "interface_width": 4,
        "rho_l": _RHO_L,
        "rho_v": _RHO_V,
        "bc_config": {edge: "wetting"} if edge else None,
    }
    params.update(overrides)
    return SimulationConfig(**params)  # ty: ignore[invalid-argument-type]


def _droplet(edge: str = "bottom", shape: tuple[int, int] = (_NX, _NY)) -> np.ndarray:
    """A tanh sessile droplet on *edge*, centred along the wall."""
    nx, ny = shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    wall_distance = {"bottom": y, "top": ny - 1 - y, "left": x, "right": nx - 1 - x}[edge]
    along = x if edge in ("bottom", "top") else y
    centre_along = (nx if edge in ("bottom", "top") else ny) / 2
    distance = np.hypot(along - centre_along, wall_distance + _CENTRE_DEPTH)
    return _RHO_V + (_RHO_L - _RHO_V) * 0.5 * (1.0 - np.tanh((distance - _RADIUS) / 2.0))


def _shape_for(edge: str) -> tuple[int, int]:
    return (_NX, _NY) if edge in ("bottom", "top") else (_NY, _NX)


def _data(rho_2d: np.ndarray, **extra: float) -> dict[str, np.ndarray]:
    data = {"rho": rho_2d[:, :, None, None, None]}
    data.update({key: np.asarray(value) for key, value in extra.items()})
    return data


# --- wall band -------------------------------------------------------------


def test_band_cells_are_exactly_the_cells_the_applicator_changes():
    rho = _droplet()
    config = _config()
    (band,) = wall_bands(rho, config)

    gp = _apply_stencil_padding(jnp.asarray(rho), tuple(determine_pad_modes(config.bc_config)))

    # A modification that moves every banded cell, so changed == modified region.
    def _apply(d_rho: float) -> jnp.ndarray:
        scalars = (jnp.asarray(v) for v in (_RHO_L, _RHO_V, 1.0, 1.0, d_rho, d_rho))
        return _apply_wetting_edge(gp, "bottom", True, True, *scalars)

    neutral, shifted = _apply(0.0), _apply(0.05)
    changed = np.flatnonzero(np.asarray(neutral[1:-1, 0]) != np.asarray(shifted[1:-1, 0]))

    left, right = band.left, band.right
    np.testing.assert_array_equal(np.union1d(left.cells, right.cells), changed)
    assert left.cells.size > 0
    assert right.cells.size > 0

    # Each side's cells sit in the window around its own contact line, and the
    # two sides are disjoint and ordered along the wall.
    half_width = anchor_window_half_width(_NX)
    assert left.anchor < right.anchor
    assert left.cells.max() < right.cells.min()
    assert np.abs(left.cells - left.anchor).max() <= half_width
    assert np.abs(right.cells - right.anchor).max() <= half_width

    # The bounds a side clips to bracket the densities it actually modifies.
    ghost = np.asarray(neutral[1:-1, 0])
    for side in (left, right):
        assert side.rho_lower < side.rho_upper
        assert ghost[side.cells].min() >= side.rho_lower
        assert ghost[side.cells].max() < side.rho_upper


@pytest.mark.parametrize("edge", _EDGES)
def test_band_is_the_same_on_every_wall(edge):
    shape = _shape_for(edge)
    (reference,) = wall_bands(_droplet(), _config())
    (band,) = wall_bands(_droplet(edge, shape), _config(edge, shape))

    assert band.edge == edge
    np.testing.assert_array_equal(band.left.cells, reference.left.cells)
    np.testing.assert_array_equal(band.right.cells, reference.right.cells)


def test_no_band_without_densities_or_wetting_wall():
    rho = _droplet()

    assert wall_bands(rho, _config(edge=None)) == []
    assert wall_bands(rho, SimulationConfig(grid_shape=(_NX, _NY), bc_config={"bottom": "wetting"})) == []


# --- contact angles --------------------------------------------------------


@pytest.mark.parametrize("edge", _EDGES)
def test_to_physical_places_the_wall_row_on_its_edge(edge):
    shape = _shape_for(edge)
    nx, ny = shape
    x, y = to_physical([0.0, 3.0], [WALL_NORMAL, WALL_NORMAL], edge, shape)
    expected_wall = {"bottom": ("y", -0.5), "top": ("y", ny - 0.5), "left": ("x", -0.5), "right": ("x", nx - 0.5)}[edge]
    coords = {"x": x, "y": y}
    axis, value = expected_wall
    np.testing.assert_allclose(coords[axis], value)
    other = coords["x" if axis == "y" else "y"]
    np.testing.assert_allclose(other, [0.0, 3.0])


def test_recomputed_angles_match_the_step_operator():
    rho = _droplet()
    angles = contact_angles(_data(rho), rho, _config())
    assert angles is not None

    ca_l, ca_r = compute_contact_angle(jnp.asarray(rho)[:, :, None, None, None], 0.5 * (_RHO_L + _RHO_V))
    assert angles.theta_left == pytest.approx(float(ca_l))
    assert angles.theta_right == pytest.approx(float(ca_r))
    assert angles.cll_left < _NX / 2 < angles.cll_right
    assert 0.0 < angles.theta_left < 90.0


@pytest.mark.parametrize("edge", ["top", "left", "right"])
def test_mirrored_walls_give_the_same_angle(edge):
    shape = _shape_for(edge)
    reference = contact_angles(_data(_droplet()), _droplet(), _config())
    rho = _droplet(edge, shape)
    angles = contact_angles(_data(rho), rho, _config(edge, shape))

    assert reference is not None
    assert angles is not None
    assert angles.theta_left == pytest.approx(reference.theta_left)
    assert angles.cll_right == pytest.approx(reference.cll_right)


def test_saved_snapshot_values_are_preferred():
    rho = _droplet()
    data = _data(rho, ca_left=40.0, ca_right=50.0, cll_left=12.0, cll_right=44.0)

    angles = contact_angles(data, rho, _config())

    assert angles is not None
    assert (angles.theta_left, angles.theta_right, angles.cll_left, angles.cll_right) == (40.0, 50.0, 12.0, 44.0)


def test_no_angles_when_the_inclusion_does_not_touch_the_wall():
    detached = np.full((_NX, _NY), _RHO_V)
    detached[20:40, 15:25] = _RHO_L

    assert contact_angles(_data(detached), detached, _config()) is None


def test_glyph_tangent_leans_over_the_dispersed_phase():
    rho = _droplet()
    angles = contact_angles(_data(rho), rho, _config())
    assert angles is not None

    left = angle_glyph(angles, "left", (_NX, _NY), length=10.0)
    right = angle_glyph(angles, "right", (_NX, _NY), length=10.0)

    # theta < 90: both tangents lean toward the droplet centre and rise off the wall.
    assert left.tangent[1, 0] > left.tangent[0, 0]
    assert right.tangent[1, 0] < right.tangent[0, 0]
    expected_rise = 10.0 * math.sin(math.radians(angles.theta_left))
    rise = left.tangent[1, 1] - left.tangent[0, 1]
    assert rise == pytest.approx(expected_rise)
    assert left.arc[0, 1] == pytest.approx(WALL_NORMAL)


# --- operator and FigureBuilder -------------------------------------------


def _line_collection_labels(ax) -> list[str]:
    return [str(c.get_label()) for c in ax.collections if isinstance(c, LineCollection)]


def _panel(fig, title_prefix: str):
    return next(ax for ax in fig.axes if ax.get_title().startswith(title_prefix))


def test_contact_angle_is_an_overlay_only_plotting_operator():
    target = get_operators("plotting")["contact_angle"].target

    assert target.supports_overlay
    assert target.overlay_only
    assert target.opt_in
    assert target.overlay_prompt_default(_config()) is True
    assert target.overlay_prompt_default(_config(edge=None)) is None


def test_both_overlays_draw_on_field_and_interface_panels(tmp_path):
    builder = FigureBuilder(
        _config(),
        run_dir=tmp_path,
        fields=["density", "interface"],
        overlays=["interface", "contact_angle"],
    )
    fig = builder.render_figure(_data(_droplet()), timestep=0)
    assert fig is not None

    density = _line_collection_labels(_panel(fig, "Density"))
    interface = _line_collection_labels(_panel(fig, "Interface"))
    for labels in (density, interface):
        assert any(label.startswith("left ρ_upper") for label in labels)
        assert any(label.startswith("right ρ_upper") for label in labels)
        assert any("wetting band (left)" in label for label in labels)
    # The interface panel draws its own contours once, not again as an overlay.
    assert sum(label.startswith("config ρ=") for label in interface) == 1
    assert sum(label.startswith("config ρ=") for label in density) == 1
    assert any(text.get_text().startswith("θ=") for text in _panel(fig, "Density").texts)
    plt.close(fig)


def test_contact_angle_named_as_a_field_warns_and_adds_no_panel(tmp_path):
    config = _config()

    with pytest.warns(UserWarning, match="overlay-only"):
        builder = FigureBuilder(config, run_dir=tmp_path, fields=["density", "contact_angle"])

    assert [op.name for op in builder.field_operators] == ["density"]


def test_contact_angle_overlay_is_unavailable_without_a_wetting_wall(tmp_path):
    builder = FigureBuilder(_config(edge=None), run_dir=tmp_path, fields=["density"], overlays=["contact_angle"])
    fig = builder.render_figure(_data(_droplet()), timestep=0)
    assert fig is not None

    assert _line_collection_labels(_panel(fig, "Density")) == []
    plt.close(fig)


def test_overlay_labels_are_gathered_into_one_figure_legend(tmp_path):
    builder = FigureBuilder(
        _config(),
        run_dir=tmp_path,
        fields=["density", "velocity", "interface"],
        overlays=["interface", "contact_angle"],
    )
    data = _data(_droplet())
    data["u"] = np.zeros((_NX, _NY, 1, 1, 2))
    fig = builder.render_figure(data, timestep=0)
    assert fig is not None

    assert all(ax.get_legend() is None for ax in fig.axes)
    (legend,) = fig.legends
    labels = [text.get_text() for text in legend.get_texts()]
    assert len(labels) == len(set(labels))
    # "config"/"measured" are the interface contour's markers; "left"/"right"
    # the contact-angle overlay's per-contact-line band bounds.
    assert {label.split()[0] for label in labels} == {"config", "measured", "bottom", "left", "right"}
    assert {label.split()[1].split("=")[0] for label in labels if label.startswith(("left", "right"))} == {
        "ρ_lower",
        "ρ_upper",
    }
    assert any("wetting band (right)" in label for label in labels)
    plt.close(fig)


def test_no_figure_legend_without_labelled_overlays(tmp_path):
    builder = FigureBuilder(_config(), run_dir=tmp_path, fields=["density"], overlays=[])
    fig = builder.render_figure(_data(_droplet()), timestep=0)
    assert fig is not None

    assert fig.legends == []
    plt.close(fig)


def test_band_contours_are_drawn_per_contact_line(tmp_path):
    builder = FigureBuilder(_config(), run_dir=tmp_path, fields=["density"], overlays=["contact_angle"])
    fig = builder.render_figure(_data(_droplet()), timestep=0)
    assert fig is not None

    band_labels = [label for label in _line_collection_labels(_panel(fig, "Density")) if "ρ_" in label]
    assert sorted(label.split()[0] for label in band_labels) == ["left", "left", "right", "right"]
    plt.close(fig)


def test_bulk_drift_below_the_prescribed_liquid_does_not_widen_the_region():
    """A gravitating wall's liquid equilibrates below ``rho_l``; the region must not follow.

    The predecessor keyed membership on the absolute window
    ``[0.05 rho_l + 0.95 rho_v, 0.95 rho_l + 0.05 rho_v)``, leaving only
    ``0.05 (rho_l - rho_v)`` of headroom above it. A measured inclined bubble run
    developed a hydrostatic drop larger than that headroom, bulk liquid entered
    the band, and the modified region grew from 54 to 158 of 200 wall cells.
    """
    rho = _droplet()
    config = _config()
    (reference,) = wall_bands(rho, config)

    # Liquid drawn down by more than the old band's headroom, and tilted along
    # the wall as an inclined domain tilts it.
    old_headroom = 0.05 * (_RHO_L - _RHO_V)
    tilt = np.linspace(0.0, 1.0, _NX)[:, None]
    drifted = np.where(rho > 0.5 * (_RHO_L + _RHO_V), rho - 2.0 * old_headroom * (1.0 + tilt), rho)
    assert drifted[:, 0].max() < 0.95 * _RHO_L + 0.05 * _RHO_V  # the old band would swallow the wall

    (band,) = wall_bands(drifted, config)
    widened = band.left.cells.size + band.right.cells.size
    baseline = reference.left.cells.size + reference.right.cells.size

    assert widened <= 2 * baseline
    assert widened < _NX // 2
    for side in (band.left, band.right):
        assert np.abs(side.cells - side.anchor).max() <= anchor_window_half_width(_NX)
