"""Tests for the circular obstacle mask builder."""

import pytest
from tud_lbm.operators.obstacle import build_obstacle_mask
from tud_lbm.operators.obstacle._circle import build_circle_mask


def test_shape() -> None:
    mask = build_circle_mask({"center_x": 10, "center_y": 10, "radius": 5}, (20, 20, 1))
    assert mask.shape == (20, 20, 1, 1, 1)
    assert mask.dtype == bool


def test_known_inside_outside_boundary_cells() -> None:
    mask = build_circle_mask({"center_x": 10, "center_y": 10, "radius": 5}, (20, 20, 1))
    assert bool(mask[10, 10, 0, 0, 0])  # center: inside
    assert not bool(mask[0, 0, 0, 0, 0])  # far corner: outside
    assert bool(mask[15, 10, 0, 0, 0])  # exactly distance r away: boundary, inclusive (<=)
    assert not bool(mask[16, 10, 0, 0, 0])  # just past the boundary: outside


def test_raises_on_3d_grid() -> None:
    with pytest.raises(ValueError, match="2D"):
        build_circle_mask({"center_x": 10, "center_y": 10, "radius": 5}, (20, 20, 2))


def test_build_obstacle_mask_dispatches_to_circle() -> None:
    mask = build_obstacle_mask({"shape": "circle", "center_x": 10, "center_y": 10, "radius": 5}, (20, 20, 1))
    assert mask is not None
    assert mask.shape == (20, 20, 1, 1, 1)


def test_build_obstacle_mask_defaults_to_circle_shape() -> None:
    mask = build_obstacle_mask({"center_x": 10, "center_y": 10, "radius": 5}, (20, 20, 1))
    assert mask is not None


def test_build_obstacle_mask_none_config_returns_none() -> None:
    assert build_obstacle_mask(None, (20, 20, 1)) is None


def test_build_obstacle_mask_unknown_shape_raises() -> None:
    with pytest.raises(ValueError, match="Unknown obstacle shape"):
        build_obstacle_mask({"shape": "square", "center_x": 10, "center_y": 10, "radius": 5}, (20, 20, 1))
