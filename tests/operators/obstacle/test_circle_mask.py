"""Tests for the circular obstacle mask builder."""

import pytest
from src.config import SimulationConfig
from src.config.obstacle import obstacle_mask
from src.operators.obstacle._circle import build_circle_mask


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
    config = SimulationConfig(
        grid_shape=(20, 20), obstacle={"shape": "circle", "center_x": 10, "center_y": 10, "radius": 5}
    )
    mask = obstacle_mask(config)
    assert mask is not None
    assert mask.shape == (20, 20, 1, 1, 1)


def test_build_obstacle_mask_none_config_returns_none() -> None:
    assert obstacle_mask(SimulationConfig(grid_shape=(20, 20))) is None
