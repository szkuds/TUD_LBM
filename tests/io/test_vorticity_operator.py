"""Tests for the vorticity plot operator, obstacle drawing and the wide-domain layout."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pytest
from src.config import SimulationConfig
from src.registry import get_operator_names
from src.simulation_io.plotting import FigureBuilder
from src.simulation_io.plotting import build_plot_operator
from src.simulation_io.plotting._vorticity import compute_vorticity
from src.simulation_io.plotting.figure_builder import _WIDE_ASPECT

_OMEGA = 0.01


def _velocity(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Pack ``(nx, ny)`` components into the saved 5D ``(nx, ny, 1, 1, 2)`` layout."""
    return np.stack([ux, uy], axis=-1)[:, :, None, None, :]


def _solid_body_rotation(nx: int, ny: int) -> np.ndarray:
    """``u = (-Omega*y, Omega*x)`` about the domain centre; its vorticity is ``2*Omega`` everywhere."""
    x, y = np.meshgrid(np.arange(nx) - nx / 2, np.arange(ny) - ny / 2, indexing="ij")
    return _velocity(-_OMEGA * y, _OMEGA * x)


def _channel_config(
    grid_shape: tuple[int, int, int],
    obstacle_config: dict[str, object] | None = None,
) -> SimulationConfig:
    return SimulationConfig(
        sim_type="single_phase", grid_shape=grid_shape, tau=0.8, nt=10, obstacle_config=obstacle_config
    )


def test_vorticity_of_solid_body_rotation_is_twice_the_angular_velocity() -> None:
    omega = compute_vorticity(_solid_body_rotation(20, 12))

    np.testing.assert_allclose(omega, 2.0 * _OMEGA, rtol=1e-12)


def test_vorticity_sign_follows_the_rotation_sense() -> None:
    clockwise = compute_vorticity(-_solid_body_rotation(20, 12))

    assert np.all(clockwise < 0.0)


def test_vorticity_is_registered_and_opt_in() -> None:
    operator_cls = build_plot_operator("vorticity")

    assert "vorticity" in get_operator_names("plotting")
    assert operator_cls.opt_in is True


def test_default_field_set_leaves_vorticity_out(tmp_path) -> None:
    builder = FigureBuilder(_channel_config((32, 32, 1)), run_dir=tmp_path)

    assert "vorticity" not in [op.name for op in builder.field_operators]


def test_obstacle_cells_are_masked_out_of_both_panels() -> None:
    config = _channel_config((60, 30, 1), obstacle_config={"center_x": 20, "center_y": 15, "radius": 5})
    solid = np.asarray(config.obstacle_mask)[:, :, 0, 0, 0]
    data = {"u": _solid_body_rotation(60, 30)}

    for name in ("vorticity", "velocity"):
        fig, ax = plt.subplots()
        build_plot_operator(name)(config)(ax, data, timestep=0)
        field = ax.images[0].get_array()
        plt.close(fig)

        assert np.ma.isMaskedArray(field), name
        np.testing.assert_array_equal(np.ma.getmaskarray(field), solid.T, err_msg=name)


def test_vorticity_colour_scale_is_symmetric_about_zero() -> None:
    config = _channel_config((40, 20, 1))
    x = np.arange(40)[:, None] * np.ones((1, 20))
    data = {"u": _velocity(np.zeros((40, 20)), 1e-3 * np.sin(2 * np.pi * x / 40))}

    fig, ax = plt.subplots()
    build_plot_operator("vorticity")(config)(ax, data, timestep=0)
    vmin, vmax = ax.images[0].get_clim()
    plt.close(fig)

    assert vmax > 0.0
    assert vmin == pytest.approx(-vmax)


def _render_axes(tmp_path, grid_shape: tuple[int, int, int]) -> list[plt.Axes]:
    nx, ny = grid_shape[:2]
    config = _channel_config(grid_shape)
    builder = FigureBuilder(config, run_dir=tmp_path, fields=["velocity", "vorticity"])
    fig = builder.render_figure({"u": _solid_body_rotation(nx, ny)}, timestep=0)
    assert fig is not None
    image_axes = [ax for ax in fig.axes if ax.images]
    plt.close(fig)
    return image_axes


def test_wide_domain_stacks_field_panels_in_one_column(tmp_path) -> None:
    nx = int(40 * _WIDE_ASPECT) * 2
    top, bottom = _render_axes(tmp_path, (nx, 40, 1))

    assert top.get_position().x0 == pytest.approx(bottom.get_position().x0)
    assert top.get_position().y0 > bottom.get_position().y1


def test_square_domain_keeps_panels_side_by_side(tmp_path) -> None:
    left, right = _render_axes(tmp_path, (40, 40, 1))

    assert left.get_position().x1 < right.get_position().x0
