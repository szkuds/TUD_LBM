"""Extra branch coverage for analysis plot helpers/operators."""

from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pytest
from tud_lbm.io.plotting.analysis import ContactAnglesPairPlot
from tud_lbm.io.plotting.analysis import ContactLineSpeedLeftPlot
from tud_lbm.io.plotting.analysis import ContactLineSpeedsPairPlot
from tud_lbm.io.plotting.analysis import _extract_rho_2d
from tud_lbm.io.plotting.analysis import _extract_u_mag_2d
from tud_lbm.io.plotting.analysis import _load_timesteps
from tud_lbm.io.plotting.analysis import _parse_timestep
from tud_lbm.io.plotting.analysis import _render_scatter

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_timestep_invalid_returns_none():
    assert _parse_timestep("timestep_x") is None


def test_extract_rho_and_u_raise_on_unsupported_ndim():
    with pytest.raises(ValueError, match="Unsupported rho shape"):
        _extract_rho_2d(np.ones((2,)))
    with pytest.raises(ValueError, match="Unsupported u shape"):
        _extract_u_mag_2d(np.ones((2, 2)))


def test_load_timesteps_skips_invalid_names_and_missing_keys(tmp_path: Path):
    np.savez(tmp_path / "nonsense.npz", rho=np.ones((2, 2, 1, 1, 1)))
    np.savez(tmp_path / "timestep_1.npz", rho=np.ones((2, 2, 1, 1, 1)))
    np.savez(tmp_path / "timestep_2.npz", rho=np.ones((2, 2, 1, 1, 1)), u=np.zeros((2, 2, 1, 1, 2)))

    iters, snaps = _load_timesteps(sorted(tmp_path.glob("*.npz")), ("u",))

    assert iters.tolist() == [2]
    assert len(snaps) == 1


def test_render_scatter_empty_state_writes_placeholder():
    fig, ax = plt.subplots()
    try:
        _render_scatter(ax, np.array([]), np.array([]), title="T", ylabel="Y", required_keys=("rho",))
        assert ax.get_title() == "T"
        assert ax.get_xlabel() == "Timestep"
    finally:
        plt.close(fig)


def test_contact_line_speed_left_single_snapshot_returns_zero(tmp_path: Path):
    np.savez(tmp_path / "timestep_5.npz", cll_left=np.array(3.0))

    result = ContactLineSpeedLeftPlot().compute(sorted(tmp_path.glob("*.npz")))

    assert result["iters"].tolist() == [5]
    assert result["values"].tolist() == [0.0]


def test_pair_renderers_cover_empty_and_non_empty_states(tmp_path: Path):
    pair = ContactAnglesPairPlot()
    speeds = ContactLineSpeedsPairPlot()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    try:
        pair.render(ax1, {"iters": np.array([]), "left": np.array([]), "right": np.array([])})
        speeds.render(ax2, {"iters": np.array([]), "left": np.array([]), "right": np.array([])})

        np.savez(
            tmp_path / "timestep_1.npz",
            ca_left=np.array(80.0),
            ca_right=np.array(90.0),
            cll_left=np.array(1.0),
            cll_right=np.array(2.0),
        )
        np.savez(
            tmp_path / "timestep_3.npz",
            ca_left=np.array(81.0),
            ca_right=np.array(89.0),
            cll_left=np.array(2.0),
            cll_right=np.array(4.0),
        )

        pre_pair = pair.compute(sorted(tmp_path.glob("*.npz")))
        pair.render(ax1, pre_pair)
        pre_speed = speeds.compute(sorted(tmp_path.glob("*.npz")))
        speeds.render(ax2, pre_speed)

        assert ax1.get_title() == "Contact angles vs timestep"
        assert ax2.get_title() == "Contact-line speeds vs timestep"
    finally:
        plt.close(fig)
