"""Extra branch coverage for analysis plot helpers/operators."""

from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pytest
from tests.support.run_dirs import build_run_dir
from tests.support.run_dirs import wetting_config
from tud_lbm.config import SimulationConfig
from tud_lbm.io.analysis.droplet_metrics import analytical_sigma_lg
from tud_lbm.io.analysis.droplet_metrics import resolve_step_x
from tud_lbm.io.plotting._analysis_common import _empty_data_message
from tud_lbm.io.plotting._analysis_common import _extract_rho_2d
from tud_lbm.io.plotting._analysis_common import _extract_u_mag_2d
from tud_lbm.io.plotting._analysis_common import _load_timesteps
from tud_lbm.io.plotting._analysis_common import _parse_timestep
from tud_lbm.io.plotting._analysis_common import _render_scatter
from tud_lbm.io.plotting.contact_angle_plot import ContactAnglesPairPlot
from tud_lbm.io.plotting.contact_line_speed_plot import ContactLineSpeedLeftPlot
from tud_lbm.io.plotting.contact_line_speed_plot import ContactLineSpeedsPairPlot
from tud_lbm.io.plotting.scalar_history_plot import DensityRatioPlot

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_timestep_invalid_returns_none():
    assert _parse_timestep("timestep_x") is None


def test_extract_rho_and_u_raise_on_unsupported_ndim():
    bad_rho = np.ones((2,))
    bad_u = np.ones((2, 2))
    with pytest.raises(ValueError, match="Unsupported rho shape"):
        _extract_rho_2d(bad_rho)
    with pytest.raises(ValueError, match="Unsupported u shape"):
        _extract_u_mag_2d(bad_u)


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
    """One snapshot has no predecessor, so its speed is 0.0 rather than undefined."""
    config = wetting_config()
    run_dir = build_run_dir(tmp_path, iterations=(5,), config=config)
    files = sorted((run_dir / "data").glob("timestep_*.npz"))

    result = ContactLineSpeedLeftPlot(config=config).compute(files)

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


def test_extract_rho_2d_covers_supported_shapes():
    rho2 = np.ones((2, 3))
    rho3 = np.ones((2, 3, 1))
    rho4 = np.ones((2, 3, 1, 1))
    rho5 = np.ones((2, 3, 1, 1, 1))

    assert _extract_rho_2d(rho2).shape == (2, 3)
    assert _extract_rho_2d(rho3).shape == (2, 3)
    assert _extract_rho_2d(rho4).shape == (2, 3)
    assert _extract_rho_2d(rho5).shape == (2, 3)


def test_extract_u_mag_2d_covers_supported_shapes():
    u5 = np.zeros((2, 3, 1, 1, 2))
    u5[:, :, 0, 0, 0] = 3.0
    u5[:, :, 0, 0, 1] = 4.0

    u4 = np.zeros((2, 3, 1, 2))
    u4[:, :, 0, 0] = 5.0
    u4[:, :, 0, 1] = 12.0

    u3 = np.zeros((2, 3, 2))
    u3[:, :, 0] = 8.0
    u3[:, :, 1] = 15.0

    assert np.allclose(_extract_u_mag_2d(u5), 5.0)
    assert np.allclose(_extract_u_mag_2d(u4), 13.0)
    assert np.allclose(_extract_u_mag_2d(u3), 17.0)


def test_empty_message_and_config_resolvers_cover_branches():
    msg = _empty_data_message(("rho", "u"))
    assert "No data" in msg
    assert "Requires" in msg

    cfg = SimulationConfig(
        sim_type="multiphase",
        grid_shape=(20, 10, 1),
        tau=0.9,
        nt=2,
        eos="double-well",
        kappa=0.12,
        interface_width=4,
        rho_l=1.0,
        rho_v=0.2,
        initialisation={"radii": [0.5]},
        chemical_step_config={"chemical_step_location": 0.25},
    )
    assert analytical_sigma_lg(cfg) is not None
    assert resolve_step_x(cfg) == 5.0

    cfg_no_step = SimulationConfig(grid_shape=(8, 8, 1), tau=0.8, nt=2, chemical_step_config={})
    assert resolve_step_x(cfg_no_step) is None


def test_density_ratio_render_uses_log_scale():
    fig, ax = plt.subplots()
    try:
        plot = DensityRatioPlot()
        plot.render(ax, {"iters": np.array([1, 2]), "values": np.array([2.0, 4.0])})
        assert ax.get_yscale() == "log"
        assert ax.get_title() == "Density ratio vs timestep"
    finally:
        plt.close(fig)
