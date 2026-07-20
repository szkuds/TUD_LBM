"""Tests for tud_lbm.io.plotting.ca_theta_plot."""

from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib as mpl
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

mpl.use("Agg")

from tud_lbm.io.plotting.ca_theta_plot import plot_contact_angle_vs_capillary_number
from tud_lbm.io.plotting.ca_theta_plot import save_figure

_CA_TRAILING = np.array([1e-4, 5e-4, 1e-3])
_THETA_TRAILING = np.array([100.0, 105.0, 110.0])
_CA_LEADING = np.array([1e-4, 5e-4, 1e-3])
_THETA_LEADING = np.array([80.0, 75.0, 70.0])


def _make_figure():
    return plot_contact_angle_vs_capillary_number(_CA_TRAILING, _THETA_TRAILING, _CA_LEADING, _THETA_LEADING)


def test_returns_figure():
    import matplotlib.figure

    fig = _make_figure()
    assert isinstance(fig, matplotlib.figure.Figure)


def test_has_two_scatter_collections():
    fig = _make_figure()
    ax = fig.axes[0]
    collections = ax.collections
    assert len(collections) == 2


def test_axis_labels():
    fig = _make_figure()
    ax = fig.axes[0]
    assert "Ca" in ax.get_xlabel()
    assert r"theta" in ax.get_ylabel() or "degrees" in ax.get_ylabel()


def test_log_scale_sets_xscale():
    fig = plot_contact_angle_vs_capillary_number(
        _CA_TRAILING, _THETA_TRAILING, _CA_LEADING, _THETA_LEADING, log_scale=True
    )
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"


def test_linear_scale_is_default():
    fig = _make_figure()
    ax = fig.axes[0]
    assert ax.get_xscale() == "linear"


def test_title_is_set_when_provided():
    fig = plot_contact_angle_vs_capillary_number(
        _CA_TRAILING,
        _THETA_TRAILING,
        _CA_LEADING,
        _THETA_LEADING,
        title="My custom title",
    )
    ax = fig.axes[0]
    assert ax.get_title() == "My custom title"


def test_no_title_by_default():
    fig = _make_figure()
    ax = fig.axes[0]
    assert ax.get_title() == ""


def _write_reference_csv(path: Path) -> Path:
    csv = path / "ref.csv"
    csv.write_text(
        "Ca_trailing,theta_trailing,Ca_leading,theta_leading\n1e-4,100.0,1e-4,80.0\n1e-3,110.0,1e-3,70.0\n",
        encoding="utf-8",
    )
    return csv


def test_reference_csv_adds_two_more_collections(tmp_path):
    csv = _write_reference_csv(tmp_path)
    fig = plot_contact_angle_vs_capillary_number(
        _CA_TRAILING,
        _THETA_TRAILING,
        _CA_LEADING,
        _THETA_LEADING,
        reference_csv=csv,
    )
    ax = fig.axes[0]
    assert len(ax.collections) == 4


def test_reference_csv_missing_column_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("Ca_trailing,theta_trailing\n1e-4,100.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        plot_contact_angle_vs_capillary_number(
            _CA_TRAILING,
            _THETA_TRAILING,
            _CA_LEADING,
            _THETA_LEADING,
            reference_csv=bad_csv,
        )


def test_save_figure_png(tmp_path):
    fig = _make_figure()
    out = tmp_path / "out.png"
    save_figure(fig, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_figure_pdf(tmp_path):
    fig = _make_figure()
    out = tmp_path / "out.pdf"
    save_figure(fig, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_empty_arrays_do_not_raise():
    fig = plot_contact_angle_vs_capillary_number(np.array([]), np.array([]), np.array([]), np.array([]))
    import matplotlib.figure

    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# plot_dual_axis_ca_theta standalone function tests
# ---------------------------------------------------------------------------

_X = np.linspace(0.0, 3.0, 5)
_CA_TR = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4])
_CA_LE = np.array([0.8e-4, 1.6e-4, 2.4e-4, 3.2e-4, 4.0e-4])
_TH_TR = np.array([95.0, 98.0, 101.0, 104.0, 107.0])
_TH_LE = np.array([85.0, 82.0, 79.0, 76.0, 73.0])


def test_plot_dual_axis_ca_theta_returns_figure():
    import matplotlib.figure
    from tud_lbm.io.plotting import plot_dual_axis_ca_theta

    fig = plot_dual_axis_ca_theta(_X, _CA_TR, _CA_LE, _TH_TR, _TH_LE)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_dual_axis_has_two_axes():
    from tud_lbm.io.plotting import plot_dual_axis_ca_theta

    fig = plot_dual_axis_ca_theta(_X, _CA_TR, _CA_LE, _TH_TR, _TH_LE)
    assert len(fig.axes) == 2


def test_dual_axis_axis_labels():
    from tud_lbm.io.plotting import plot_dual_axis_ca_theta

    fig = plot_dual_axis_ca_theta(_X, _CA_TR, _CA_LE, _TH_TR, _TH_LE, x_label="x-label")
    ax1, ax2 = fig.axes
    assert "Ca" in ax1.get_ylabel() or "ca" in ax1.get_ylabel().lower()
    assert "degrees" in ax2.get_ylabel() or "theta" in ax2.get_ylabel().lower()


def test_dual_axis_four_scatter_collections():
    from tud_lbm.io.plotting import plot_dual_axis_ca_theta

    fig = plot_dual_axis_ca_theta(_X, _CA_TR, _CA_LE, _TH_TR, _TH_LE)
    total_collections = sum(len(ax.collections) for ax in fig.axes)
    assert total_collections == 4


# ---------------------------------------------------------------------------
# CaThetaVsTimePlot and CaThetaVsXPlot operator tests
# ---------------------------------------------------------------------------


def _wetting_config():
    from tud_lbm.config import SimulationConfig

    return SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(16, 12),
        tau=0.9,
        nt=20,
        save_interval=5,
        eos="double-well",
        kappa=0.02,
        interface_width=2,
        rho_l=1.0,
        rho_v=0.2,
        wetting_config={"advancing_ca": 100.0},
        initialisation={"radii": [0.25], "centres": [[0.5, 0.5]]},
    )


def _write_ca_snapshot(
    path: Path, step: int, ca_left: float, ca_right: float, cll_left: float, cll_right: float
) -> None:
    rho = np.full((16, 12, 1, 1, 1), 0.2)
    rho[4:12, 1:8, 0, 0, 0] = 1.0
    u = np.zeros((16, 12, 1, 1, 2))
    np.savez(
        path / f"timestep_{step}.npz",
        rho=rho,
        u=u,
        ca_left=np.array(ca_left),
        ca_right=np.array(ca_right),
        cll_left=np.array(cll_left),
        cll_right=np.array(cll_right),
    )


def test_ca_theta_vs_time_operator_compute(tmp_path: Path):
    from tud_lbm.io.plotting.ca_theta_plot import CaThetaVsTimePlot

    _write_ca_snapshot(tmp_path, 5, ca_left=85.0, ca_right=95.0, cll_left=3.0, cll_right=10.0)
    _write_ca_snapshot(tmp_path, 10, ca_left=86.0, ca_right=96.0, cll_left=3.5, cll_right=10.5)

    op = CaThetaVsTimePlot(config=_wetting_config())
    result = op.compute(sorted(tmp_path.glob("timestep_*.npz")))

    assert set(result.keys()) == {
        "theta_trailing",
        "theta_leading",
        "ca_trailing",
        "ca_leading",
        "x_time",
        "x_pos",
        "timesteps",
    }
    assert len(result["theta_trailing"]) == 2
    assert len(result["x_time"]) == 2
    # Both x-axis variants come from one shared series, so x_pos is populated
    # here too even though this operator renders against x_time.
    assert len(result["x_pos"]) == 2
    assert list(result["timesteps"]) == [5, 10]


def test_ca_theta_vs_time_operator_render(tmp_path: Path):
    import matplotlib.pyplot as plt
    from tud_lbm.io.plotting.ca_theta_plot import CaThetaVsTimePlot

    _write_ca_snapshot(tmp_path, 5, ca_left=85.0, ca_right=95.0, cll_left=3.0, cll_right=10.0)
    _write_ca_snapshot(tmp_path, 10, ca_left=86.0, ca_right=96.0, cll_left=3.5, cll_right=10.5)

    op = CaThetaVsTimePlot(config=_wetting_config())
    precomputed = op.compute(sorted(tmp_path.glob("timestep_*.npz")))

    fig, ax = plt.subplots()
    op.render(ax, precomputed)

    total = sum(len(a.collections) for a in fig.axes)
    assert total == 4
    plt.close(fig)


def test_ca_theta_vs_x_operator_compute(tmp_path: Path):
    from tud_lbm.io.plotting.ca_theta_plot import CaThetaVsXPlot

    _write_ca_snapshot(tmp_path, 5, ca_left=85.0, ca_right=95.0, cll_left=3.0, cll_right=10.0)
    _write_ca_snapshot(tmp_path, 10, ca_left=86.0, ca_right=96.0, cll_left=3.5, cll_right=10.5)

    op = CaThetaVsXPlot(config=_wetting_config())
    result = op.compute(sorted(tmp_path.glob("timestep_*.npz")))

    assert len(result["x_pos"]) == 2


def test_ca_theta_vs_time_operator_no_config(tmp_path: Path):
    import matplotlib.pyplot as plt
    from tud_lbm.io.plotting.ca_theta_plot import CaThetaVsTimePlot

    _write_ca_snapshot(tmp_path, 5, ca_left=85.0, ca_right=95.0, cll_left=3.0, cll_right=10.0)

    op = CaThetaVsTimePlot(config=None)
    precomputed = op.compute(sorted(tmp_path.glob("timestep_*.npz")))
    assert len(precomputed["x_time"]) == 0

    fig, ax = plt.subplots()
    op.render(ax, precomputed)
    assert "No data" in ax.texts[0].get_text()
    plt.close(fig)
