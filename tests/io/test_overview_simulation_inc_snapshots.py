"""Tests for tud_lbm.io.plotting.overview_simulation_inc_snapshots."""

from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib as mpl
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

mpl.use("Agg")

from tud_lbm.config import SimulationConfig
from tud_lbm.io.plotting.overview_simulation_inc_snapshots import SnapshotOverviewPlot
from tud_lbm.registry import get_operators


def _wetting_config() -> SimulationConfig:
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


def _write_snapshot(path: Path, step: int, ca_left: float, ca_right: float, cll_left: float, cll_right: float) -> None:
    rho = np.full((16, 12, 1, 1, 1), 0.2)
    # Plateau value encodes the step so tests can confirm which file's data got rendered.
    rho[4:12, 1:8, 0, 0, 0] = 1.0 + step * 0.01
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


def _write_snapshots(path: Path) -> list[Path]:
    _write_snapshot(path, 0, ca_left=85.0, ca_right=95.0, cll_left=3.0, cll_right=10.0)
    _write_snapshot(path, 5, ca_left=86.0, ca_right=96.0, cll_left=3.5, cll_right=10.5)
    _write_snapshot(path, 10, ca_left=87.0, ca_right=97.0, cll_left=4.0, cll_right=11.0)
    _write_snapshot(path, 15, ca_left=88.0, ca_right=98.0, cll_left=4.5, cll_right=11.5)
    return sorted(path.glob("timestep_*.npz"))


def test_registered_under_snapshot_fig():
    entry = get_operators("analysis").get("snapshot_fig")
    assert entry is not None
    assert entry.target is SnapshotOverviewPlot


def test_is_multi_panel_flag():
    assert SnapshotOverviewPlot.is_multi_panel is True


def test_render_figure_without_timesteps_is_empty_state(tmp_path: Path):
    import matplotlib.figure

    files = _write_snapshots(tmp_path)
    op = SnapshotOverviewPlot(config=_wetting_config())

    fig = op.render_figure(files)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1
    assert "No data" in fig.axes[0].texts[0].get_text()


def test_render_figure_without_config_is_empty_state(tmp_path: Path):
    files = _write_snapshots(tmp_path)
    op = SnapshotOverviewPlot(config=None)
    op.timesteps = [5, 10]

    fig = op.render_figure(files)

    assert len(fig.axes) == 1


def _density_plateau_by_letter(fig) -> dict[str, float]:
    """Map each block letter to the rendered density panel's plateau value."""
    plateaus: dict[str, float] = {}
    for ax in fig.axes[1:]:
        label = ax.get_label()
        if label and ax.images:
            plateaus[label] = float(ax.images[0].get_array().max())
    return plateaus


def test_render_figure_builds_lettered_panels(tmp_path: Path):
    files = _write_snapshots(tmp_path)
    op = SnapshotOverviewPlot(config=_wetting_config())
    op.timesteps = [0, 10]

    fig = op.render_figure(files)

    # One label per snapshot block (covering both density and velocity panels), lettered A and B.
    labels = [ax.get_label() for ax in fig.axes]
    assert labels.count("A") == 1
    assert labels.count("B") == 1
    letter_texts = {t.get_text() for t in fig.texts if t.get_text() in ("A", "B")}
    assert letter_texts == {"A", "B"}


def test_render_figure_letters_follow_ascending_timestep_order(tmp_path: Path):
    files = _write_snapshots(tmp_path)
    op = SnapshotOverviewPlot(config=_wetting_config())
    # Requested out of order: letters should still follow ascending timestep.
    op.timesteps = [10, 0]

    fig = op.render_figure(files)

    plateaus = _density_plateau_by_letter(fig)
    assert plateaus["A"] == pytest.approx(1.0)  # step 0
    assert plateaus["B"] == pytest.approx(1.10)  # step 10


def test_render_figure_nearest_timestep_match(tmp_path: Path):
    files = _write_snapshots(tmp_path)
    op = SnapshotOverviewPlot(config=_wetting_config())
    # 4 isn't a saved timestep; nearest saved value is 5.
    op.timesteps = [4]

    fig = op.render_figure(files)

    plateaus = _density_plateau_by_letter(fig)
    assert plateaus["A"] == pytest.approx(1.05)  # step 5


def test_render_figure_draws_vertical_lines(tmp_path: Path):
    files = _write_snapshots(tmp_path)
    op = SnapshotOverviewPlot(config=_wetting_config())
    op.timesteps = [0, 5, 10]

    fig = op.render_figure(files)
    ax_main = fig.axes[0]
    assert len(ax_main.lines) == 3


def test_too_many_timesteps_raises(tmp_path: Path):
    files = _write_snapshots(tmp_path)
    op = SnapshotOverviewPlot(config=_wetting_config())
    op.timesteps = list(range(27))

    with pytest.raises(ValueError, match="at most 26"):
        op.render_figure(files)


def test_render_is_placeholder_only():
    import matplotlib.pyplot as plt

    op = SnapshotOverviewPlot(config=_wetting_config())
    fig, ax = plt.subplots()
    op.render(ax, op.compute([]))
    assert "No data" in ax.texts[0].get_text()
    plt.close(fig)
