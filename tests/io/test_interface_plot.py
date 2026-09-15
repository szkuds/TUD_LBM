"""The ``interface`` plot operator as overlay and panel, and the evolution figures."""

from __future__ import annotations
from itertools import pairwise
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from src.config import SimulationConfig
from src.registry import get_operators
from src.simulation_io.plotting import FigureBuilder
from src.simulation_io.plotting.animator import Animator

if TYPE_CHECKING:
    from pathlib import Path

_NX, _NY = 40, 30
_STEPS = (0, 10, 20)


def _config(**overrides: object) -> SimulationConfig:
    params: dict[str, object] = {
        "sim_type": "multiphase",
        "grid_shape": (_NX, _NY),
        "eos": "double-well",
        "kappa": 0.02,
        "interface_width": 4,
        "rho_l": 1.0,
        "rho_v": 0.1,
    }
    params.update(overrides)
    return SimulationConfig(**params)  # ty: ignore[invalid-argument-type]


def _snapshot(shift: int = 0) -> dict[str, np.ndarray]:
    x, y = np.meshgrid(np.arange(_NX), np.arange(_NY), indexing="ij")
    distance = np.hypot(x - 15 - shift, y - 15)
    rho = 0.1 + 0.9 * 0.5 * (1.0 - np.tanh((distance - 7.0) / 2.0))
    u = np.zeros((_NX, _NY, 1, 1, 2))
    u[..., 0, 0, 0] = 0.01
    return {"rho": rho[:, :, None, None, None], "u": u}


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    for index, step in enumerate(_STEPS):
        np.savez(data_dir / f"timestep_{step}.npz", **_snapshot(shift=3 * index))  # ty: ignore[invalid-argument-type]
    return run_dir


def _collections(ax) -> list[LineCollection]:
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def _panel(fig, title_prefix: str):
    return next(ax for ax in fig.axes if ax.get_title().startswith(title_prefix))


def test_interface_is_an_opt_in_overlay_capable_plotting_operator():
    target = get_operators("plotting")["interface"].target

    assert target.supports_overlay
    assert target.opt_in
    assert not target.accepts_overlays


def test_interface_is_not_in_the_default_figure(tmp_path):
    builder = FigureBuilder(_config(), run_dir=tmp_path)

    assert "interface" not in {op.name for op in builder.field_operators}
    assert builder.overlay_operators == []


def test_overlay_draws_one_contour_per_level_on_every_field_panel(tmp_path):
    builder = FigureBuilder(_config(), run_dir=tmp_path, fields=["density", "velocity"], overlays=["interface"])
    fig = builder.render_figure(_snapshot(), timestep=0)
    assert fig is not None

    for prefix in ("Density", "Velocity"):
        ax = _panel(fig, prefix)
        styles = {str(c.get_label()).split()[0] for c in _collections(ax)}
        assert styles == {"config", "measured"}, prefix
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        assert xlim == pytest.approx((-0.5, _NX - 0.5))
        assert ylim == pytest.approx((-0.5, _NY - 0.5))
    plt.close(fig)


def test_overlay_fields_are_read_from_config(tmp_path):
    config = _config(overlay_fields=["interface"], interface_levels=["config"])
    builder = FigureBuilder(config, run_dir=tmp_path, fields=["density"])
    fig = builder.render_figure(_snapshot(), timestep=0)
    assert fig is not None

    labels = [str(c.get_label()) for c in _collections(_panel(fig, "Density"))]
    assert labels == ["config ρ=0.55"]
    plt.close(fig)


def test_standalone_interface_panel_is_not_overlaid_twice(tmp_path):
    builder = FigureBuilder(_config(), run_dir=tmp_path, fields=["interface"], overlays=["interface"])
    fig = builder.render_figure(_snapshot(), timestep=5)
    assert fig is not None

    ax = _panel(fig, "Interface")
    assert ax.get_title() == "Interface  t=5"
    assert len(_collections(ax)) == 2
    plt.close(fig)


def test_non_overlay_operator_is_rejected_as_overlay(tmp_path):
    with pytest.warns(UserWarning, match="overlay-capable") as record:
        builder = FigureBuilder(_config(), run_dir=tmp_path, overlays=["density", "nonexistent"])

    messages = [str(warning.message) for warning in record]
    assert builder.overlay_operators == []
    assert any("'density'" in m and "['interface']" in m for m in messages)
    assert any("'nonexistent'" in m for m in messages)


def test_unknown_interface_level_fails_when_the_builder_is_constructed(tmp_path):
    with pytest.raises(ValueError, match="bogus"):
        FigureBuilder(_config(interface_levels=["bogus"]), run_dir=tmp_path, overlays=["interface"])


def test_single_phase_snapshot_draws_only_the_measured_contour(tmp_path):
    config = SimulationConfig(grid_shape=(_NX, _NY))
    builder = FigureBuilder(config, run_dir=tmp_path, fields=["density"], overlays=["interface"])
    fig = builder.render_figure(_snapshot(), timestep=0)
    assert fig is not None

    labels = [str(c.get_label()).split()[0] for c in _collections(_panel(fig, "Density"))]
    assert labels == ["measured"]
    plt.close(fig)


def test_evolution_figures_are_written_one_per_level(tmp_path):
    run_dir = _run_dir(tmp_path)
    fields = ["interface_evolution_config", "interface_evolution_measured"]
    builder = FigureBuilder(_config(), run_dir=run_dir, fields=fields)

    saved = builder.build_analysis()

    assert sorted(path.name for path in saved) == [f"{name}.png" for name in fields]


def test_evolution_compute_keeps_one_closed_contour_per_snapshot(tmp_path):
    run_dir = _run_dir(tmp_path)
    operator = get_operators("analysis")["interface_evolution_config"].target(config=_config())
    files = sorted((run_dir / "data").glob("*.npz"), key=lambda p: int(p.stem.split("_")[1]))

    result = operator.compute(files)

    np.testing.assert_array_equal(result["timesteps"], _STEPS)
    np.testing.assert_allclose(result["levels"], 0.55)
    np.testing.assert_array_equal(result["line_snapshot"], [0, 1, 2])
    assert result["line_offsets"][-1] == len(result["vertices"])
    # The droplet moves +3 cells in x per snapshot.
    centres = [result["vertices"][start:stop, 0].mean() for start, stop in pairwise(result["line_offsets"])]
    np.testing.assert_allclose(np.diff(centres), 3.0, atol=0.05)


def test_evolution_render_colours_contours_by_timestep(tmp_path):
    run_dir = _run_dir(tmp_path)
    operator = get_operators("analysis")["interface_evolution_measured"].target(config=_config())
    fig, ax = plt.subplots()

    operator.render(ax, operator.compute(sorted((run_dir / "data").glob("*.npz"))))

    (collection,) = _collections(ax)
    assert set(np.asarray(collection.get_array()).tolist()) == set(_STEPS)
    assert ax.get_title().startswith("Interface evolution (measured level)")
    plt.close(fig)


def test_evolution_render_without_snapshots_shows_empty_state():
    operator = get_operators("analysis")["interface_evolution_config"].target(config=_config())
    fig, ax = plt.subplots()

    operator.render(ax, operator.compute([]))

    assert _collections(ax) == []
    assert any("No data" in text.get_text() for text in ax.texts)
    plt.close(fig)


def test_animator_frames_carry_the_overlay(tmp_path):
    run_dir = _run_dir(tmp_path)
    animator = Animator(config=_config(), run_dir=run_dir, fields=["density"], overlays=["interface"])

    frames = animator.build_frames()

    assert len(frames) == len(_STEPS)
    assert [op.name for op in animator.builder.overlay_operators] == ["interface"]


def test_operator_without_overlay_support_refuses_draw_overlay():
    density = get_operators("plotting")["density"].target(_config())
    fig, ax = plt.subplots()

    with pytest.raises(NotImplementedError, match="density"):
        density.draw_overlay(ax, _snapshot(), timestep=0)
    plt.close(fig)
