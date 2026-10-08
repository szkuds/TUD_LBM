"""The setup figure: snapshot selection, geometry and the figure it writes."""

from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib as mpl
import numpy as np
import pytest
from src.config.run_config import ANALYSIS_DIRNAME
from src.config.run_config import PLOTS_DIRNAME
from src.config.run_config import SETUP_FIGURE_FILENAME
from src.config.run_config import SIMULATION_CSV_FILENAME
from src.simulation_io.plotting import setup_figure
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE
from src.simulation_io.plotting.setup_figure import edge_labels
from src.simulation_io.plotting.setup_figure import key_text
from src.simulation_io.plotting.setup_figure import page_rotation_deg
from src.simulation_io.plotting.setup_figure import rotate
from src.simulation_io.plotting.setup_figure import select_setup_timesteps
from src.simulation_io.plotting.setup_figure import write_setup_figure
from tests.support.run_dirs import GRID_NX
from tests.support.run_dirs import UNIFORM_ITERATIONS
from tests.support.run_dirs import build_run_dir
from tests.support.run_dirs import wetting_config

if TYPE_CHECKING:
    from pathlib import Path
    from src.config import SimulationConfig

mpl.use("Agg")

_STEP_LOCATION = 0.75
_CSV_HEADER = "iteration,cll_left,cll_right,ca_left,ca_right"


def step_config(**overrides: object) -> SimulationConfig:
    """A chemical-step run on an incline: 120/100 before the step, 70/50 after."""
    params: dict[str, object] = {
        "sim_type": "multiphase_hysteresis_chemical_step",
        "wetting_config": {"phi_left": 1.0, "phi_right": 1.0, "d_rho_left": 0.0, "d_rho_right": 0.0},
        "chemical_step_config": {
            "chemical_step_location": _STEP_LOCATION,
            "ca_advancing_pre_step": 120.0,
            "ca_receding_pre_step": 100.0,
            "ca_advancing_post_step": 70.0,
            "ca_receding_post_step": 50.0,
        },
        "bc_config": {"bottom": "wetting", "top": "symmetry"},
        "plot_fields": ["density"],
    }
    params.update(overrides)
    return wetting_config(**params)


def _write_csv(run_dir: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    lines = [_CSV_HEADER, *(",".join(str(value) for value in row) for row in rows)]
    (run_dir / SIMULATION_CSV_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render(config: SimulationConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: object):
    """Write the figure and hand back the live matplotlib figure it was built on."""
    import matplotlib.pyplot as plt

    captured = []
    monkeypatch.setattr(plt, "close", captured.append)
    run_dir = build_run_dir(tmp_path, config=config)
    path = write_setup_figure(config, run_dir, **kwargs)  # ty: ignore[invalid-argument-type]
    return path, captured[0]


def _lines(fig, gid: str) -> list:
    return [line for line in fig.axes[0].lines if line.get_gid() == gid]


# --- Snapshot selection ------------------------------------------------------


def test_default_pair_is_hysteresis_snapshot_and_last(tmp_path):
    config = step_config()
    run_dir = build_run_dir(tmp_path, config=config)
    _write_csv(
        run_dir,
        [
            (5, 3.0, 8.0, 110.0, 111.0),  # neither bound
            (10, 3.0, 8.0, 106.0, 119.5),  # advancing only: the rear line is still relaxing
            (15, 3.0, 8.0, 100.5, 119.2),  # both bounds, within tolerance
            (20, 9.0, 14.0, 100.0, 70.0),
        ],
    )

    choice = select_setup_timesteps(config, run_dir)

    assert choice.timesteps == (15, 20)
    assert "receding" in choice.reason


def test_rows_past_the_step_do_not_qualify(tmp_path):
    config = step_config()
    run_dir = build_run_dir(tmp_path, config=config)
    past_step = _STEP_LOCATION * GRID_NX + 1.0
    _write_csv(run_dir, [(5, 3.0, past_step, 100.0, 120.0), (10, 3.0, 8.0, 100.0, 120.0)])

    assert select_setup_timesteps(config, run_dir).timesteps == (10, UNIFORM_ITERATIONS[-1])


def test_falls_back_to_advancing_bound_alone(tmp_path):
    config = step_config()
    run_dir = build_run_dir(tmp_path, config=config)
    _write_csv(run_dir, [(5, 3.0, 8.0, 110.0, 111.0), (10, 3.0, 8.0, 106.0, 119.5)])

    choice = select_setup_timesteps(config, run_dir)

    assert choice.timesteps == (10, UNIFORM_ITERATIONS[-1])
    assert "never" in choice.reason


@pytest.mark.parametrize("rows", [None, [(5, 3.0, 8.0, 90.0, 90.0)]])
def test_falls_back_to_first_snapshot(tmp_path, rows):
    config = step_config()
    run_dir = build_run_dir(tmp_path, config=config)
    if rows is not None:
        _write_csv(run_dir, rows)

    choice = select_setup_timesteps(config, run_dir)

    assert choice.timesteps == (UNIFORM_ITERATIONS[0], UNIFORM_ITERATIONS[-1])
    assert "first snapshot" in choice.reason


def test_single_surface_wall_draws_only_the_last_snapshot(tmp_path):
    config = wetting_config()
    run_dir = build_run_dir(tmp_path, config=config)

    assert select_setup_timesteps(config, run_dir).timesteps == (UNIFORM_ITERATIONS[-1],)


def test_no_snapshots_selects_nothing(tmp_path):
    assert select_setup_timesteps(step_config(), tmp_path).timesteps == ()


# --- Geometry and labels -----------------------------------------------------


@pytest.mark.parametrize("draw_angle", [5.0, 15.0, 40.0])
@pytest.mark.parametrize("inclination", [30.0, -30.0])
def test_rotation_puts_schematic_gravity_straight_down(draw_angle, inclination):
    config = step_config(gravity_force={"force_g": 1e-6, "inclination_angle_deg": inclination})
    schematic = np.radians(np.copysign(draw_angle, inclination))
    gravity = np.array([[np.sin(schematic), -np.cos(schematic)]])

    on_page = rotate(gravity, page_rotation_deg(config, draw_angle))

    np.testing.assert_allclose(on_page, [[0.0, -1.0]], atol=1e-12)


def test_flat_or_gravity_free_run_is_not_rotated():
    flat = step_config(gravity_force={"force_g": 1e-6, "inclination_angle_deg": 0.0})

    assert page_rotation_deg(flat, 15.0) == 0.0
    assert page_rotation_deg(step_config(gravity_force=None), 15.0) == 0.0


def test_edge_labels_come_from_the_config():
    labels = edge_labels(step_config())

    assert labels == {"bottom": "NS+W", "top": "FS", "left": "PBC", "right": "PBC"}


def test_unknown_boundary_name_labels_itself(monkeypatch):
    monkeypatch.delitem(setup_figure.BC_ABBREVIATIONS, "symmetry")

    assert edge_labels(step_config())["top"] == "symmetry"


def test_key_lists_both_surface_windows():
    key = key_text(step_config())

    assert "PBC: periodic" in key
    assert r"$W_1$: $\theta_A=120^\circ$, $\theta_R=100^\circ$" in key
    assert r"$W_2$: $\theta_A=70^\circ$, $\theta_R=50^\circ$" in key


# --- The figure --------------------------------------------------------------


def test_writes_pdf_with_one_red_interface_per_snapshot(tmp_path, monkeypatch):
    path, fig = _render(step_config(), tmp_path, monkeypatch)

    assert path == tmp_path / "run" / PLOTS_DIRNAME / ANALYSIS_DIRNAME / SETUP_FIGURE_FILENAME
    assert path.stat().st_size > 0
    interfaces = _lines(fig, "interface")
    assert len(interfaces) == len({UNIFORM_ITERATIONS[0], UNIFORM_ITERATIONS[-1]})
    assert {line.get_color() for line in interfaces} == {DEFAULT_STYLE.setup_interface_color}
    assert {line.get_linestyle() for line in interfaces} == {"-"}


def test_explicit_timestep_draws_one_interface(tmp_path, monkeypatch):
    _, fig = _render(step_config(), tmp_path, monkeypatch, timesteps=[UNIFORM_ITERATIONS[1]])

    assert len(_lines(fig, "interface")) == 1


def test_wall_is_split_into_two_surfaces_at_the_step(tmp_path, monkeypatch):
    config = step_config(gravity_force=None)
    _, fig = _render(config, tmp_path, monkeypatch)

    patches = {patch.get_gid(): patch for patch in fig.axes[0].patches}

    assert set(patches) == {"pre", "post"}
    wall = config.chemical_step_wall
    assert wall is not None
    step_x = wall.step_x
    assert patches["pre"].get_xy()[:, 0].max() == pytest.approx(step_x)
    assert patches["post"].get_xy()[:, 0].min() == pytest.approx(step_x)
    assert patches["pre"].get_facecolor() != patches["post"].get_facecolor()


def test_wall_without_a_step_is_one_surface(tmp_path, monkeypatch):
    _, fig = _render(wetting_config(), tmp_path, monkeypatch)

    assert [patch.get_gid() for patch in fig.axes[0].patches] == ["plain"]


def test_gravity_free_run_has_no_arrow_and_no_incline(tmp_path, monkeypatch):
    path, fig = _render(step_config(gravity_force=None), tmp_path, monkeypatch)

    assert path.is_file()
    assert not _lines(fig, "alpha")
    assert not [text for text in fig.axes[0].texts if text.get_gid() == "gravity"]


def test_inclined_run_draws_gravity_and_alpha(tmp_path, monkeypatch):
    _, fig = _render(step_config(), tmp_path, monkeypatch)

    assert len(_lines(fig, "alpha")) == 1
    (arrow,) = [text for text in fig.axes[0].texts if text.get_gid() == "gravity"]
    assert arrow.xy[0] == pytest.approx(arrow.xyann[0])
    assert arrow.xy[1] < arrow.xyann[1]


def test_missing_timestep_raises(tmp_path):
    config = step_config()
    run_dir = build_run_dir(tmp_path, config=config)

    with pytest.raises(ValueError, match="No snapshot"):
        write_setup_figure(config, run_dir, timesteps=[999])


def test_run_without_snapshots_writes_nothing(tmp_path):
    assert write_setup_figure(step_config(), tmp_path) is None
