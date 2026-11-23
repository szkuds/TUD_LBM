"""End-to-end tests for src.simulation_io.plotting.regime_map_plot."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from src.cli.analysis_routing import parse_run_dir_list
from src.config import SimulationConfig
from src.config.analysis_options import AxisScale
from src.simulation_io.analysis.accelerations import Regime
from src.simulation_io.analysis.physical_parameters import DimensionlessNumbers
from src.simulation_io.plotting.figure_config import REGIME_COLORS
from src.simulation_io.plotting.figure_config import REGIME_MARKERS
from src.simulation_io.plotting.regime_map_plot import RunRegimeEntry
from src.simulation_io.plotting.regime_map_plot import build_regime_map
from src.simulation_io.plotting.regime_map_plot import plot_regime_map
from src.simulation_io.plotting.regime_map_plot import process_run_dir
from src.simulation_io.readers import TomlAdapter
from tests.support.config import merge_overrides

if TYPE_CHECKING:
    from pathlib import Path

_NX, _NY = 60, 12


def _run_config(**kwargs) -> SimulationConfig:
    base = {
        "multiphase": {"eos": "double-well", "kappa": 0.02, "rho_l": 1.0, "rho_v": 0.2, "interface_width": 2},
        "forces": {"gravity_masked_force": {"force_g": 1e-6, "inclination_angle_deg": 30.0}},
        "grid_shape": (_NX, _NY),
        "tau": 0.9,
        "nt": 30,
        "save_interval": 10,
        "initialisation": {"radii": [0.25], "centres": [[0.5, 0.5]]},
        "wetting": {"phi_left": 1.0},
    }
    merge_overrides(base, kwargs)
    return SimulationConfig(**base)  # ty: ignore[invalid-argument-type]


def _write_snapshot(data_dir: Path, step: int, x_start: int, ux: float) -> None:
    rho = np.full((_NX, _NY, 1, 1, 1), 0.2)
    rho[x_start : x_start + 8, 1:8, 0, 0, 0] = 1.0
    u = np.zeros((_NX, _NY, 1, 1, 2))
    u[:, :, 0, 0, 0] = ux

    np.savez(
        data_dir / f"timestep_{step}.npz",
        rho=rho,
        u=u,
        ca_left=np.array(80.0),
        ca_right=np.array(95.0),
        cll_left=np.array(3.0 + step * 0.05),
        cll_right=np.array(10.0 + step * 0.05),
    )


def _build_run_dir(run_dir: Path, x_starts: list[int], config: SimulationConfig) -> None:
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    for step, x_start in zip(range(0, len(x_starts) * 10, 10), x_starts, strict=True):
        _write_snapshot(data_dir, step, x_start, ux=0.01 * (step + 1))
    TomlAdapter().save(config, str(run_dir / "config.toml"))


def test_process_run_dir_classifies_pinned_run(tmp_path: Path):
    run_dir = tmp_path / "pinned_run"
    _build_run_dir(run_dir, [4, 4, 4, 4], _run_config())

    entry = process_run_dir(run_dir)

    assert entry is not None
    assert entry.regime == "Pinning"
    assert (run_dir / "plots" / "analysis" / "acceleration_analysis.png").exists()


def test_process_run_dir_classifies_mobile_run(tmp_path: Path):
    run_dir = tmp_path / "mobile_run"
    _build_run_dir(run_dir, [4, 10, 30, 48], _run_config())

    entry = process_run_dir(run_dir)

    assert entry is not None
    assert entry.regime != "pinning"
    assert (run_dir / "plots" / "analysis" / "acceleration_analysis.png").exists()


def test_process_run_dir_none_for_missing_config(tmp_path: Path):
    run_dir = tmp_path / "no_config"
    run_dir.mkdir()

    assert process_run_dir(run_dir) is None


def test_calibration_only_eos_classifies_but_resolves_no_numbers(tmp_path: Path):
    """Classification no longer depends on the axes, so the gate moved to plotting.

    An uncalibrated Carnahan-Starling run has no surface tension and therefore no
    dimensionless numbers at all; it still classifies, and is dropped when the
    figure is drawn rather than when the snapshots are read.
    """
    run_dir = tmp_path / "no_dimensionless"
    cfg = _run_config(multiphase={"eos": "carnahan-starling", "a_eos": 1.0, "b_eos": 4.0, "r_eos": 1.0, "t_eos": 0.07})
    _build_run_dir(run_dir, [4, 4, 4, 4], cfg)

    entry = process_run_dir(run_dir)

    assert entry is not None
    assert entry.numbers.get("oh") is None
    assert plot_regime_map([entry], tmp_path / "out.png") is None


def test_regime_style_tables_cover_every_regime():
    """figure_config's style tables are keyed by the Regime enum's own values."""
    assert set(REGIME_MARKERS) == {regime.value for regime in Regime}
    assert set(REGIME_COLORS) == {regime.value for regime in Regime}


def _entry(tmp_path: Path, name: str, regime: str, **values: float | None) -> RunRegimeEntry:
    return RunRegimeEntry(
        run_dir=tmp_path / name,
        label=name,
        numbers=DimensionlessNumbers(values=values),
        regime=regime,
    )


def test_plot_regime_map_writes_file_with_all_regime_markers(tmp_path: Path):
    entries = [
        _entry(tmp_path, "a", "Pinning", bo_parallel=1.0, oh=0.1),
        _entry(tmp_path, "b", "Dissipative", bo_parallel=2.0, oh=0.2),
        _entry(tmp_path, "c", "Capillary", bo_parallel=3.0, oh=0.3),
        _entry(tmp_path, "d", "Steady", bo_parallel=4.0, oh=0.4),
        _entry(tmp_path, "e", "unknown", bo_parallel=5.0, oh=0.5),
    ]
    assert set(REGIME_MARKERS) == {e.regime for e in entries}
    assert set(REGIME_COLORS) == {e.regime for e in entries}

    out_path = plot_regime_map(entries, tmp_path / "regime_map.png")

    assert out_path is not None
    assert out_path.exists()


def test_plot_regime_map_drops_runs_missing_an_axis(tmp_path: Path):
    """A run can classify fine yet lack the number an axis asks for."""
    entries = [
        _entry(tmp_path, "a", "Pinning", bo_parallel=1.0, oh=0.1, la=100.0),
        _entry(tmp_path, "b", "Capillary", bo_parallel=2.0, oh=0.2, la=None),
    ]

    assert plot_regime_map(entries, tmp_path / "la.png", x_key="la", y_key="oh") is not None


def test_plot_regime_map_returns_none_when_no_run_resolves_the_axes(tmp_path: Path):
    entries = [_entry(tmp_path, "a", "Pinning", bo_parallel=1.0, oh=0.1, ar=None)]

    assert plot_regime_map(entries, tmp_path / "ar.png", x_key="ar", y_key="oh") is None


def test_build_regime_map_end_to_end(tmp_path: Path):
    _build_run_dir(tmp_path / "pinned_run", [4, 4, 4, 4], _run_config())
    _build_run_dir(tmp_path / "mobile_run", [4, 10, 30, 48], _run_config())
    txt_path = tmp_path / "dirs.txt"
    txt_path.write_text(
        "# comparison run list\n\npinned_run\nmobile_run\n",
        encoding="utf-8",
    )

    out_path = build_regime_map(parse_run_dir_list(txt_path, allowed_roots=[tmp_path]), tmp_path / "regime_map")

    assert out_path is not None
    assert out_path.exists()
    assert out_path.parent == tmp_path / "regime_map"
    assert out_path.name == "regime_map_bo_parallel_vs_oh.png"


def test_build_regime_map_names_the_figure_after_the_axis_pair(tmp_path: Path):
    """Each pair gets its own file, so plotting a second pair does not clobber the first."""
    _build_run_dir(tmp_path / "mobile_run", [4, 10, 30, 48], _run_config())
    out_path = build_regime_map(
        [tmp_path / "mobile_run"], tmp_path / "regime_map", x_key="la", y_key="bo", xscale=AxisScale.LOG
    )

    assert out_path is not None
    assert out_path.name == "regime_map_la_vs_bo.png"
    assert out_path.exists()


def test_build_regime_map_none_when_no_runs_usable(tmp_path: Path):
    assert build_regime_map([tmp_path / "missing_run"], tmp_path / "regime_map") is None
