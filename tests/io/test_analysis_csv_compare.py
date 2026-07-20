"""Branch-oriented tests for analysis CSV export and comparison helpers."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tud_lbm.config import SimulationConfig
from tud_lbm.io.analysis.droplet_metrics import DropletSeries
from tud_lbm.io.analysis.droplet_metrics import MetricScales
from tud_lbm.io.plotting.run_comparison import _clean_dir_label
from tud_lbm.io.plotting.run_comparison import _safe_load_config
from tud_lbm.io.plotting.run_comparison import process_parent_dir
from tud_lbm.io.plotting.simulation_csv import build_simulation_csv


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
        gravity_force={"force_g": 1e-6, "inclination_angle_deg": 30.0},
        initialisation={"radii": [0.25], "centres": [[0.5, 0.5]]},
        wetting_config={"advancing_ca": 100.0},
    )


def _write_snapshot(path: Path, step: int, ux: float, uy: float) -> None:
    rho = np.full((16, 12, 1, 1, 1), 0.2)
    rho[4:12, 1:8, 0, 0, 0] = 1.0

    u = np.zeros((16, 12, 1, 1, 2))
    u[:, :, 0, 0, 0] = ux
    u[:, :, 0, 0, 1] = uy

    np.savez(
        path / f"timestep_{step}.npz",
        rho=rho,
        u=u,
        ca_left=np.array(80.0),
        ca_right=np.array(95.0),
        cll_left=np.array(3.0 + step * 0.1),
        cll_right=np.array(10.0 + step * 0.1),
    )


def test_build_simulation_csv_writes_expected_columns(tmp_path: Path):
    run_dir = tmp_path / "run"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)

    _write_snapshot(data_dir, 5, ux=0.02, uy=0.01)
    _write_snapshot(data_dir, 10, ux=0.03, uy=0.01)

    out = build_simulation_csv(run_dir, _wetting_config())

    assert out is not None
    assert out.exists()
    csv_path = Path(out)
    df = pd.read_csv(csv_path)
    assert {"iteration", "Ca", "Ca_norm", "ca_left", "ca_right", "v_left"}.issubset(df.columns)


def test_build_simulation_csv_skips_unsupported_sim_type(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "data").mkdir(parents=True)
    cfg = SimulationConfig(grid_shape=(8, 8), tau=0.8, nt=2, sim_type="single_phase")

    assert build_simulation_csv(run_dir, cfg) is None


def _series(*, incl_deg: float) -> DropletSeries:
    """A two-sample series with simple, hand-checkable scales."""
    scales = MetricScales(
        rho_mean=0.6,
        sigma_lg=0.5,
        nu=0.1,
        r_zero=2.0,
        r_zero_is_fallback=False,
        offset_x=8.0,
        incl_deg=incl_deg,
        save_interval=5,
    )
    return DropletSeries(
        iteration=np.array([5, 10]),
        avg_u_x=np.array([0.01, 0.02]),
        avg_u_y=np.array([0.0, 0.0]),
        avg_x_location=np.array([2.0, 3.0]),
        cll_left=np.array([1.0, 1.5]),
        cll_right=np.array([4.0, 5.0]),
        theta_left=np.array([80.0, 81.0]),
        theta_right=np.array([95.0, 94.0]),
        cm_x=np.array([5.0, 5.5]),
        cm_y=np.array([3.0, 3.2]),
        scales=scales,
    )


def test_ca_norm_divides_by_sine_of_inclination_when_positive():
    """At 30 degrees, Ca_norm is Ca / sin(30) = Ca / 0.5."""
    series = _series(incl_deg=30.0)

    assert np.allclose(series.ca_norm, series.ca / 0.5)


def test_ca_norm_equals_ca_when_not_inclined():
    """With no inclination there is nothing to normalise against."""
    series = _series(incl_deg=0.0)

    assert np.allclose(series.ca_norm, series.ca)


def test_safe_load_config_returns_none_on_parse_error(monkeypatch, tmp_path: Path):
    bad = tmp_path / "config.toml"
    bad.write_text("bad = [", encoding="utf-8")

    def _raise_load(_self, _path):
        msg = "broken config"
        raise ValueError(msg)

    monkeypatch.setattr("tud_lbm.io.readers.TomlAdapter.load", _raise_load)

    assert _safe_load_config(bad) is None


def test_clean_dir_label_removes_numeric_prefix_and_underscores():
    assert _clean_dir_label("001_case_alpha") == "Case alpha"
    assert _clean_dir_label("plain_name") == "Plain name"


def test_process_parent_dir_skips_special_folders(monkeypatch, tmp_path: Path):
    parent = tmp_path / "runs"
    run_ok = parent / "001_valid"
    run_skip_init = parent / "init" / "002_skip"
    run_skip_cmp = parent / "comparison_analysis" / "003_skip"

    run_ok.mkdir(parents=True)
    run_skip_init.mkdir(parents=True)
    run_skip_cmp.mkdir(parents=True)

    for rd in (run_ok, run_skip_init, run_skip_cmp):
        (rd / "config.toml").write_text("[simulation_type]\ntype='single_phase'\n", encoding="utf-8")

    cfg = _wetting_config()
    monkeypatch.setattr("tud_lbm.io.plotting.run_comparison._safe_load_config", lambda _p: cfg)
    monkeypatch.setattr(
        "tud_lbm.io.plotting.run_comparison.build_simulation_csv", lambda rd, _cfg: Path(rd) / "simulation_data.csv"
    )

    called = {"n": 0}

    def _fake_compare(_parent):
        called["n"] += 1

    monkeypatch.setattr("tud_lbm.io.plotting.run_comparison.compare_runs", _fake_compare)

    n_runs, n_ok = process_parent_dir(parent)

    assert n_runs == 1
    assert n_ok == 1
    assert called["n"] == 1
