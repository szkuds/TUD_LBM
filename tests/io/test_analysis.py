"""Coverage boost for tud_lbm/io/plotting/analysis.py.

Targets the uncovered paths in compare_runs, _load_comparison_entries,
the analysis.main() entry point, and render paths for operators with
and without data.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from unittest.mock import patch
import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from tud_lbm.config import SimulationConfig
from tud_lbm.io.plotting.analysis import _load_comparison_entries
from tud_lbm.io.plotting.analysis import _set_empty_state
from tud_lbm.io.plotting.analysis import build_simulation_csv
from tud_lbm.io.plotting.analysis import compare_runs
from tud_lbm.io.plotting.analysis import main
from tud_lbm.io.plotting.analysis import process_parent_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RHO_L = 1.0
_RHO_V = 0.2
_RHO_MEAN = 0.5 * (_RHO_L + _RHO_V)


def _wetting_cfg(**extra) -> SimulationConfig:
    base = {
        "sim_type": "multiphase_wetting",
        "grid_shape": (30, 10),
        "tau": 0.8,
        "nt": 100,
        "save_interval": 10,
        "eos": "double-well",
        "kappa": 0.02,
        "rho_l": _RHO_L,
        "rho_v": _RHO_V,
        "interface_width": 2,
    }
    base.update(extra)
    return SimulationConfig(**base)  # ty: ignore[invalid-argument-type]


def _write_snapshot(data_dir: Path, step: int) -> None:
    rho = np.full((30, 10, 1, 1, 1), _RHO_V)
    rho[8:22, :, 0, 0, 0] = _RHO_L
    u = np.zeros((30, 10, 1, 1, 2))
    np.savez(
        data_dir / f"timestep_{step}.npz",
        rho=rho,
        u=u,
        ca_left=np.array(80.0),
        ca_right=np.array(95.0),
        cll_left=np.array(3.0),
        cll_right=np.array(10.0),
    )


def _minimal_csv_df() -> pd.DataFrame:
    """Create a minimal DataFrame with all columns expected by compare_runs plots."""
    n = 3
    return pd.DataFrame(
        {
            "iteration": [10, 20, 30],
            "normalised_iteration": [0.33, 0.67, 1.0],
            "avg_x_location": [1.0, 2.0, 3.0],
            "avg_x_location_norm": [0.1, 0.2, 0.3],
            "avg_u_x": [0.01, 0.02, 0.03],
            "avg_u_y": [0.0, 0.0, 0.0],
            "cll_left": [3.0, 3.5, 4.0],
            "cll_right": [10.0, 10.5, 11.0],
            "v_left": [0.0, 0.05, 0.05],
            "v_right": [0.0, 0.05, 0.05],
            "v_cm": [0.0, 0.01, 0.01],
            "ca_left": [80.0] * n,
            "ca_right": [95.0] * n,
            "cm_x": [15.0] * n,
            "cm_y": [5.0] * n,
            "Ca": [0.002, 0.004, 0.006],
            "Ca_cll_left": [0.001, 0.002, 0.003],
            "Ca_cll_right": [0.001, 0.002, 0.003],
            "Ca_cm": [0.001, 0.001, 0.001],
            "Ca_norm": [0.003, 0.006, 0.009],
        }
    )


# ---------------------------------------------------------------------------
# compare_runs — actual execution with synthetic entries
# ---------------------------------------------------------------------------


class TestCompareRunsActual:
    """Test compare_runs with real (synthetic) DataFrame entries."""

    def test_compare_runs_no_entries_prints_message(self, tmp_path, capsys):
        """compare_runs with no valid CSV files should print a message and return."""
        compare_runs(tmp_path)
        captured = capsys.readouterr()
        assert "No processed simulation data found" in captured.out

    def test_compare_runs_creates_output_plots(self, tmp_path):
        """compare_runs with one synthetic entry should produce 8 PNG files."""
        pytest.importorskip("pandas")

        entries = [{"label": "test_run", "sort_key": 100, "data": _minimal_csv_df()}]
        with patch("tud_lbm.io.plotting.analysis._load_comparison_entries", return_value=entries):
            compare_runs(tmp_path)

        out_dir = tmp_path / "comparison_analysis"
        assert out_dir.exists()
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) == 8

    def test_compare_runs_with_multiple_entries(self, tmp_path):
        """compare_runs should handle multiple runs without errors."""
        pytest.importorskip("pandas")

        entries = [
            {"label": "run_a", "sort_key": 90, "data": _minimal_csv_df()},
            {"label": "run_b", "sort_key": 100, "data": _minimal_csv_df()},
        ]
        with patch("tud_lbm.io.plotting.analysis._load_comparison_entries", return_value=entries):
            compare_runs(tmp_path)

        pngs = list((tmp_path / "comparison_analysis").glob("*.png"))
        assert len(pngs) == 8

    def test_compare_runs_skips_missing_x_column(self, tmp_path):
        """If x column is missing in a run's data, that run is skipped gracefully."""
        pytest.importorskip("pandas")

        df_no_x = _minimal_csv_df().drop(columns=["normalised_iteration"])
        entries = [{"label": "broken_run", "sort_key": 90, "data": df_no_x}]
        with patch("tud_lbm.io.plotting.analysis._load_comparison_entries", return_value=entries):
            compare_runs(tmp_path)  # must not raise

    def test_compare_runs_skips_missing_y_column(self, tmp_path):
        """If y column is absent in a single-y plot, that run is skipped gracefully."""
        pytest.importorskip("pandas")

        df_no_y = _minimal_csv_df().drop(columns=["Ca"])
        entries = [{"label": "no_Ca", "sort_key": 90, "data": df_no_y}]
        with patch("tud_lbm.io.plotting.analysis._load_comparison_entries", return_value=entries):
            compare_runs(tmp_path)  # must not raise


# ---------------------------------------------------------------------------
# _load_comparison_entries
# ---------------------------------------------------------------------------


class TestLoadComparisonEntries:
    """Tests for _load_comparison_entries CSV discovery."""

    def test_returns_empty_when_no_csvs(self, tmp_path):
        pytest.importorskip("pandas")
        result = _load_comparison_entries(tmp_path)
        assert result == []

    def test_skips_comparison_analysis_dir(self, tmp_path):
        pytest.importorskip("pandas")

        # Put a CSV inside comparison_analysis subdirectory - should be skipped
        cmp_dir = tmp_path / "comparison_analysis" / "run1"
        cmp_dir.mkdir(parents=True)
        (cmp_dir / "simulation_data.csv").write_text("iteration,Ca\n1,0.001\n", encoding="utf-8")

        result = _load_comparison_entries(tmp_path)
        assert result == []

    def test_skips_run_without_config_toml(self, tmp_path):
        pytest.importorskip("pandas")

        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "simulation_data.csv").write_text("iteration,Ca\n1,0.001\n", encoding="utf-8")
        # No config.toml → should be skipped

        result = _load_comparison_entries(tmp_path)
        assert result == []

    def test_loads_entry_when_config_toml_present(self, tmp_path):
        pytest.importorskip("pandas")

        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "simulation_data.csv").write_text("iteration,Ca\n10,0.002\n20,0.004\n", encoding="utf-8")
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")

        cfg = _wetting_cfg(simulation_name="my_run", wetting_config={"advancing_ca": 100})
        with patch("tud_lbm.io.plotting.analysis._safe_load_config", return_value=cfg):
            result = _load_comparison_entries(tmp_path)

        assert len(result) == 1
        assert result[0]["label"] == "my_run"
        assert result[0]["sort_key"] == 100

    def test_uses_dir_name_when_simulation_name_missing(self, tmp_path):
        pytest.importorskip("pandas")

        run_dir = tmp_path / "001_test_case"
        run_dir.mkdir()
        (run_dir / "simulation_data.csv").write_text("iteration,Ca\n10,0.002\n", encoding="utf-8")
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")

        cfg = _wetting_cfg()  # no simulation_name
        with patch("tud_lbm.io.plotting.analysis._safe_load_config", return_value=cfg):
            result = _load_comparison_entries(tmp_path)

        assert len(result) == 1
        # Label derived from dir name (digit prefix stripped)
        assert "test" in result[0]["label"].lower()

    def test_skips_run_when_config_fails_to_load(self, tmp_path):
        pytest.importorskip("pandas")

        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "simulation_data.csv").write_text("iteration,Ca\n", encoding="utf-8")
        (run_dir / "config.toml").write_text("bad=[", encoding="utf-8")

        with patch("tud_lbm.io.plotting.analysis._safe_load_config", return_value=None):
            result = _load_comparison_entries(tmp_path)

        assert result == []


class TestAnalysisMain:
    """Tests for the analysis.main() CLI entry point."""

    def test_main_exits_1_for_missing_directory(self):
        with pytest.raises(SystemExit) as exc:
            main("/definitely/not/a/real/path")
        assert exc.value.code == 1

    def test_main_exits_1_for_empty_directory(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            main(str(tmp_path))
        assert exc.value.code == 1

    def test_main_exits_1_when_no_runs_produce_csv(self, tmp_path):
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")

        cfg = _wetting_cfg(sim_type="single_phase")  # unsupported type → no CSV
        with (
            patch("tud_lbm.io.plotting.analysis._safe_load_config", return_value=cfg),
            pytest.raises(SystemExit) as exc,
        ):
            main(str(tmp_path))

        assert exc.value.code == 1

    def test_main_succeeds_when_csv_is_produced(self, tmp_path):
        pytest.importorskip("pandas")

        run_dir = tmp_path / "run1"
        data_dir = run_dir / "data"
        data_dir.mkdir(parents=True)
        _write_snapshot(data_dir, 10)
        _write_snapshot(data_dir, 20)
        (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")

        cfg = _wetting_cfg(simulation_name="run1")
        with (
            patch("tud_lbm.io.plotting.analysis._safe_load_config", return_value=cfg),
            patch("tud_lbm.io.plotting.analysis.compare_runs"),
        ):
            main(str(tmp_path))  # should not raise SystemExit


# ---------------------------------------------------------------------------
# process_parent_dir — n_ok > 0 path (compare_runs is actually called)
# ---------------------------------------------------------------------------


def test_process_parent_dir_calls_compare_runs_when_csv_produced(tmp_path):
    pytest.importorskip("pandas")

    run_dir = tmp_path / "run1"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    _write_snapshot(data_dir, 10)
    _write_snapshot(data_dir, 20)
    (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")

    cfg = _wetting_cfg()
    compare_called = {"n": 0}

    def _fake_compare(_parent_dir):
        compare_called["n"] += 1

    with (
        patch("tud_lbm.io.plotting.analysis._safe_load_config", return_value=cfg),
        patch("tud_lbm.io.plotting.analysis.compare_runs", side_effect=_fake_compare),
    ):
        n_runs, n_ok = process_parent_dir(tmp_path)

    assert n_runs == 1
    assert n_ok == 1
    assert compare_called["n"] == 1


def test_process_parent_dir_does_not_call_compare_when_all_fail(tmp_path):
    pytest.importorskip("pandas")

    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "config.toml").write_text("[simulation_type]\n", encoding="utf-8")

    cfg = _wetting_cfg(sim_type="single_phase")
    compare_called = {"n": 0}

    with patch("tud_lbm.io.plotting.analysis._safe_load_config", return_value=cfg):
        n_runs, n_ok = process_parent_dir(tmp_path)

    assert n_runs == 1
    assert n_ok == 0
    assert compare_called["n"] == 0


# ---------------------------------------------------------------------------
# _set_empty_state — direct coverage
# ---------------------------------------------------------------------------


def test_set_empty_state_without_required_keys():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        _set_empty_state(ax, title="Test", ylabel="Y")
        assert ax.get_title() == "Test"
    finally:
        plt.close(fig)


def test_set_empty_state_with_required_keys():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        _set_empty_state(ax, title="Test", ylabel="Y", required_keys=("rho", "u"))
        texts = [t.get_text() for t in ax.texts]
        assert any("rho" in t for t in texts)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# build_simulation_csv — pandas import missing path
# ---------------------------------------------------------------------------


def test_build_simulation_csv_returns_none_for_unsupported_sim_type(tmp_path):
    """build_simulation_csv returns None for non-wetting sim types."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert build_simulation_csv(run_dir, _wetting_cfg(sim_type="single_phase")) is None
