"""Coverage for tud_lbm/io/plotting/analysis.py CSV-builder internals.

Targets the 228 uncovered lines (22.4 → ~70%+) by exercising:
- _resolve_r_zero: all three paths (contact-line, init-radii, fallback 27)
- _inclination_angle_deg: from gravity_force dict, and no-force fallback
- _sigma_lg: basic computation
- _interpolate_interface: left/right interface sub-cell positions
- _ca_from_rho, _cll_from_rho: from synthetic rho fields
- _center_of_mass, _avg_x_location: from synthetic rho fields
- _backward_diff: with interval>1 and interval=1
- _collect_csv_rows: end-to-end over real npz files
- _finalize_csv_dataframe: column presence and normalisation
- build_simulation_csv: skip when wrong sim_type, skip when no data dir,
  and successful write when valid files exist
- process_parent_dir: no-run and single-run paths (mocked CSV write)
- _clean_dir_label: digit prefix stripping
- _parse_timestep_from_path: valid and invalid stems
- SimulationCsvExport.compute and .render
"""

from __future__ import annotations
from unittest.mock import patch
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from tud_lbm.config import SimulationConfig
from tud_lbm.io.plotting.analysis import SimulationCsvExport
from tud_lbm.io.plotting.analysis import _avg_x_location
from tud_lbm.io.plotting.analysis import _backward_diff
from tud_lbm.io.plotting.analysis import _ca_from_rho
from tud_lbm.io.plotting.analysis import _center_of_mass
from tud_lbm.io.plotting.analysis import _clean_dir_label
from tud_lbm.io.plotting.analysis import _cll_from_rho
from tud_lbm.io.plotting.analysis import _inclination_angle_deg
from tud_lbm.io.plotting.analysis import _interpolate_interface
from tud_lbm.io.plotting.analysis import _parse_timestep_from_path
from tud_lbm.io.plotting.analysis import _resolve_r_zero
from tud_lbm.io.plotting.analysis import _sigma_lg
from tud_lbm.io.plotting.analysis import build_simulation_csv
from tud_lbm.io.plotting.analysis import process_parent_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RHO_L = 1.0
_RHO_V = 0.2
_RHO_MEAN = 0.5 * (_RHO_L + _RHO_V)


def _droplet_rho_2d(nx: int = 30, ny: int = 10, lo: int = 8, hi: int = 22) -> np.ndarray:
    """Return a (nx, ny) rho_2d array with a liquid column from lo to hi."""
    rho = np.full((nx, ny), _RHO_V)
    rho[lo:hi, :] = _RHO_L
    return rho


def _droplet_rho_5d(nx: int = 30, ny: int = 10) -> np.ndarray:
    return _droplet_rho_2d(nx, ny)[:, :, np.newaxis, np.newaxis, np.newaxis]


def _wetting_config() -> SimulationConfig:
    return SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(30, 10),
        tau=0.8,
        nt=100,
        save_interval=10,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
        initialisation={"radii": [0.3]},
    )


# ---------------------------------------------------------------------------
# _resolve_r_zero
# ---------------------------------------------------------------------------


def test_resolve_r_zero_from_contact_line_length(tmp_path):
    rho = _droplet_rho_5d()
    npz = tmp_path / "init.npz"
    np.savez(npz, rho=rho)
    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(30, 10),
        tau=0.8,
        nt=100,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
        init_type="init_from_file",
        init_dir=str(npz),
    )
    r = _resolve_r_zero(cfg)
    assert r > 0.0


def test_resolve_r_zero_from_init_radii():
    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(30, 10),
        tau=0.8,
        nt=100,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
        initialisation={"radii": [0.3]},
    )
    r = _resolve_r_zero(cfg)
    # 0.3 * min(30, 10) = 3.0
    assert r == pytest.approx(3.0)


def test_resolve_r_zero_fallback_27():
    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(30, 10),
        tau=0.8,
        nt=100,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
        initialisation={},
    )
    assert _resolve_r_zero(cfg) == pytest.approx(27.0)


# ---------------------------------------------------------------------------
# _inclination_angle_deg
# ---------------------------------------------------------------------------


def test_inclination_angle_from_gravity_force():
    cfg = SimulationConfig(gravity_force={"force_g": 1e-6, "inclination_angle_deg": 45.0})
    assert _inclination_angle_deg(cfg) == pytest.approx(45.0)


def test_inclination_angle_defaults_zero_when_no_force():
    cfg = SimulationConfig()
    assert _inclination_angle_deg(cfg) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _sigma_lg
# ---------------------------------------------------------------------------


def test_sigma_lg_computation():
    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(8, 8),
        tau=0.8,
        nt=10,
        eos="double-well",
        kappa=0.02,
        rho_l=1.0,
        rho_v=0.5,
        interface_width=2,
    )
    expected = (2.0 / 3.0) * (0.02 / 2) * (0.5**2)
    assert _sigma_lg(cfg) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _interpolate_interface
# ---------------------------------------------------------------------------


def test_interpolate_interface_returns_pair():
    rho_2d = _droplet_rho_2d()
    xl, xr = _interpolate_interface(rho_2d[:, 1], _RHO_MEAN)
    assert xl < xr


# ---------------------------------------------------------------------------
# _ca_from_rho and _cll_from_rho
# ---------------------------------------------------------------------------


def test_ca_from_rho_returns_two_floats():
    rho_2d = _droplet_rho_2d()
    ca_l, ca_r = _ca_from_rho(rho_2d, _RHO_MEAN)
    assert isinstance(ca_l, float)
    assert isinstance(ca_r, float)
    # Both contact angles should be in a physically plausible range
    assert 0 < ca_l < 180
    assert 0 < ca_r < 180


def test_cll_from_rho_returns_left_right():
    rho_2d = _droplet_rho_2d()
    xl, xr = _cll_from_rho(rho_2d, _RHO_MEAN)
    assert xl < xr


# ---------------------------------------------------------------------------
# _center_of_mass and _avg_x_location
# ---------------------------------------------------------------------------


def test_center_of_mass_symmetric_droplet():
    rho_2d = _droplet_rho_2d(nx=30, ny=10, lo=10, hi=20)
    cm_x, cm_y = _center_of_mass(rho_2d, _RHO_MEAN)
    # Symmetric droplet centred at x≈14.5
    assert 9 < cm_x < 21
    assert 0 <= cm_y < 10


def test_avg_x_location():
    rho_2d = _droplet_rho_2d(nx=30, ny=10, lo=10, hi=20)
    avg_x = _avg_x_location(rho_2d, _RHO_MEAN, offset_x=15.0)
    # Should be close to 0 for a centred droplet with offset=15
    assert abs(avg_x) < 2.0


# ---------------------------------------------------------------------------
# _backward_diff
# ---------------------------------------------------------------------------


def test_backward_diff_interval_one():
    arr = np.array([0.0, 2.0, 4.0, 8.0])
    diff = _backward_diff(arr, 1)
    np.testing.assert_allclose(diff, [0.0, 2.0, 2.0, 4.0])


def test_backward_diff_interval_greater_than_one():
    arr = np.array([0.0, 10.0, 20.0, 30.0])
    diff = _backward_diff(arr, 10)
    # Each step is 10 position / 10 interval = 1.0
    np.testing.assert_allclose(diff[1:], [1.0, 1.0, 1.0])


def test_backward_diff_zero_interval_treated_as_one():
    arr = np.array([0.0, 3.0, 6.0])
    diff = _backward_diff(arr, 0)
    np.testing.assert_allclose(diff, [0.0, 3.0, 3.0])


# ---------------------------------------------------------------------------
# _parse_timestep_from_path
# ---------------------------------------------------------------------------


def test_parse_timestep_from_path_valid(tmp_path):
    p = tmp_path / "timestep_42.npz"
    assert _parse_timestep_from_path(p) == 42


def test_parse_timestep_from_path_invalid_returns_minus_one(tmp_path):
    p = tmp_path / "bad_name.npz"
    assert _parse_timestep_from_path(p) == -1


# ---------------------------------------------------------------------------
# _clean_dir_label
# ---------------------------------------------------------------------------


def test_clean_dir_label_strips_digit_prefix():
    assert _clean_dir_label("001_my_simulation") == "My simulation"


def test_clean_dir_label_no_digit_prefix():
    label = _clean_dir_label("my_simulation")
    assert "simulation" in label.lower()


# ---------------------------------------------------------------------------
# SimulationCsvExport — compute and render
# ---------------------------------------------------------------------------


def test_simulation_csv_export_compute_returns_empty_payload(tmp_path):
    cfg = _wetting_config()
    op = SimulationCsvExport(config=cfg)
    result = op.compute([])
    assert result["iters"].size == 0
    assert result["values"].size == 0


def test_simulation_csv_export_render_shows_placeholder():
    cfg = _wetting_config()
    op = SimulationCsvExport(config=cfg)
    fig, ax = plt.subplots()
    try:
        op.render(ax, {"iters": np.array([]), "values": np.array([])})
        assert "CSV export operator" in ax.get_title()
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# build_simulation_csv
# ---------------------------------------------------------------------------


def test_build_simulation_csv_skips_wrong_sim_type(tmp_path):
    cfg = SimulationConfig(
        sim_type="single",
        grid_shape=(8, 8),
        tau=0.8,
        nt=10,
        eos="double-well",
        kappa=0.02,
        rho_l=1.0,
        rho_v=0.5,
        interface_width=2,
    )
    result = build_simulation_csv(tmp_path, cfg)
    assert result is None


def test_build_simulation_csv_skips_missing_data_dir(tmp_path):
    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(8, 8),
        tau=0.8,
        nt=10,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
    )
    result = build_simulation_csv(tmp_path, cfg)
    assert result is None


def test_build_simulation_csv_skips_empty_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(8, 8),
        tau=0.8,
        nt=10,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
    )
    result = build_simulation_csv(tmp_path, cfg)
    assert result is None


def test_build_simulation_csv_writes_file(tmp_path):
    pytest.importorskip("pandas")

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    rho_2d = _droplet_rho_2d(nx=30, ny=10)
    rho = rho_2d[:, :, np.newaxis, np.newaxis, np.newaxis]
    u = np.zeros((30, 10, 1, 1, 2))

    for step in (10, 20, 30):
        np.savez(data_dir / f"timestep_{step}.npz", rho=rho, u=u)

    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(30, 10),
        tau=0.8,
        nt=100,
        save_interval=10,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
        initialisation={"radii": [0.3]},
    )

    result = build_simulation_csv(tmp_path, cfg)
    assert result is not None
    assert result.exists()
    assert result.suffix == ".csv"


# ---------------------------------------------------------------------------
# process_parent_dir
# ---------------------------------------------------------------------------


def test_process_parent_dir_returns_zero_zero_when_no_runs(tmp_path):
    n_runs, n_ok = process_parent_dir(tmp_path)
    assert (n_runs, n_ok) == (0, 0)


def test_process_parent_dir_skips_init_dirs(tmp_path):
    init_dir = tmp_path / "init" / "run1"
    init_dir.mkdir(parents=True)
    (init_dir / "config.toml").write_text("[simulation_type]\nsim_type='single'\n", encoding="utf-8")
    n_runs, _ = process_parent_dir(tmp_path)
    assert n_runs == 0


def test_process_parent_dir_counts_valid_run(tmp_path):
    pytest.importorskip("pandas")

    run_dir = tmp_path / "run1"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)

    rho_2d = _droplet_rho_2d()
    rho = rho_2d[:, :, np.newaxis, np.newaxis, np.newaxis]
    u = np.zeros((30, 10, 1, 1, 2))
    np.savez(data_dir / "timestep_10.npz", rho=rho, u=u)

    cfg = SimulationConfig(
        sim_type="multiphase_wetting",
        grid_shape=(30, 10),
        tau=0.8,
        nt=100,
        save_interval=10,
        eos="double-well",
        kappa=0.02,
        rho_l=_RHO_L,
        rho_v=_RHO_V,
        interface_width=2,
        initialisation={"radii": [0.3]},
        simulation_name="test_run",
    )

    # Write a config.toml so the discovery loop finds the run
    (run_dir / "config.toml").write_text("", encoding="utf-8")
    with (
        patch("tud_lbm.io.plotting.analysis._safe_load_config", return_value=cfg),
        patch("tud_lbm.io.plotting.analysis.compare_runs"),
    ):
        n_runs, _ = process_parent_dir(tmp_path)

    assert n_runs == 1
